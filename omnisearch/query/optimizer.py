"""Cost-based optimizer: LogicalPlan -> PhysicalPlan.

Decisions, each made by comparing estimated costs:
  1. filter - estimated selectivity from histograms / bitmap counts, and a
     column-scan vs B+-tree access path per range predicate (in filters.py);
  2. each SIMILAR clause - exact pre-filtered scan, ANN with the filter pushed
     in, or ANN + post-filter with over-fetching;
  3. each MATCH clause - term-at-a-time BM25 or WAND;
  4. fusion - materialise every list to a fixed depth ("full"), or stream
     them through the Threshold Algorithm / NRA (score-based fusion only).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from omnisearch.query import filters as flt
from omnisearch.query.cost import (
    Alternative, keyword_alternatives, multivector_alternatives, vector_alternatives,
)
from omnisearch.query.planner import LogicalPlan, RankedClause

DEFAULT_DEPTH = 100


@dataclass
class FilterPlan:
    expr: object
    est_sel: float
    access: str = "auto"
    actual: dict = field(default_factory=dict)


@dataclass
class BranchPlan:
    clause: RankedClause
    operator: str
    strategy: str
    params: dict
    est_cost: float
    alternatives: list
    k: int
    terms: list | None = None
    query: object = None
    actual: dict = field(default_factory=dict)


@dataclass
class FusePlan:
    method: str
    algorithm: str  # single | full | ta | nra | exhaustive | none
    depth: int
    params: dict
    weights: dict
    est_cost: float
    alternatives: list
    actual: dict = field(default_factory=dict)


@dataclass
class PhysicalPlan:
    logical: LogicalPlan
    filter: FilterPlan | None
    branches: list
    fuse: FusePlan
    limit: int
    n_docs: int

    @property
    def est_cost(self) -> float:
        return self.fuse.est_cost


def _vector_label(fidx, strategy, filtered):
    from omnisearch.index.vector.hnsw import HNSWIndex

    quant = f" over {fidx.quant.name} codes + rerank" if fidx.quant is not None and fidx.quant.trained else ""
    if strategy == "exact":
        return ("Pre-filter + " if filtered else "") + "Exact Vector Scan" + quant
    if isinstance(fidx.ann, HNSWIndex):
        if strategy == "postfilter":
            return "HNSW Search + Post-filter"
        return "HNSW Search (in-graph filter)" if filtered else "HNSW Search"
    name = "IVF" + (f"-{fidx.quant.name.upper()}" if quant else "")
    return f"{name} Probe" + (" (masked candidates)" if filtered else "")


def _choose(alts: list[Alternative], forced: str | None):
    if forced and forced != "auto":
        for a in alts:
            if a.strategy == forced:
                return a
    return min(alts, key=lambda a: a.cost)


def plan_branch(coll, clause: RankedClause, k: int, sel: float, filtered: bool, opts: dict, terms=None) -> BranchPlan:
    cm = coll.cost
    o = {**opts, **{key: v for key, v in clause.params.items() if key in ("ef", "nprobe", "overfetch", "strategy")}}
    if clause.kind == "keyword":
        alts = keyword_alternatives(cm, coll.text[clause.field], terms or [], k, o)
        a = _choose(alts, o.get("keyword_algo"))
        label = "BM25 WAND (document-at-a-time)" if a.strategy == "wand" else "BM25 Scan (term-at-a-time)"
    elif clause.kind == "vector":
        fidx = coll.vectors[clause.field]
        fidx.build()
        alts = vector_alternatives(cm, fidx, k, sel, filtered, o)
        a = _choose(alts, o.get("strategy"))
        label = _vector_label(fidx, a.strategy, filtered)
    else:
        mv = coll.multivectors[clause.field]
        mv.build()
        alts = multivector_alternatives(cm, mv, k, sel, filtered, o)
        a = _choose(alts, o.get("strategy"))
        label = "MaxSim HNSW Search" if a.strategy == "index" else "MaxSim Scan"
    others = [x for x in alts if x is not a]
    params = dict(a.params)
    if clause.kind == "keyword":
        params["df"] = {t: coll.text[clause.field].df(t) for t, _ in terms or []}
    return BranchPlan(clause, label, a.strategy, params, a.cost, others, k, terms=terms)


def optimize(logical: LogicalPlan, coll, bound: dict) -> PhysicalPlan:
    from omnisearch.query.fusion import auto_weights

    opts = logical.options
    n = coll.capacity
    alive_frac = coll.n_alive / n if n else 1.0
    fplan = None
    sel = 1.0
    if logical.filter is not None:
        sel = flt.estimate(logical.filter, coll)
        fplan = FilterPlan(logical.filter, sel, opts.get("filter_access", "auto"))
    filtered = fplan is not None or coll.has_deletions
    eff_sel = sel * alive_frac

    limit = logical.limit
    clauses = logical.ranked
    m = len(clauses)
    method = logical.fuse.method
    params = dict(logical.fuse.params)
    if method == "rrf":
        params.setdefault("k", 60)
    if method == "weighted":
        params.setdefault("norm", "max")
    depth = max(limit, int(opts.get("depth") or params.pop("depth", DEFAULT_DEPTH)))
    terms_of = {c.name: bound.get(c.name, {}).get("terms") for c in clauses}

    def branches_for(k):
        return [plan_branch(coll, c, k, eff_sel, filtered, opts, terms_of[c.name]) for c in clauses]

    cm = coll.cost
    if m == 0:
        algo, branches, cost, alts = "none", [], cm.op_overhead + n * cm.col_row, []
    elif m == 1:
        branches = branches_for(limit)
        algo, cost, alts = "single", branches[0].est_cost, []
    else:
        cands = {}
        full = branches_for(depth)
        cands["full"] = (full, sum(b.est_cost for b in full) + cm.random_access * m * depth * (m - 1))
        if method == "weighted" and params["norm"] in ("max", "none"):
            ta_depth = min(n, 5 * limit)
            ta = branches_for(ta_depth)
            cands["ta"] = (ta, sum(b.est_cost for b in ta) * 2 + cm.random_access * m * (m - 1) * ta_depth
                           + cm.sorted_access * m * ta_depth)
            nra_depth = min(n, 10 * limit)
            nb = branches_for(nra_depth)
            cands["nra"] = (nb, sum(b.est_cost for b in nb) * 2 + cm.sorted_access * m * nra_depth * 4)
        forced = opts.get("topk", "auto")
        if forced == "exhaustive":
            exh = branches_for(n)
            cands["exhaustive"] = (exh, sum(b.est_cost for b in exh))
        if forced in cands:
            algo = forced
        else:
            algo = min(cands, key=lambda a: cands[a][1])
        branches, cost = cands[algo]
        alts = [Alternative(a, c) for a, (_, c) in cands.items() if a != algo]

    weights = auto_weights(clauses, bound, coll) if method == "auto" else {c.name: c.weight for c in clauses}
    fuse = FusePlan(method, algo, depth, params, weights, cost, alts)
    return PhysicalPlan(logical, fplan, branches, fuse, limit, n)
