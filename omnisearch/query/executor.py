"""Query execution: bind -> optimize -> filter -> retrieve -> fuse -> fetch."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from omnisearch.query import fusion
from omnisearch.query.filters import FilterEvaluator
from omnisearch.query.optimizer import BranchPlan, PhysicalPlan, optimize
from omnisearch.query.planner import LogicalPlan
from omnisearch.query.topk import Source, nra, threshold_algorithm


@dataclass
class Hit:
    id: str
    doc: int
    score: float
    sources: dict
    record: dict
    moment: float | None = None
    matched_terms: list = field(default_factory=list)

    def to_dict(self):
        return {"id": self.id, "doc": self.doc, "score": self.score, "sources": self.sources,
                "record": self.record, "moment": self.moment, "matched_terms": self.matched_terms}


@dataclass
class QueryResult:
    hits: list
    plan: PhysicalPlan
    timings: dict
    stats: dict = field(default_factory=dict)

    def explain(self) -> str:
        from omnisearch.query.explain import render

        return render(self)

    def to_dict(self):
        from omnisearch.query.explain import plan_to_dict

        return {"hits": [h.to_dict() for h in self.hits], "timings": self.timings,
                "stats": self.stats, "plan": plan_to_dict(self), "explain": self.explain()}


def bind(coll, logical: LogicalPlan, encode: bool = True) -> dict:
    """Resolve clause inputs: analyzed terms for MATCH, query vectors for SIMILAR."""
    out = {}
    for c in logical.ranked:
        if c.kind == "keyword":
            inv = coll.text[c.field]
            out[c.name] = {"terms": inv.query_terms(c.node.text),
                           "unknown": [t for t in inv.analyzer(c.node.text) if t not in inv.postings]}
        else:
            out[c.name] = {"query": coll.encode_query(c.field, c.node.items) if encode else None}
    return out


class _Runner:
    def __init__(self, coll, allow):
        self.coll = coll
        self.allow = allow

    def run(self, b: BranchPlan, k: int):
        c = b.clause
        if c.kind == "keyword":
            algo = "wand" if b.strategy == "wand" else "exhaustive"
            return self.coll.text[c.field].search(b.terms, k, allow=self.allow, algo=algo)
        if c.kind == "vector":
            p = dict(b.params)
            ef = max(p.get("ef") or 0, k) or None
            return self.coll.vectors[c.field].search(
                b.query, k, allow=self.allow, strategy=b.strategy, ef=ef,
                nprobe=p.get("nprobe"), overfetch=max(p.get("overfetch") or 0, k) if b.strategy == "postfilter" else None)
        mv = self.coll.multivectors[c.field]
        return mv.search(b.query, k, allow=self.allow, strategy=b.strategy, ef=b.params.get("ef"),
                         overfetch=b.params.get("overfetch", 8))

    def random(self, b: BranchPlan, ids):
        c = b.clause
        if c.kind == "keyword":
            return self.coll.text[c.field].score_docs(b.terms, ids)
        if c.kind == "vector":
            return self.coll.vectors[c.field].score(b.query, ids)
        return self.coll.multivectors[c.field].score(b.query, ids)

    def score_all(self, b: BranchPlan, n: int):
        """Exhaustive scoring of every allowed doc (the naive reference plan)."""
        c = b.clause
        if c.kind == "keyword":
            ids, scores, _ = self.coll.text[c.field].score_all(b.terms, self.allow)
            return ids, scores
        if c.kind == "vector":
            fidx = self.coll.vectors[c.field]
            ids = fidx.pos2doc.data
            if self.allow is not None:
                ids = ids[fidx._allow_pos(self.allow)]
            return ids, fidx.score(b.query, ids)
        ids, scores, _ = self.coll.multivectors[c.field].search(b.query, n, allow=self.allow)
        return ids, scores


def execute(coll, logical: LogicalPlan) -> QueryResult:
    t_start = time.perf_counter()
    timings = {}
    t0 = time.perf_counter()
    bound = bind(coll, logical, encode=not logical.explain or logical.analyze)
    timings["bind_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    plan = optimize(logical, coll, bound)
    for b in plan.branches:
        b.query = bound[b.clause.name].get("query")
    timings["plan_ms"] = (time.perf_counter() - t0) * 1e3
    if logical.explain and not logical.analyze:
        timings["total_ms"] = (time.perf_counter() - t_start) * 1e3
        return QueryResult([], plan, timings)

    n = coll.capacity
    t0 = time.perf_counter()
    allow = coll.alive_mask() if coll.has_deletions else None
    if plan.filter is not None:
        ev = FilterEvaluator(coll, n, plan.filter.access)
        mask = ev(plan.filter.expr)
        allow = mask if allow is None else (allow & mask)
        plan.filter.actual = {"rows": int(mask.sum()), "sel": float(mask.sum() / max(n, 1)),
                              "decisions": ev.decisions}
    timings["filter_ms"] = (time.perf_counter() - t0) * 1e3
    if plan.filter is not None:
        plan.filter.actual["ms"] = timings["filter_ms"]

    runner = _Runner(coll, allow)
    t0 = time.perf_counter()
    algo = plan.fuse.algorithm
    weights = plan.fuse.weights
    lists = {}
    ranks: dict[int, dict] = {}
    stats = {}
    limit = plan.limit

    if algo == "none":
        ids = np.flatnonzero(allow) if allow is not None else np.arange(n)
        fused_ids = ids[:limit].tolist()
        fused_scores = [0.0] * len(fused_ids)
        stats["matched"] = int(len(ids))
    elif algo in ("single", "full"):
        k = limit if algo == "single" else plan.fuse.depth
        for b in plan.branches:
            tb = time.perf_counter()
            ids, scores, st = runner.run(b, k)
            b.actual = {"ms": (time.perf_counter() - tb) * 1e3, "rows": int(len(ids)), **st}
            lists[b.clause.name] = (ids, scores)
        if algo == "single":
            name = plan.branches[0].clause.name
            ids, scores = lists[name]
            fused_ids, fused_scores = ids.tolist(), scores.tolist()
            ranks = {d: {name: r} for r, d in enumerate(fused_ids, 1)}
        elif plan.fuse.method == "weighted":
            union = sorted({d for ids, _ in lists.values() for d in ids.tolist()})
            completed = {}
            for b in plan.branches:
                raw = runner.random(b, union)
                completed[b.clause.name] = dict(zip(union, raw.tolist()))
            stats["random_accesses"] = len(union) * len(plan.branches)
            fused, ranks = fusion.weighted(lists, completed, weights, plan.fuse.params.get("norm", "max"))
            fused_ids, fused_scores = _top(fused, limit)
        else:
            fused, ranks = fusion.rrf(lists, weights, int(plan.fuse.params.get("k", 60)))
            fused_ids, fused_scores = _top(fused, limit)
        stats["candidates"] = len(ranks)
    elif algo in ("ta", "nra"):
        norm = plan.fuse.params.get("norm", "max")
        sources = [
            Source(b.clause.name, weights.get(b.clause.name, 1.0),
                   (lambda kk, b=b: runner.run(b, kk)[:2]), (lambda ids, b=b: runner.random(b, ids)), norm=norm,
                   first_batch=max(32, 2 * limit))
            for b in plan.branches
        ]
        fn = threshold_algorithm if algo == "ta" else nra
        fused_ids, fused_scores, stats = fn(sources, limit)
        if algo == "nra" and fused_ids:
            # NRA proves top-k membership from score bounds; the final order needs the
            # exact scores of those k docs (k random accesses per source, reported separately)
            exact = np.zeros(len(fused_ids))
            for src, b in zip(sources, plan.branches):
                exact += [src.contrib(x) for x in runner.random(b, fused_ids).tolist()]
            order = np.lexsort((np.arange(len(fused_ids)), -exact))
            fused_ids = [fused_ids[i] for i in order]
            fused_scores = [float(exact[i]) for i in order]
            stats["final_random_accesses"] = len(fused_ids) * len(sources)
        for b, s in zip(plan.branches, sources):
            b.actual = {"sorted": s.sorted_accesses, "random": s.random_accesses, "fetches": s.fetches}
    elif algo == "exhaustive":
        norm = plan.fuse.params.get("norm", "max")
        total: dict[int, float] = {}
        for b in plan.branches:
            ids, scores = runner.score_all(b, n)
            b.actual = {"rows": int(len(ids))}
            if not len(ids):
                continue
            top = float(scores.max())
            w = weights.get(b.clause.name, 1.0)
            if norm == "max":
                contrib = w * np.maximum(scores / top, 0.0) if top > 0 else np.zeros(len(scores))
            else:
                contrib = w * np.maximum(scores, 0.0)
            for d, c in zip(ids.tolist(), contrib.tolist()):
                total[d] = total.get(d, 0.0) + c
        fused_ids, fused_scores = _top(total, limit)
        stats["scored"] = len(total)
    else:
        raise ValueError(algo)
    timings["search_ms"] = (time.perf_counter() - t0) * 1e3
    plan.fuse.actual = {"ms": timings["search_ms"], **stats}

    t0 = time.perf_counter()
    hits = _materialize(coll, plan, runner, bound, fused_ids, fused_scores, ranks)
    timings["fetch_ms"] = (time.perf_counter() - t0) * 1e3
    timings["total_ms"] = (time.perf_counter() - t_start) * 1e3
    return QueryResult(hits, plan, timings, stats)


def _top(fused: dict, k: int):
    if not fused:
        return [], []
    docs = np.fromiter(fused.keys(), np.int64, len(fused))
    scores = np.fromiter(fused.values(), np.float64, len(fused))
    o = np.lexsort((docs, -scores))[:k]  # ties broken by doc id, as in TA/NRA
    return docs[o].tolist(), scores[o].tolist()


def _materialize(coll, plan, runner, bound, ids, scores, ranks) -> list[Hit]:
    if not ids:
        return []
    records = coll.docstore.get_many(ids)
    per_branch = {b.clause.name: runner.random(b, ids) for b in plan.branches}
    hits = []
    for i, (d, s) in enumerate(zip(ids, scores)):
        rec = records.get(d, {"_id": str(d)})
        ext = rec.pop("_id")
        sources = {}
        matched, moment = [], None
        for b in plan.branches:
            name = b.clause.name
            raw = float(per_branch[name][i])
            sources[name] = {"score": raw if np.isfinite(raw) else None,
                             "rank": ranks.get(d, {}).get(name),
                             "weight": plan.fuse.weights.get(name, 1.0)}
            if b.clause.kind == "keyword":
                inv = coll.text[b.clause.field]
                matched += [t for t, _ in b.terms if inv.score_docs([(t, 1)], [d])[0] > 0]
        # a transcript/OCR segment containing a matched keyword is the most precise moment;
        # otherwise fall back to the best-matching keyframe
        if matched:
            moment = _segment_moment(rec, matched, coll)
        for b in plan.branches:
            if moment is None and b.clause.kind == "multivector":
                moment = coll.multivectors[b.clause.field].best_moment(b.query, d)
        hits.append(Hit(ext, int(d), float(s), sources, rec, moment, matched))
    return hits


def _segment_moment(rec, terms, coll):
    """First transcript/OCR segment that mentions a matched term."""
    analyzer = next(iter(coll.text.values())).analyzer if coll.text else None
    for key, segs in rec.items():
        if not key.endswith(".segments") or not isinstance(segs, list) or analyzer is None:
            continue
        for seg in segs:
            if set(analyzer(seg.get("text", ""))) & set(terms):
                return float(seg.get("start", 0.0))
    return None
