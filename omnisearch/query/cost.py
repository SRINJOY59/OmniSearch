"""Cost model for physical operators.

Like Postgres' ``seq_page_cost`` / ``cpu_tuple_cost``, every operator cost is
a sum of unit costs times estimated work. Unit costs (seconds) default to
values measured on a 4-core cloud VM; ``calibrate()`` re-measures them on the
current machine and collection, which matters because pure-Python graph
traversal and vectorised numpy scans differ by ~1000x per element.

HNSW work is modelled as node expansions: an unfiltered search with beam
width ef expands about ``a*ef + b*log2(N)`` nodes (a, b are fitted). With a
filter of selectivity s the beam must expand ~1/s times more nodes to collect
ef allowed ones.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CostModel:
    scan_per_dim: float = 0.12e-9      # contiguous matrix-vector product, per element
    gather_per_dim: float = 0.35e-9    # same over a gathered subset of rows
    hnsw_expansion: float = 6.0e-6     # one node expansion in the beam search
    hnsw_a: float = 1.6                # expansions ~ a * ef + b * log2(N)
    hnsw_b: float = 4.0
    posting: float = 8.0e-9            # decode + score one posting (vectorised)
    term_overhead: float = 20e-6       # fixed cost per query term (lookup, numpy calls)
    wand_step: float = 4.0e-6          # one WAND pivot step (interpreted)
    btree_row: float = 0.35e-6         # one row out of a B+-tree range scan
    col_row: float = 1.5e-9            # one row of a vectorised column scan
    random_access: float = 1.5e-6      # score one (doc, source) pair by random access
    sorted_access: float = 0.8e-6      # consume one item from a sorted stream
    op_overhead: float = 25e-6         # fixed cost per operator invocation
    calibrated: bool = False
    notes: dict = field(default_factory=dict)

    def hnsw_expansions(self, ef: int, n: int) -> float:
        return min(float(n), self.hnsw_a * ef + self.hnsw_b * math.log2(n + 1))

    def save(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CostModel":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls(**json.loads(p.read_text(encoding="utf-8")))


@dataclass
class Alternative:
    strategy: str
    cost: float  # seconds
    params: dict = field(default_factory=dict)
    note: str = ""


def vector_alternatives(cm: CostModel, fidx, k: int, sel: float, filtered: bool, opts: dict) -> list[Alternative]:
    """Candidate access paths for SIMILAR on a single-vector field."""
    from omnisearch.index.vector.hnsw import HNSWIndex
    from omnisearch.index.vector.ivf import IVFIndex

    n = max(len(fidx), 1)
    d = fidx.field.dim
    sel = min(max(sel, 1.0 / n), 1.0)
    allowed = sel * n if filtered else n
    per_dim = cm.gather_per_dim if filtered else cm.scan_per_dim
    alts = [Alternative("exact", cm.op_overhead + allowed * d * per_dim + (n * cm.col_row if filtered else 0),
                        note=f"scan {int(allowed)} vectors")]
    ann = fidx.ann
    if isinstance(ann, HNSWIndex):
        ef = max(opts.get("ef") or ann.ef_search, k)
        base = cm.hnsw_expansions(ef, n)
        if filtered:
            e = min(n, base / sel)
            alts.append(Alternative("index", cm.op_overhead + e * cm.hnsw_expansion, {"ef": ef},
                                    note=f"~{int(e)} node expansions"))
            over = int(min(n, math.ceil(1.5 * k / sel)))
            if opts.get("overfetch"):
                over = opts["overfetch"]
            e2 = cm.hnsw_expansions(max(ef, over), n)
            alts.append(Alternative("postfilter", cm.op_overhead + e2 * cm.hnsw_expansion,
                                    {"ef": max(ef, over), "overfetch": over}, note=f"fetch {over}, keep ~{int(over * sel)}"))
        else:
            alts.append(Alternative("index", cm.op_overhead + base * cm.hnsw_expansion, {"ef": ef},
                                    note=f"~{int(base)} node expansions"))
    elif isinstance(ann, IVFIndex):
        nlist = len(ann.lists) or 1
        nprobe = opts.get("nprobe") or ann.nprobe
        if filtered and ann.trained:  # widen the probe so ~4k allowed candidates are expected
            nprobe = min(nlist, max(nprobe, math.ceil(4 * k * nlist / max(sel * n, 1))))
        frac = min(1.0, nprobe / nlist)
        cand = n * frac * (sel if filtered else 1.0)
        cost = cm.op_overhead + nlist * d * cm.scan_per_dim + n * frac * cm.col_row + cand * d * cm.gather_per_dim
        alts.append(Alternative("index", cost, {"nprobe": nprobe}, note=f"probe {nprobe}/{nlist} lists, ~{int(cand)} candidates"))
    return alts


def multivector_alternatives(cm: CostModel, mv, k: int, sel: float, filtered: bool, opts: dict) -> list[Alternative]:
    n_vec = max(len(mv), 1)
    d = mv.store.dim
    alts = [Alternative("exact", cm.op_overhead + n_vec * d * cm.scan_per_dim, note=f"MaxSim over {n_vec} vectors")]
    if mv.hnsw is not None:
        over = opts.get("overfetch") or 8
        ef = max(opts.get("ef") or 64, k * over)
        e = cm.hnsw_expansions(ef, n_vec) / (sel if filtered else 1.0)
        alts.append(Alternative("index", cm.op_overhead + min(e, n_vec) * cm.hnsw_expansion,
                                {"ef": ef, "overfetch": over}, note=f"~{int(min(e, n_vec))} node expansions"))
    return alts


def keyword_alternatives(cm: CostModel, inv, terms, k: int, opts: dict) -> list[Alternative]:
    postings = sum(inv.df(t) for t, _ in terms)
    alts = [Alternative("exhaustive", cm.op_overhead + len(terms) * cm.term_overhead + postings * cm.posting,
                        note=f"{postings} postings")]
    if len(terms) > 1:
        steps = min(postings, 30 * k * len(terms))
        alts.append(Alternative("wand", cm.op_overhead + steps * cm.wand_step, note=f"~{steps} pivot steps (heuristic)"))
    return alts


# ------------------------------------------------------------------ calibrate
def _timeit(fn, repeat=5) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat


def calibrate(collection, seed: int = 0) -> CostModel:
    """Measure unit costs on this machine using the collection's own indexes."""
    from omnisearch.index.vector.hnsw import HNSWIndex

    rng = np.random.default_rng(seed)
    cm = CostModel()
    notes = {}
    fields = sorted(collection.vectors.values(), key=len, reverse=True)
    if fields and len(fields[0]) >= 1000:
        X = fields[0].store.vectors
    else:
        X = rng.normal(size=(20000, 128)).astype(np.float32)
    n, d = X.shape
    q = X[rng.integers(n)].copy()
    cm.scan_per_dim = _timeit(lambda: X @ q) / (n * d)
    pos = np.sort(rng.choice(n, n // 5, replace=False))
    cm.gather_per_dim = _timeit(lambda: X[pos] @ q) / (len(pos) * d)
    mask = rng.random(n) < 0.5
    cm.col_row = _timeit(lambda: np.flatnonzero(mask & mask)) / n
    notes["vectors"] = f"{n}x{d}"

    hnsw_fields = [f for f in fields if isinstance(f.ann, HNSWIndex) and len(f.ann) >= 500]
    if hnsw_fields:
        f = hnsw_fields[0]
        f.build()
        nn = len(f.ann)
        rows, ys, total_t, total_e = [], [], 0.0, 0
        for ef in (16, 48, 128):
            for _ in range(8):
                qq = f.store.vectors[rng.integers(nn)] + 0.05 * rng.normal(size=f.field.dim).astype(np.float32)
                qq = f.prepare_query(qq)
                t0 = time.perf_counter()
                _, _, st = f.ann.search(qq, 10, ef=ef)
                total_t += time.perf_counter() - t0
                total_e += st["expansions"]
                rows.append([ef, math.log2(nn + 1)])
                ys.append(st["expansions"])
        cm.hnsw_expansion = total_t / max(total_e, 1)
        coef, *_ = np.linalg.lstsq(np.array(rows, dtype=float), np.array(ys, dtype=float), rcond=None)
        cm.hnsw_a, cm.hnsw_b = float(max(coef[0], 0.1)), float(max(coef[1], 0.0))
        notes["hnsw"] = f"fitted on {f.field.name} ({nn} nodes)"

    texts = [t for t in collection.text.values() if t.n_docs >= 100]
    if texts:
        inv = texts[0]
        vocab = sorted(inv.postings, key=inv.df, reverse=True)[:200]
        queries = [[(t, 1) for t in rng.choice(vocab, int(rng.integers(1, 5)), replace=False)] for _ in range(16)]
        rows = [[len(qq), sum(inv.df(t) for t, _ in qq)] for qq in queries]
        times = [_timeit(lambda qq=qq: inv.score_all(qq), 3) for qq in queries]
        coef, *_ = np.linalg.lstsq(np.array(rows, dtype=float), np.array(times), rcond=None)
        cm.term_overhead, cm.posting = float(max(coef[0], 1e-7)), float(max(coef[1], 1e-10))
        steps, tw = 0, 0.0
        for qq in queries:
            t0 = time.perf_counter()
            _, _, st = inv.search(qq, 10, algo="wand")
            tw += time.perf_counter() - t0
            steps += st["postings"] + st["evaluated"]
        cm.wand_step = tw / max(steps, 1)
        notes["text"] = f"{inv.n_docs} docs"

    nums = [a for a in collection.attrs.values() if getattr(a, "kind", "") == "numeric" and len(a.values) >= 100]
    if nums:
        a = nums[0]
        vals = a.values.data[~np.isnan(a.values.data)]
        lo, hi = np.quantile(vals, [0.45, 0.55])
        m = len(a.values)
        rows = int(((vals >= lo) & (vals <= hi)).sum())
        cm.btree_row = _timeit(lambda: a.tree_range(lo, hi, True, True, m), 3) / max(rows, 1)

    if fields and len(fields[0]) >= 100:
        f = fields[0]
        ids = rng.choice(f.pos2doc.data, min(200, len(f)), replace=False)
        qq = f.prepare_query(f.store.vectors[0])
        cm.random_access = _timeit(lambda: f.score(qq, ids)) / len(ids)

    cm.calibrated = True
    cm.notes = notes
    return cm
