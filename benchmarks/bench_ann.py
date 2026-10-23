"""ANN indexes: recall@10 vs latency, build time and memory - ours vs FAISS.

    python -m benchmarks.bench_ann [--n 20000] [--dim 128] [--queries 200]

Both libraries run single-threaded so per-query latencies are comparable.
Ours is pure Python + numpy; FAISS is optimised C++ - the interesting result
is the shape of the recall/latency curves, and how far an interpreted
implementation sits from a production one.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from benchmarks.common import exact_topk, recall, save
from omnisearch.datasets.synthetic import clustered_vectors, queries_like
from omnisearch.index.vector import VectorFieldIndex
from omnisearch.schema import VectorField


def ours(x, qs, truth, spec, sweep_key, sweep):
    f = VectorFieldIndex(VectorField("v", x.shape[1], "precomputed", "v", **spec))
    t0 = time.perf_counter()
    f.add(np.arange(len(x)), x)
    f.build()
    build_s = time.perf_counter() - t0
    points = []
    for val in sweep:
        lat, rec = [], []
        for q, gt in zip(qs, truth):
            t = time.perf_counter()
            ids, _, _ = f.search(q, 10, strategy="index", **{sweep_key: val})
            lat.append((time.perf_counter() - t) * 1e3)
            rec.append(recall(ids, gt))
        points.append({sweep_key: val, "recall": float(np.mean(rec)), "ms": float(np.mean(lat))})
    mem = f.memory()
    return {"build_s": build_s, "points": points, "memory": mem}


def faiss_runs(x, qs, truth, nlist, efs, nprobes, pq_m):
    import faiss

    faiss.omp_set_num_threads(1)
    d = x.shape[1]
    out = {}

    def sweep(index, setter, values, key):
        pts = []
        for v in values:
            setter(v)
            lat, rec = [], []
            for q, gt in zip(qs, truth):
                t = time.perf_counter()
                _, ids = index.search(q[None], 10)
                lat.append((time.perf_counter() - t) * 1e3)
                rec.append(recall(ids[0], gt))
            pts.append({key: v, "recall": float(np.mean(rec)), "ms": float(np.mean(lat))})
        return pts

    h = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT)
    h.hnsw.efConstruction = 100
    t0 = time.perf_counter()
    h.add(x)
    out["faiss-hnsw"] = {"build_s": time.perf_counter() - t0,
                         "points": sweep(h, lambda v: setattr(h.hnsw, "efSearch", v), efs, "ef")}
    quant = faiss.IndexFlatIP(d)
    ivf = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    ivf.train(x)
    ivf.add(x)
    out["faiss-ivf"] = {"build_s": time.perf_counter() - t0,
                        "points": sweep(ivf, lambda v: setattr(ivf, "nprobe", v), nprobes, "nprobe")}
    quant2 = faiss.IndexFlatIP(d)
    ivfpq = faiss.IndexIVFPQ(quant2, d, nlist, pq_m, 8, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    ivfpq.train(x)
    refined = faiss.IndexRefineFlat(ivfpq)  # re-rank with exact vectors, like ours (refine=4)
    refined.k_factor = 4
    refined.add(x)
    out["faiss-ivfpq"] = {"build_s": time.perf_counter() - t0,
                          "points": sweep(refined, lambda v: setattr(ivfpq, "nprobe", v), nprobes, "nprobe")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=200)
    args = ap.parse_args(argv)
    x, _, centers = clustered_vectors(args.n, args.dim, seed=0)
    qs, _ = queries_like(centers, args.queries)
    truth = exact_topk(x, qs, 10)
    nlist = int(round(4 * math.sqrt(args.n)))
    efs, nprobes = [10, 20, 40, 80, 160], [1, 2, 4, 8, 16, 32]
    pq_m = 16
    runs = {
        "hnsw": ours(x, qs, truth, {"index": "hnsw", "params": {"M": 16, "ef_construction": 100}}, "ef", efs),
        "ivf": ours(x, qs, truth, {"index": "ivf", "params": {"nlist": nlist}}, "nprobe", nprobes),
        f"ivf-pq{pq_m}": ours(x, qs, truth, {"index": "ivf", "quantization": f"pq{pq_m}", "params": {"nlist": nlist}},
                              "nprobe", nprobes),
    }
    flat = ours(x, qs, truth, {"index": "flat"}, "ef", [None])
    runs["flat"] = flat
    try:
        runs.update(faiss_runs(x, qs, truth, nlist, efs, nprobes, pq_m))
    except ImportError:
        print("faiss not installed; skipping baseline")
    for name, r in runs.items():
        print(f"{name:<14} build {r['build_s']:7.2f}s   " +
              "  ".join(f"r={p['recall']:.3f}/{p['ms']:.3f}ms" for p in r["points"]))
    save("ann", {"n": args.n, "dim": args.dim, "queries": args.queries, "nlist": nlist, "runs": runs,
                 "data": "synthetic clustered vectors (64 clusters), cosine"})


if __name__ == "__main__":
    main()
