"""Vector compression: memory vs recall vs latency for float32, int8 (sq8) and
product quantization (pq8/pq16/pq32), with and without exact re-ranking.

    python -m benchmarks.bench_quantization [--n 50000] [--dim 128]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from benchmarks.common import exact_topk, recall, save
from omnisearch.datasets.synthetic import clustered_vectors, queries_like
from omnisearch.index.vector import VectorFieldIndex
from omnisearch.schema import VectorField


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=100)
    args = ap.parse_args(argv)
    x, _, centers = clustered_vectors(args.n, args.dim, seed=0)
    qs, _ = queries_like(centers, args.queries)
    truth = exact_topk(x, qs, 10)
    rows = []
    for quant in [None, "sq8", "pq32", "pq16", "pq8"]:
        for refine in ([0] if quant is None else [0, 4]):
            f = VectorFieldIndex(VectorField("v", args.dim, "precomputed", "v", index="flat", quantization=quant,
                                             params={"refine": refine}))
            t0 = time.perf_counter()
            f.add(np.arange(args.n), x)
            f.build()
            train_s = time.perf_counter() - t0
            lat, rec = [], []
            for q, gt in zip(qs, truth):
                t = time.perf_counter()
                ids, _, _ = f.search(q, 10, strategy="exact")
                lat.append((time.perf_counter() - t) * 1e3)
                rec.append(recall(ids, gt))
            mem = f.memory()
            code_bytes = mem.get("code_bytes", mem["float32_bytes"])
            rows.append({"quantization": quant or "float32", "refine": refine, "recall": float(np.mean(rec)),
                         "ms": float(np.mean(lat)), "bytes_per_vector": code_bytes / args.n, "build_s": train_s})
            r = rows[-1]
            print(f"{r['quantization']:<8} refine={refine} recall={r['recall']:.3f} {r['ms']:.2f}ms "
                  f"{r['bytes_per_vector']:.0f} B/vec")
    save("quantization", {"n": args.n, "dim": args.dim, "rows": rows, "data": "synthetic clustered vectors"})


if __name__ == "__main__":
    main()
