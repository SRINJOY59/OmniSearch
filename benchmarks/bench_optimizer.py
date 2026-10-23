"""Filtered vector search: fixed access paths vs the cost-based optimizer.

    python -m benchmarks.bench_optimizer [--n 20000] [--dim 128] [--queries 40]

For filter selectivities from 0.1% to 100% every query runs through the real
engine with each access path forced (exact pre-filter scan, HNSW with
in-graph filtering, HNSW + post-filter) and with the optimizer choosing
(auto). Reports latency and recall@10 against the exact filtered answer.

A second experiment uses a filter correlated with the query (same cluster)
and an anti-correlated one, where post-filtering and in-graph filtering
behave very differently even at equal selectivity.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.common import exact_topk, recall, save, vec_literal
from omnisearch.datasets.synthetic import attributes, clustered_vectors, queries_like
from omnisearch.engine import Collection
from omnisearch.schema import AttributeField, Schema, VectorField

STRATEGIES = ["exact", "index", "postfilter", "auto"]
SELECTIVITIES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def build(n, dim, path):
    x, labels, centers = clustered_vectors(n, dim, seed=0)
    attrs = attributes(labels)
    schema = Schema("bench", [VectorField("vec", dim, "precomputed", "vec", index="hnsw"),
                              AttributeField("bucket", "int"), AttributeField("group", "category")])
    coll = Collection.create(path, schema, overwrite=True)
    recs = [{"id": str(i), "vec": x[i], "bucket": int(attrs["bucket"][i]), "group": int(attrs["group"][i])}
            for i in range(n)]
    coll.add(recs, batch_size=5000)
    coll.calibrate()
    return coll, x, labels, centers, attrs


def run_point(coll, x, qs, where_fn, allow_fn, strategy):
    lat, rec, chosen = [], [], []
    for qi, q in enumerate(qs):
        allow = allow_fn(qi)
        truth = exact_topk(x, q[None], 10, allow)[0] if allow.any() else set()
        res = coll.query(f"SEARCH bench WHERE SIMILAR(vec, {vec_literal(q)}) AND {where_fn(qi)} LIMIT 10"
                         + ("" if strategy == "auto" else f" WITH (strategy='{strategy}')"))
        lat.append(res.timings["filter_ms"] + res.timings["search_ms"])
        rec.append(recall([int(h.id) for h in res.hits], truth) if truth else 1.0)
        chosen.append(res.plan.branches[0].strategy)
    return {"ms": float(np.mean(lat)), "p95_ms": float(np.percentile(lat, 95)), "recall": float(np.mean(rec)),
            "chosen": max(set(chosen), key=chosen.count)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=40)
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        coll, x, labels, centers, attrs = build(args.n, args.dim, Path(tmp) / "bench")
        qs, qlabels = queries_like(centers, args.queries, seed=11)
        bucket = attrs["bucket"]
        sweep = []
        for s in SELECTIVITIES:
            cut = int(round(s * 1000))
            allow = bucket < cut
            row = {"selectivity": s, "actual": float(allow.mean())}
            for strat in STRATEGIES:
                row[strat] = run_point(coll, x, qs, lambda qi: f"bucket < {cut}", lambda qi: allow, strat)
            print(f"sel={s:<6} " + "  ".join(f"{k}={row[k]['ms']:.2f}ms/r{row[k]['recall']:.2f}" for k in STRATEGIES)
                  + f"  auto->{row['auto']['chosen']}")
            sweep.append(row)

        groups = attrs["group"]
        corr = {}
        for name, pick in [("correlated", lambda qi: int(qlabels[qi])),
                           ("anti-correlated", lambda qi: int((qlabels[qi] + 32) % 64))]:
            corr[name] = {}
            for strat in STRATEGIES:
                corr[name][strat] = run_point(coll, x, qs, lambda qi: f"group = '{pick(qi)}'",
                                              lambda qi: groups == pick(qi), strat)
            print(name, {k: (round(v["ms"], 2), round(v["recall"], 3)) for k, v in corr[name].items()})
        cost = coll.cost.__dict__
        coll.close()
    save("optimizer", {"n": args.n, "dim": args.dim, "queries": args.queries, "sweep": sweep,
                       "correlation": corr, "cost_model": cost,
                       "data": "synthetic clustered vectors; bucket uniform 0..999; group = cluster for 50% of docs"})


if __name__ == "__main__":
    main()
