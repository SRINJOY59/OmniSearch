"""Top-k fusion algorithms: Threshold Algorithm and NRA vs materialise-and-merge
vs scoring every document.

    python -m benchmarks.bench_topk [--n 20000] [--queries 40]

Queries combine MATCH(text) with one or two SIMILAR clauses under WEIGHTED
fusion. In the "correlated" setting the text and the vectors of a query come
from the same topic, so good documents score high in every list and TA can
stop early; in the "independent" setting they come from different topics.
Correctness is the fraction of returned docs whose fused score reaches the
exact k-th best score (from the exhaustive plan), which is robust to ties.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.common import save, vec_literal
from omnisearch.datasets.synthetic import clustered_vectors, queries_like, topic_corpus, topic_queries
from omnisearch.engine import Collection
from omnisearch.schema import Schema, TextField, VectorField

ALGOS = ["exhaustive", "full", "ta", "nra"]


def build(n, dim, path):
    x, labels, centers = clustered_vectors(n, dim, seed=0)
    x2, _, _ = clustered_vectors(n, dim, seed=5)  # a second, independent embedding
    texts, topics = topic_corpus(labels)
    schema = Schema("topk", [TextField("text", stem=False),
                             VectorField("v1", dim, "precomputed", "v1", index="flat"),
                             VectorField("v2", dim, "precomputed", "v2", index="flat")])
    coll = Collection.create(path, schema, overwrite=True)
    coll.add([{"id": str(i), "text": texts[i], "v1": x[i], "v2": x2[i]} for i in range(n)], batch_size=5000)
    coll.build()
    return coll, centers, topics


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--queries", type=int, default=40)
    args = ap.parse_args(argv)
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        coll, centers, topics = build(args.n, args.dim, Path(tmp) / "topk")
        qv, qlab = queries_like(centers, args.queries, seed=21)
        settings = {
            "correlated": topic_queries(qlab, topics),
            "independent": topic_queries((qlab + 17) % len(topics), topics),
        }
        for setting, texts in settings.items():
            for m in (2, 3):
                for qi in range(args.queries):  # warm-up: decode posting lists, fill caches
                    sims = f"SIMILAR(v1, {vec_literal(qv[qi])})" + (f" AND SIMILAR(v2, {vec_literal(qv[qi])})" if m == 3 else "")
                    coll.query(f"SEARCH topk WHERE MATCH(text, '{texts[qi]}') AND {sims} FUSE WEIGHTED LIMIT 10")
                for k in (1, 10, 50):
                    row = {"setting": setting, "sources": m, "k": k}
                    ref_ids = None
                    for algo in ALGOS:
                        lat, sorted_acc, random_acc, exact = [], [], [], []
                        for qi in range(args.queries):
                            sims = f"SIMILAR(v1, {vec_literal(qv[qi])})"
                            if m == 3:
                                sims += f" AND SIMILAR(v2, {vec_literal(qv[qi])})"
                            q = (f"SEARCH topk WHERE MATCH(text, '{texts[qi]}') AND {sims} FUSE WEIGHTED LIMIT {k} "
                                 f"WITH (topk='{algo}', strategy='exact')")
                            res = coll.query(q)
                            lat.append(res.timings["search_ms"])
                            st = res.plan.fuse.actual
                            sorted_acc.append(st.get("sorted_accesses", 0))
                            random_acc.append(st.get("random_accesses", 0))
                            scores = [h.score for h in res.hits]
                            if algo == "exhaustive":
                                ref_ids = ref_ids or {}
                                ref_ids[qi] = scores[-1] if scores else 0.0  # exact k-th best score
                            # correct = scores at least the true k-th score (robust to ties)
                            exact.append(np.mean([s >= ref_ids[qi] - 1e-6 for s in scores]) if scores else 1.0)
                        row[algo] = {"ms": float(np.median(lat)), "mean_ms": float(np.mean(lat)),
                                     "sorted": float(np.mean(sorted_acc)),
                                     "random": float(np.mean(random_acc)), "overlap": float(np.mean(exact))}
                    print(f"{setting:<12} m={m} k={k:<3} " + "  ".join(
                        f"{a}={row[a]['ms']:.2f}ms(ok {row[a]['overlap']:.2f})" for a in ALGOS))
                    out.append(row)
        coll.close()
    save("topk", {"n": args.n, "dim": args.dim, "queries": args.queries, "rows": out,
                  "data": "synthetic: topic corpus + clustered vectors; exact sorted access (strategy=exact)"})


if __name__ == "__main__":
    main()
