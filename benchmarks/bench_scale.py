"""Scaling: build time, index size and hybrid query latency as the collection grows.

    python -m benchmarks.bench_scale [--sizes 10000 25000 50000]
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

from benchmarks.common import save, summary, vec_literal
from omnisearch.datasets.synthetic import attributes, clustered_vectors, queries_like, topic_corpus, topic_queries
from omnisearch.engine import Collection
from omnisearch.schema import AttributeField, Schema, TextField, VectorField


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[10000, 25000, 50000])
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--queries", type=int, default=50)
    args = ap.parse_args(argv)
    rows = []
    for n in args.sizes:
        x, labels, centers = clustered_vectors(n, args.dim, seed=0)
        texts, topics = topic_corpus(labels)
        attrs = attributes(labels)
        tmp = Path(tempfile.mkdtemp())
        schema = Schema("scale", [TextField("text", stem=False),
                                  VectorField("vec", args.dim, "precomputed", "vec", index="hnsw"),
                                  AttributeField("price", "float")])
        coll = Collection.create(tmp / "scale", schema)
        t0 = time.perf_counter()
        coll.add([{"id": str(i), "text": texts[i], "vec": x[i], "price": float(attrs["price"][i])} for i in range(n)],
                 batch_size=5000)
        ingest_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        coll.build()
        build_s = time.perf_counter() - t0
        coll.save()
        coll.calibrate()
        size = dir_bytes(tmp / "scale")
        try:
            import faiss

            faiss.omp_set_num_threads(1)
            h = faiss.IndexHNSWFlat(args.dim, 16, faiss.METRIC_INNER_PRODUCT)
            h.hnsw.efConstruction = 100
            t0 = time.perf_counter()
            h.add(x)
            faiss_build = time.perf_counter() - t0
        except ImportError:
            faiss_build = None
        qv, ql = queries_like(centers, args.queries, seed=3)
        qt = topic_queries(ql, topics)
        lat = {"keyword": [], "vector": [], "hybrid": [], "hybrid+filter": []}
        for i in range(args.queries):
            v = vec_literal(qv[i])
            qs = {
                "keyword": f"SEARCH scale WHERE MATCH(text, '{qt[i]}') LIMIT 10",
                "vector": f"SEARCH scale WHERE SIMILAR(vec, {v}) LIMIT 10",
                "hybrid": f"SEARCH scale WHERE MATCH(text, '{qt[i]}') AND SIMILAR(vec, {v}) LIMIT 10",
                "hybrid+filter": f"SEARCH scale WHERE MATCH(text, '{qt[i]}') AND SIMILAR(vec, {v}) AND price < 30 LIMIT 10",
            }
            for k, q in qs.items():
                res = coll.query(q)
                lat[k].append(res.timings["total_ms"] - res.timings["bind_ms"])
        row = {"n": n, "ingest_s": ingest_s, "hnsw_build_s": build_s, "faiss_hnsw_build_s": faiss_build,
               "disk_bytes": size, "latency": {k: summary(v) for k, v in lat.items()}}
        rows.append(row)
        print(f"n={n:<7} ingest={ingest_s:.1f}s hnsw={build_s:.1f}s faiss={faiss_build or 0:.2f}s "
              f"disk={size/1e6:.1f}MB " + " ".join(f"{k}={v['p50']:.2f}ms" for k, v in row["latency"].items()))
        coll.close()
        shutil.rmtree(tmp)
    save("scale", {"dim": args.dim, "rows": rows,
                   "note": "latency excludes query parsing of the 128-float VECTOR literal (bind)"})


if __name__ == "__main__":
    main()
