"""Load the synthetic benchmark corpus into PostgreSQL + pgvector and time the
same hybrid queries OmniSearch runs (bench_scale-style data).

    cd baselines/pgvector && docker compose up -d
    pip install "psycopg[binary]" pgvector
    python -m baselines.pgvector.run_pgvector --n 20000

Reports load/index build time and p50/p95 latency for keyword, vector and
hybrid (RRF) queries, with and without a filter, plus how many of the vector
search's top-10 match the exact answer (pgvector's HNSW with a WHERE clause
post-filters, so recall drops for selective filters - compare with
benchmarks/bench_optimizer.py).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from benchmarks.common import exact_topk, recall, save, summary
from omnisearch.datasets.synthetic import attributes, clustered_vectors, queries_like, topic_corpus, topic_queries

HYBRID = (Path(__file__).parent / "hybrid_query.sql").read_text(encoding="utf-8").replace("$1", "%(tsq)s").replace(
    "$2", "%(vec)s").replace("$3", "%(cut)s").replace("$4", "%(k)s")


def main():
    import psycopg
    from pgvector.psycopg import register_vector

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--dsn", default="postgresql://postgres:omnisearch@localhost:5433/omnisearch")
    args = ap.parse_args()
    x, labels, centers = clustered_vectors(args.n, 128, seed=0)
    texts, topics = topic_corpus(labels)
    attrs = attributes(labels)
    qv, ql = queries_like(centers, args.queries, seed=3)
    qt = topic_queries(ql, topics)

    with psycopg.connect(args.dsn, autocommit=True) as con:
        register_vector(con)
        con.execute("TRUNCATE docs")
        t0 = time.perf_counter()
        with con.cursor().copy("COPY docs (id, body, embedding, price, bucket) FROM STDIN") as cp:
            for i in range(args.n):
                cp.write_row((i, texts[i], x[i], float(attrs["price"][i]), int(attrs["bucket"][i])))
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        con.execute("CREATE INDEX IF NOT EXISTS docs_tsv ON docs USING gin (tsv)")
        con.execute("CREATE INDEX IF NOT EXISTS docs_hnsw ON docs USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m = 16, ef_construction = 100)")
        con.execute("ANALYZE docs")
        index_s = time.perf_counter() - t0

        lat = {"keyword": [], "vector": [], "hybrid": [], "hybrid+filter(1%)": []}
        rec = {"unfiltered": [], "filter 1%": []}
        for i in range(args.queries):
            tsq = " | ".join(qt[i].split())
            t = time.perf_counter()
            con.execute("SELECT id FROM docs, to_tsquery('simple', %s) q WHERE tsv @@ q "
                        "ORDER BY ts_rank_cd(tsv, q) DESC LIMIT 10", (tsq,)).fetchall()
            lat["keyword"].append((time.perf_counter() - t) * 1e3)
            for cut, key in [(1000, "unfiltered"), (10, "filter 1%")]:
                t = time.perf_counter()
                rows = con.execute("SELECT id FROM docs WHERE bucket < %s ORDER BY embedding <=> %s LIMIT 10",
                                   (cut, qv[i])).fetchall()
                if cut == 1000:
                    lat["vector"].append((time.perf_counter() - t) * 1e3)
                allow = attrs["bucket"] < cut
                rec[key].append(recall([r[0] for r in rows], exact_topk(x, qv[i][None], 10, allow)[0]))
            for cut, key in [(1000, "hybrid"), (10, "hybrid+filter(1%)")]:
                t = time.perf_counter()
                con.execute(HYBRID, {"tsq": tsq, "vec": qv[i], "cut": cut, "k": 10}).fetchall()
                lat[key].append((time.perf_counter() - t) * 1e3)
    out = {"n": args.n, "load_s": load_s, "index_s": index_s,
           "latency": {k: summary(v) for k, v in lat.items()},
           "vector_recall@10": {k: float(np.mean(v)) for k, v in rec.items()}}
    print(out)
    save("pgvector", out)


if __name__ == "__main__":
    main()
