"""Keyword index: posting-list compression, exhaustive BM25 vs WAND, and a
comparison with SQLite FTS5 (a production inverted index with BM25).

    python -m benchmarks.bench_text [--docs 50000] [--queries 200]
"""
from __future__ import annotations

import argparse
import sqlite3
import time

import numpy as np

from benchmarks.common import save, summary
from omnisearch.datasets.synthetic import clustered_vectors, topic_corpus, topic_queries
from omnisearch.index.text import Analyzer, InvertedIndex, vbyte_decode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=50000)
    ap.add_argument("--queries", type=int, default=200)
    args = ap.parse_args(argv)
    _, labels, _ = clustered_vectors(args.docs, 8, n_clusters=200, seed=0)
    texts, topics = topic_corpus(labels, vocab_size=30000)
    rng = np.random.default_rng(9)
    qlab = rng.integers(0, len(topics), args.queries)
    queries = topic_queries(qlab, topics, n_words=(2, 5))
    # add some queries with a very common word, where WAND can skip the most
    queries += [q + " w1 w2" for q in queries[: args.queries // 4]]

    inv = InvertedIndex(Analyzer(stem=False, stopwords=False))
    t0 = time.perf_counter()
    for i, t in enumerate(texts):
        inv.add(i, t)
    inv.flush()
    build_s = time.perf_counter() - t0
    st = inv.stats()

    # decode throughput over the 200 longest lists
    longest = sorted(inv.postings.values(), key=len, reverse=True)[:200]
    t0 = time.perf_counter()
    decoded = sum(len(vbyte_decode(p.doc_bytes)) + len(vbyte_decode(p.tf_bytes)) for p in longest)
    decode_s = time.perf_counter() - t0

    algos = {}
    for algo in ("exhaustive", "wand"):
        lat, evaluated, postings = [], [], []
        for q in queries:
            terms = inv.query_terms(q)
            t = time.perf_counter()
            _, _, s = inv.search(terms, 10, algo=algo)
            lat.append((time.perf_counter() - t) * 1e3)
            evaluated.append(s["evaluated"])
            postings.append(s["postings"])
        algos[algo] = {"latency": summary(lat), "docs_scored": float(np.mean(evaluated)),
                       "postings_touched": float(np.mean(postings))}

    # SQLite FTS5 baseline (same BM25 parameters k1=1.2, b=0.75)
    con = sqlite3.connect(":memory:")
    con.execute("CREATE VIRTUAL TABLE docs USING fts5(body, tokenize='unicode61')")
    t0 = time.perf_counter()
    con.executemany("INSERT INTO docs(rowid, body) VALUES (?, ?)", enumerate(texts))
    con.commit()
    fts_build = time.perf_counter() - t0
    lat, overlap = [], []
    for q in queries:
        expr = " OR ".join(dict.fromkeys(q.split()))
        t = time.perf_counter()
        rows = con.execute("SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT 10", (expr,)).fetchall()
        lat.append((time.perf_counter() - t) * 1e3)
        ours, scores, _ = inv.search(inv.query_terms(q), 10)
        overlap.append(len({r[0] for r in rows} & set(ours.tolist())) / 10)
    fts = {"build_s": fts_build, "latency": summary(lat), "top10_overlap_with_ours": float(np.mean(overlap))}

    print(f"docs={args.docs} postings={st['postings']} compressed={st['compressed_bytes']/1e6:.2f}MB "
          f"raw={st['uncompressed_bytes']/1e6:.2f}MB ({8 * st['compressed_bytes'] / st['postings']:.1f} bits/posting)")
    for a, r in algos.items():
        print(f"{a:<10} p50={r['latency']['p50']:.2f}ms docs_scored={r['docs_scored']:.0f} postings={r['postings_touched']:.0f}")
    print(f"fts5       p50={fts['latency']['p50']:.2f}ms overlap={fts['top10_overlap_with_ours']:.2f}")
    save("text", {
        "docs": args.docs, "queries": len(queries), "build_s": build_s, "index": st,
        "bits_per_posting": 8 * st["compressed_bytes"] / st["postings"],
        "decode_million_ints_per_s": decoded / decode_s / 1e6,
        "algos": algos, "fts5": fts,
        "data": "synthetic Zipfian topic corpus, 30k-word vocabulary",
    })


if __name__ == "__main__":
    main()
