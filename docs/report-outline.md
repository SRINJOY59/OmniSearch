# Course report outline

A suggested structure for the Advanced Database Systems write-up, mapping each
section to the code and the experiment that supports it.

## 1. Introduction
- Problem: keyword search misses meaning; vector search misses exact terms;
  real collections mix text, images, audio and video plus structured attributes.
- Contribution: a hybrid engine whose *database* parts are the focus - index
  structures, a query language, a cost-based optimizer for filtered vector
  search, and instance-optimal top-k aggregation.

## 2. Data model and query language
- Records with media, text, vector, multi-vector and attribute fields
  (`omnisearch/schema.py`).
- Every modality becomes both text (captions / OCR / transcripts) and vectors
  (`omnisearch/ingest/pipeline.py`).
- OmniQL grammar and semantics: scoring clauses vs filters, three-valued logic
  for NULLs (`omnisearch/query/parser.py`, `filters.py`).

## 3. Storage and indexes
- Inverted index: gap + variable-byte compressed postings, BM25, WAND
  (`index/text/`). Figure: `keyword_index.png`.
- Vector indexes: HNSW and IVF from scratch, SQ8 / PQ compression, MaxSim
  multi-vector for video (`index/vector/`). Figures: `ann_recall_latency.png`,
  `quantization.png`.
- Attribute indexes: bitmaps, B+-tree, equi-depth histograms + MCVs
  (`index/attribute/`). Figure: `filters.png`.

## 4. Query processing and optimization
- Cost model with calibrated unit costs (`query/cost.py`) - compare with
  Postgres' `seq_page_cost` / `cpu_tuple_cost`.
- Access-path selection for filtered vector search: pre-filter exact scan vs
  in-graph filtering vs post-filtering (`query/optimizer.py`).
  Figure: `optimizer_selectivity.png`; table: correlated vs anti-correlated filters.
- Seq scan vs B+-tree for range predicates (`query/filters.py`).
- EXPLAIN / EXPLAIN ANALYZE output (estimated vs actual rows and time).

## 5. Rank fusion and top-k aggregation
- RRF vs weighted fusion vs query-adaptive AUTO (`query/fusion.py`).
- Threshold Algorithm and NRA (`query/topk.py`); why monotone aggregation
  matters; exactness vs fixed-depth fusion. Figure: `topk_algorithms.png`.

## 6. Evaluation
- Systems experiments (synthetic, controlled): `results/SUMMARY.md`.
- Retrieval quality on Flickr30k / COCO / MSR-VTT / Clotho:
  keyword vs vector vs hybrid (`benchmarks/retrieval_quality.py`).
- Baseline: PostgreSQL + pgvector (`baselines/pgvector/`), SQLite FTS5, FAISS.

## 7. Limitations and future work
- Pure Python: ~10x slower than FAISS per query; a C/Rust core is the obvious next step.
- Single writer; index files are checkpointed by `save()` (no WAL for indexes).
- Filtered HNSW degrades under anti-correlated filters - ACORN-style neighbour
  expansion would help.
- Captioner leakage on COCO; synthetic data for systems benchmarks.

## References
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009.
- Broder et al., *Efficient Query Evaluation using a Two-Level Retrieval Process* (WAND), CIKM 2003.
- Malkov & Yashunin, *Efficient and Robust ANN Search using HNSW Graphs*, TPAMI 2018.
- Jégou, Douze & Schmid, *Product Quantization for Nearest Neighbor Search*, TPAMI 2011.
- Fagin, Lotem & Naor, *Optimal Aggregation Algorithms for Middleware*, PODS 2001.
- Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet...*, SIGIR 2009.
- Wang et al., *Milvus: A Purpose-Built Vector Data Management System*, SIGMOD 2021.
- Wei et al., *AnalyticDB-V: A Hybrid Analytical Engine Towards Query Fusion for Structured and Unstructured Data*, VLDB 2020.
- Patel et al., *ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data*, SIGMOD 2024.
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021.
- Selinger et al., *Access Path Selection in a Relational Database Management System*, SIGMOD 1979.
