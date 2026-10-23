-- PostgreSQL baseline: the same hybrid search with built-in full-text search
-- (tsvector + GIN) and pgvector (HNSW). Dimension 128 matches the synthetic
-- benchmark data; change it for CLIP (512).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS docs (
    id      bigint PRIMARY KEY,
    body    text NOT NULL,
    tsv     tsvector GENERATED ALWAYS AS (to_tsvector('simple', body)) STORED,
    embedding vector(128) NOT NULL,
    price   double precision,
    bucket  integer
);
