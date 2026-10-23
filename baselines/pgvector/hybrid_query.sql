-- Hybrid search in one SQL statement: BM25-like full-text ranking and HNSW
-- vector search, merged with Reciprocal Rank Fusion (k = 60).
-- Parameters: $1 = tsquery text, $2 = query vector, $3 = filter upper bound, $4 = limit
-- Note: Postgres' ts_rank_cd is not BM25 (no IDF, different length normalisation).
WITH keyword AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
    FROM docs, to_tsquery('simple', $1) q
    WHERE tsv @@ q AND bucket < $3
    ORDER BY ts_rank_cd(tsv, q) DESC
    LIMIT 100
),
semantic AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> $2) AS r
    FROM docs
    WHERE bucket < $3
    ORDER BY embedding <=> $2
    LIMIT 100
)
SELECT id, SUM(1.0 / (60 + r)) AS rrf
FROM (SELECT * FROM keyword UNION ALL SELECT * FROM semantic) u
GROUP BY id
ORDER BY rrf DESC
LIMIT $4;
