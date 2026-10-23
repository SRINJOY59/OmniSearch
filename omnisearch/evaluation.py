"""Retrieval-quality evaluation: keyword vs vector vs hybrid.

For each query the same collection is searched in several modes:

    keyword          MATCH(text, q)                      BM25 over captions/OCR/transcripts
    vector           SIMILAR(<vec>, q)                   cross-modal embedding search
    hybrid-rrf       MATCH + SIMILAR, FUSE RRF
    hybrid-weighted  MATCH + SIMILAR, FUSE WEIGHTED
    hybrid-auto      MATCH + SIMILAR, FUSE AUTO

and scored with Recall@{1,5,10}, MRR@10 and nDCG@10 against the dataset's qrels.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from omnisearch.datasets.base import RetrievalDataset
from omnisearch.engine import Collection
from omnisearch.schema import AttributeField, MediaField, MultiVectorField, Schema, TextField, VectorField


def recall_at(ranked, relevant, k) -> float:
    return 1.0 if any(d in relevant for d in ranked[:k]) else 0.0


def mrr_at(ranked, relevant, k=10) -> float:
    for i, d in enumerate(ranked[:k], 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked, rels: dict, k=10) -> float:
    dcg = sum((2 ** rels.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1))
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def _q(s):
    return "'" + s.replace("'", "''") + "'"


def mode_query(mode: str, coll: str, text_field: str, vec_field: str, q: str, k: int) -> str:
    m, s = f"MATCH({text_field}, {_q(q)})", f"SIMILAR({vec_field}, {_q(q)})"
    where = {"keyword": m, "vector": s}.get(mode, f"{m} AND {s}")
    fuse = {"hybrid-rrf": " FUSE RRF", "hybrid-weighted": " FUSE WEIGHTED", "hybrid-auto": " FUSE AUTO"}.get(mode, "")
    return f"SEARCH {coll} WHERE {where}{fuse} LIMIT {k}"


MODES = ("keyword", "vector", "hybrid-rrf", "hybrid-weighted", "hybrid-auto")


def evaluate(coll: Collection, ds: RetrievalDataset, vec_field: str, text_field: str = "text",
             modes=MODES, k: int = 10, progress=None) -> dict:
    results = {}
    for mode in modes:
        rows, lat = [], []
        for i, q in enumerate(ds.queries):
            t0 = time.perf_counter()
            res = coll.query(mode_query(mode, coll.name, text_field, vec_field, q["text"], k))
            lat.append((time.perf_counter() - t0) * 1e3)
            ranked = [h.id for h in res.hits]
            rel = ds.qrels[q["qid"]]
            rows.append([recall_at(ranked, rel, 1), recall_at(ranked, rel, 5), recall_at(ranked, rel, 10),
                         mrr_at(ranked, rel, k), ndcg_at(ranked, rel, k)])
            if progress and (i + 1) % 200 == 0:
                progress(f"{mode}: {i + 1}/{len(ds.queries)}")
        m = np.mean(rows, axis=0) if rows else np.zeros(5)
        results[mode] = {"R@1": m[0], "R@5": m[1], "R@10": m[2], "MRR@10": m[3], "nDCG@10": m[4],
                         "p50_ms": float(np.percentile(lat, 50)) if lat else 0.0,
                         "p95_ms": float(np.percentile(lat, 95)) if lat else 0.0, "queries": len(rows)}
    return results


def dataset_schema(ds: RetrievalDataset, encoder: str = "clip", dim: int = 512) -> tuple[Schema, str]:
    """Schema for a benchmark dataset, and the vector field to query.

    encoder/dim select the image-text model, e.g. ("clip", 512) or
    ("clip:ViT-B-16-SigLIP/webli", 768). Audio always uses CLAP.
    """
    name = ds.name.replace("-", "_")
    img_enc = encoder
    if ds.modality == "image":
        return Schema(name, [
            MediaField("image", "image", extract=["caption", "ocr"]),
            TextField("text", sources=["image.caption", "image.ocr"]),
            VectorField("image_vec", dim, img_enc, source="image", index="hnsw"),
        ], models={"caption": "caption"}), "image_vec"
    if ds.modality == "video":
        return Schema(name, [
            MediaField("video", "video", extract=["transcript", "ocr"], frame_interval=1.0),
            TextField("text", sources=["video.transcript", "video.ocr"]),
            MultiVectorField("frames", dim, img_enc, source="video"),
        ]), "frames"
    if ds.modality == "audio":
        return Schema(name, [
            MediaField("audio", "audio", extract=["transcript"]),
            TextField("text", sources=["audio.transcript"]),
            VectorField("audio_vec", 512, "clap", source="audio", index="flat"),
        ]), "audio_vec"
    return Schema(name, [
        MediaField("image", "image", extract=["ocr"]),
        TextField("text", sources=["title", "description", "image.ocr"]),
        VectorField("image_vec", dim, img_enc, source="image", index="hnsw"),
        AttributeField("category", "category"), AttributeField("brand", "category"), AttributeField("color", "tags"),
    ]), "image_vec"


def build(ds: RetrievalDataset, out: Path, schema: Schema, progress=None) -> Collection:
    out = Path(out)
    if (out / "schema.json").exists():
        coll = Collection.open(out)
        if coll.n_alive == len(ds.records):
            return coll
    coll = Collection.create(out, schema, overwrite=True)
    coll.add(ds.records, base_dir=ds.root, batch_size=64,
             progress=(lambda i, n: progress(f"ingest {i}/{n}")) if progress else None)
    coll.save()
    return coll
