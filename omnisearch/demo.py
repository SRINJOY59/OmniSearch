"""Schemas and a builder for the demo collection.

Two encoder profiles:
  classic - offline: hashed n-gram text vectors, colour-layout image vectors,
            OCR. Image search is query-by-example only.
  clip    - CLIP for images, video keyframes and text (cross-modal: type
            "red sneaker" and match pictures), plus BLIP captions and
            Whisper transcripts. Needs requirements-models.txt + downloads.
"""
from __future__ import annotations

import json
from pathlib import Path

from omnisearch.engine import Collection
from omnisearch.schema import (
    AttributeField, MediaField, MultiVectorField, Schema, TextField, VectorField,
)

TEXT_SOURCES = ["title", "description", "image.caption", "image.ocr", "video.ocr", "video.transcript"]
ATTRS = [AttributeField("price", "float"), AttributeField("year", "int"), AttributeField("category", "category"),
         AttributeField("brand", "category"), AttributeField("color", "tags")]


def demo_schema(profile: str = "classic", name: str = "demo") -> Schema:
    if profile == "classic":
        return Schema(name, [
            MediaField("image", "image", extract=["ocr"]),
            MediaField("video", "video", extract=["ocr"], frame_interval=2.0),
            TextField("text", sources=TEXT_SOURCES),
            VectorField("text_vec", 512, "hashed-ngram", source="text", index="hnsw"),
            VectorField("image_vec", 240, "color-layout", source="image", index="hnsw"),
            MultiVectorField("frames", 240, "color-layout", source="video"),
            *ATTRS,
        ])
    if profile == "clip":
        return Schema(name, [
            MediaField("image", "image", extract=["caption", "ocr"]),
            MediaField("video", "video", extract=["ocr", "transcript"], frame_interval=2.0),
            TextField("text", sources=TEXT_SOURCES),
            VectorField("image_vec", 512, "clip", source="image", index="hnsw"),
            MultiVectorField("frames", 512, "clip", source="video"),
            *ATTRS,
        ])
    raise ValueError("profile must be 'classic' or 'clip'")


def build_demo(data_dir="data/demo", out="collections", profile="classic", name="demo", progress=print) -> Collection:
    data_dir = Path(data_dir)
    records = [json.loads(line) for line in open(data_dir / "records.jsonl", encoding="utf-8")]
    coll = Collection.create(Path(out) / name, demo_schema(profile, name), overwrite=True)
    coll.add(records, base_dir=data_dir, batch_size=32,
             progress=(lambda i, n: progress(f"ingested {i}/{n}")) if progress else None)
    coll.save()
    coll.calibrate()
    return coll
