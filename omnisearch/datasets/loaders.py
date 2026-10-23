"""Loaders for public multimodal retrieval benchmarks.

Each loader reads the files as distributed (see scripts/download_data.py for
where to get them) and returns a RetrievalDataset for text-to-X retrieval.

* COCO / Flickr30k - Karpathy splits (``dataset_coco.json`` /
  ``dataset_flickr30k.json``): 5,000 / 1,000 test images, 5 captions each.
* MSR-VTT 1k-A - ``MSRVTT_JSFUSION_test.csv``: 1,000 test videos, 1 query each.
* Clotho v2 evaluation - ``clotho_captions_evaluation.csv``: 1,045 audio clips,
  5 captions each.
* Amazon Berkeley Objects - product listings + images (no qrels; used for
  filters, scale and demos).
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from omnisearch.datasets.base import RetrievalDataset


def karpathy(json_path, image_root, name: str, split: str = "test") -> RetrievalDataset:
    """COCO or Flickr30k in the Karpathy split format."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    ds = RetrievalDataset(name, "image", root=str(image_root))
    for img in data["images"]:
        if img.get("split") != split:
            continue
        rid = str(img.get("cocoid", img.get("imgid", img["filename"])))
        rel = Path(img.get("filepath", "")) / img["filename"]
        ds.records.append({"id": rid, "image": rel.as_posix()})
        for j, sent in enumerate(img["sentences"]):
            qid = f"{rid}_{j}"
            ds.queries.append({"qid": qid, "text": sent["raw"].strip()})
            ds.qrels[qid] = {rid: 1}
    return ds


def coco(root="data/coco", split="test") -> RetrievalDataset:
    root = Path(root)
    return karpathy(root / "dataset_coco.json", root, "coco", split)


def flickr30k(root="data/flickr30k", split="test") -> RetrievalDataset:
    root = Path(root)
    return karpathy(root / "dataset_flickr30k.json", root / "flickr30k-images", "flickr30k", split)


def msrvtt(root="data/msrvtt") -> RetrievalDataset:
    root = Path(root)
    ds = RetrievalDataset("msrvtt-1kA", "video", root=str(root))
    with open(root / "MSRVTT_JSFUSION_test.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["video_id"]
            ds.records.append({"id": vid, "video": f"videos/all/{vid}.mp4"})
            qid = f"q_{vid}"
            ds.queries.append({"qid": qid, "text": row["sentence"].strip()})
            ds.qrels[qid] = {vid: 1}
    return ds


def clotho(root="data/clotho", split="evaluation") -> RetrievalDataset:
    root = Path(root)
    ds = RetrievalDataset(f"clotho-{split}", "audio", root=str(root))
    with open(root / f"clotho_captions_{split}.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = Path(row["file_name"]).stem
            ds.records.append({"id": rid, "audio": f"{split}/{row['file_name']}"})
            for j in range(1, 6):
                cap = row.get(f"caption_{j}")
                if cap:
                    qid = f"{rid}_{j}"
                    ds.queries.append({"qid": qid, "text": cap.strip()})
                    ds.qrels[qid] = {rid: 1}
    return ds


def _first(values, lang_prefix="en"):
    if not values:
        return None
    for v in values:
        if str(v.get("language_tag", "en")).startswith(lang_prefix):
            return v.get("value")
    return values[0].get("value")


def abo(root="data/abo", limit: int | None = None) -> RetrievalDataset:
    """Amazon Berkeley Objects listings joined with image paths."""
    root = Path(root)
    images = {}
    with gzip.open(root / "images" / "metadata" / "images.csv.gz", "rt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            images[row["image_id"]] = row["path"]
    ds = RetrievalDataset("abo", "product", root=str(root / "images" / "small"))
    for path in sorted((root / "listings" / "metadata").glob("listings_*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                img = images.get(item.get("main_image_id", ""))
                title = _first(item.get("item_name"))
                if not img or not title:
                    continue
                ptype = (item.get("product_type") or [{}])[0].get("value")
                color = _first(item.get("color"))
                ds.records.append({
                    "id": item["item_id"], "title": title, "image": img,
                    "brand": _first(item.get("brand")), "category": ptype,
                    "color": [color.lower()] if color else None,
                    "description": " ".join(v["value"] for v in item.get("bullet_point", [])
                                            if str(v.get("language_tag", "en")).startswith("en"))[:1000],
                })
                if limit and len(ds.records) >= limit:
                    return ds
    return ds


LOADERS = {"coco": coco, "flickr30k": flickr30k, "msrvtt": msrvtt, "clotho": clotho, "abo": abo}
