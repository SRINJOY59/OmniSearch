"""Retrieval quality on real benchmarks: keyword vs vector vs hybrid.

    python -m benchmarks.retrieval_quality --dataset flickr30k --root data/flickr30k
    python -m benchmarks.retrieval_quality --dataset coco --root data/coco --queries 5000
    python -m benchmarks.retrieval_quality --dataset msrvtt --root data/msrvtt
    python -m benchmarks.retrieval_quality --dataset clotho --root data/clotho

Needs the neural models (pip install -r requirements-models.txt) and the
dataset files (scripts/download_data.py). Ingest runs captioning / OCR /
Whisper + CLIP over the whole corpus; use a GPU, or precompute on Colab with
scripts/precompute.py and pass --records-jsonl to ingest without models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.common import save
from omnisearch.datasets import LOADERS
from omnisearch.evaluation import MODES, build, dataset_schema, evaluate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(set(LOADERS) - {"abo"}))
    ap.add_argument("--root", required=True)
    ap.add_argument("--encoder", default="clip", help="image-text encoder spec, e.g. clip or clip:ViT-B-16-SigLIP/webli")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--records", type=int, default=None, help="use only the first N corpus items")
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--collections", default="collections")
    ap.add_argument("--records-jsonl", default=None,
                    help="precomputed records (captions, OCR, vectors) from scripts/precompute.py")
    args = ap.parse_args(argv)

    ds = LOADERS[args.dataset](args.root)
    ds = ds.subset(args.records, args.queries)
    if args.records_jsonl:
        pre = {r["id"]: r for r in map(json.loads, open(args.records_jsonl, encoding="utf-8"))}
        ds.records = [pre.get(r["id"], r) for r in ds.records]
    print(ds.summary())
    schema, vec_field = dataset_schema(ds, args.encoder, args.dim)
    coll = build(ds, Path(args.collections) / schema.name, schema, progress=print)
    res = evaluate(coll, ds, vec_field, progress=print)
    print(f"\n{'mode':<16} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR@10':>7} {'nDCG@10':>8} {'p50 ms':>7}")
    for mode in MODES:
        r = res[mode]
        print(f"{mode:<16} {r['R@1']:6.3f} {r['R@5']:6.3f} {r['R@10']:6.3f} {r['MRR@10']:7.3f} {r['nDCG@10']:8.3f} {r['p50_ms']:7.2f}")
    save(f"quality_{args.dataset}", {"dataset": ds.summary(), "encoder": args.encoder, "results": res,
                                     "schema": schema.to_dict()})


if __name__ == "__main__":
    main()
