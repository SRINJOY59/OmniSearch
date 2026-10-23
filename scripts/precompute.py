"""Run the expensive model steps once on a GPU (e.g. Google Colab) and write
records that can be ingested anywhere without models.

    # on Colab (GPU runtime):
    !git clone <your repo> && cd omnisearch && pip install -e . -r requirements-models.txt
    !python scripts/precompute.py --dataset flickr30k --root data/flickr30k --out flickr30k_records.jsonl

    # locally (no GPU, no model downloads for ingest):
    python -m benchmarks.retrieval_quality --dataset flickr30k --root data/flickr30k \\
        --records-jsonl flickr30k_records.jsonl

Each output record carries the derived text ("image.caption", "image.ocr",
"video.transcript", ...) and the vectors under the schema's field names; the
ingest pipeline reuses anything already present. Query-time text encoding
still needs the CLIP text tower, which runs fine on a CPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from omnisearch.datasets import LOADERS
from omnisearch.evaluation import dataset_schema
from omnisearch.ingest.pipeline import IngestPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="clip")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--records", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    ds = LOADERS[args.dataset](args.root)
    if args.records:
        ds = ds.subset(args.records)
    schema, _ = dataset_schema(ds, args.encoder, args.dim)
    pipe = IngestPipeline(schema)
    derived = [f"{m.name}.{x}" for m in schema.media_fields() for x in m.extract]
    with open(args.out, "w", encoding="utf-8") as f:
        for start in range(0, len(ds.records), args.batch):
            batch = ds.records[start : start + args.batch]
            processed = pipe.process(batch, base_dir=ds.root)
            for rec, p in zip(batch, processed):
                out = dict(rec)
                for key in derived:
                    if key in p["payload"]:
                        out[key] = p["payload"][key]
                    if key + ".segments" in p["payload"]:
                        out[key + ".segments"] = p["payload"][key + ".segments"]
                for name, vec in p["vectors"].items():
                    out[name] = np.asarray(vec).round(6).tolist()
                for name, (vecs, times) in p["multivectors"].items():
                    out[name] = {"vectors": np.asarray(vecs).round(6).tolist(), "times": np.asarray(times).tolist()}
                f.write(json.dumps(out) + "\n")
            print(f"{min(start + args.batch, len(ds.records))}/{len(ds.records)}", flush=True)
    print(f"wrote {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
