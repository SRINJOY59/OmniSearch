"""Where to get each benchmark dataset, and the layout the loaders expect.

    python scripts/download_data.py            # print instructions
    python scripts/download_data.py flickr30k  # print one dataset's steps

Most of these datasets require accepting a licence or a login, so this script
prints the steps instead of downloading silently.
"""
from __future__ import annotations

import sys

DATASETS = {
    "flickr30k": """Flickr30k (1,000 test images x 5 captions) - the easiest to start with.
  1. Images: request access at https://shannon.cs.illinois.edu/DenotationGraph/
     (or the Kaggle mirror "flickr30k") -> data/flickr30k/flickr30k-images/*.jpg
  2. Karpathy split: dataset_flickr30k.json from
     https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
     -> data/flickr30k/dataset_flickr30k.json""",
    "coco": """MS-COCO, Karpathy test split (5,000 images x 5 captions).
  1. Images: http://images.cocodataset.org/zips/val2014.zip (6 GB)
     -> data/coco/val2014/*.jpg  (test images all come from val2014)
  2. Karpathy split: dataset_coco.json from the same caption_datasets.zip as above
     -> data/coco/dataset_coco.json
  Note: BLIP-base captions were fine-tuned on COCO. Report COCO keyword/hybrid
  numbers with that caveat, or use Flickr30k.""",
    "msrvtt": """MSR-VTT, 1k-A test split (1,000 videos).
  1. Videos: MSRVTT.zip (e.g. from https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip)
     -> data/msrvtt/videos/all/video*.mp4
  2. MSRVTT_JSFUSION_test.csv (in the same archive under annotation/ or from the
     CLIP4Clip repo) -> data/msrvtt/MSRVTT_JSFUSION_test.csv""",
    "clotho": """Clotho v2.1 evaluation split (1,045 sound clips x 5 captions).
  https://zenodo.org/records/4783391
  -> data/clotho/evaluation/*.wav
  -> data/clotho/clotho_captions_evaluation.csv""",
    "abo": """Amazon Berkeley Objects (product listings + images; no relevance labels).
  https://amazon-berkeley-objects.s3.amazonaws.com/index.html
  -> data/abo/listings/metadata/listings_*.json.gz
  -> data/abo/images/metadata/images.csv.gz
  -> data/abo/images/small/<path from images.csv>""",
}


def main():
    names = sys.argv[1:] or list(DATASETS)
    for n in names:
        print(f"== {n}\n{DATASETS[n]}\n")
    print("Then run, e.g.:\n  python -m benchmarks.retrieval_quality --dataset flickr30k --root data/flickr30k")


if __name__ == "__main__":
    main()
