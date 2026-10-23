"""Retrieval quality on the offline demo collection: keyword vs vector vs hybrid.

    python -m benchmarks.demo_quality      # -> results/quality_demo.json, results/figures/quality_demo.png

The demo generator (scripts/make_demo_data.py) knows the ground truth for every
item, so relevance labels come from the generator, never from the engine.
Five query sets probe different paths into the index:

    brand+category   "zenith sneaker"        words present in titles
    colour+category  "red mug"               a third of titles omit the colour
    typos            "zenth sneker"          only the n-gram vectors can match
    sign text (OCR)  "no parking"            the text exists only inside the image
    lecture (OCR)    "extendible hashing"    the text exists only on video slides

plus query-by-example on the colour-layout image vectors. Uses the "classic"
profile (no neural models), so text queries cannot reach image pixels directly.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

from benchmarks.common import save
from omnisearch.datasets.base import RetrievalDataset
from omnisearch.engine import Collection
from omnisearch.evaluation import MODES, evaluate

ROOT = Path(__file__).resolve().parent.parent
NOUN = {"sneakers": "sneaker", "t-shirts": "t-shirt", "mugs": "mug", "backpacks": "backpack", "caps": "cap"}


def generator():
    spec = importlib.util.spec_from_file_location("make_demo_data", ROOT / "scripts" / "make_demo_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def typo(word: str) -> str:
    """Drop the middle letter of every alphabetic word of 4+ letters."""
    return word[: len(word) // 2] + word[len(word) // 2 + 1:] if word.isalpha() and len(word) >= 4 else word


def query_sets(records, gen) -> dict[str, RetrievalDataset]:
    products = [r for r in records if r["id"].startswith("p")]
    sets = {}

    def make(name, items):
        ds = RetrievalDataset(name, "product")
        for i, (text, rel) in enumerate(items):
            if rel:
                ds.queries.append({"qid": f"{name}-{i}", "text": text})
                ds.qrels[f"{name}-{i}"] = {d: 1 for d in rel}
        sets[name] = ds

    pairs = sorted({(p["brand"], p["category"]) for p in products})
    brand_cat = [(f"{b.lower()} {NOUN[c]}", [p["id"] for p in products if p["brand"] == b and p["category"] == c])
                 for b, c in pairs]
    make("brand+category", brand_cat)
    make("colour+category", [(f"{col} {NOUN[c]}", [p["id"] for p in products if col in p["color"] and p["category"] == c])
                             for col, c in itertools.product(sorted(gen.COLORS), sorted(NOUN))])
    make("typos", [(" ".join(typo(w) for w in q.split()), rel) for q, rel in brand_cat])
    make("sign text (OCR)", [(text.lower(), [f"s{j:03d}" for j in range(len(gen.SIGNS) * 3) if gen.SIGNS[j % len(gen.SIGNS)][0] == text])
                             for text, _ in gen.SIGNS])
    make("lecture (OCR)", [(slide.lower(), [vid]) for vid, _, slides in gen.LECTURES for slide in slides[1:]])
    return sets


def query_by_example(coll: Collection, products, n=60, k=10, seed=0) -> dict:
    """SIMILAR(image_vec, DOC(id)): how many of the top k share the query's colour / category."""
    rng = np.random.default_rng(seed)
    by_id = {p["id"]: p for p in products}
    same_colour, same_cat = [], []
    for pid in rng.choice(sorted(by_id), n, replace=False):
        q = by_id[pid]
        hits = [h.id for h in coll.query(f"SEARCH {coll.name} WHERE SIMILAR(image_vec, DOC('{pid}')) LIMIT {k + 1}").hits
                if h.id != pid][:k]
        hits = [by_id[h] for h in hits if h in by_id]
        same_colour.append(np.mean([q["color"][0] in h["color"] for h in hits]) if hits else 0.0)
        same_cat.append(np.mean([h["category"] == q["category"] for h in hits]) if hits else 0.0)
    return {"queries": n, "k": k, "P@10_same_colour": float(np.mean(same_colour)),
            "P@10_same_category": float(np.mean(same_cat)),
            "chance_colour": 1 / len({p["color"][0] for p in products}),
            "chance_category": 1 / len({p["category"] for p in products})}


def main():
    records = [json.loads(line) for line in open(ROOT / "data" / "demo" / "records.jsonl", encoding="utf-8")]
    coll = Collection.open(ROOT / "collections" / "demo")
    gen = generator()
    out = {}
    print(f"{'query set':<18} {'mode':<16} {'n':>3} {'R@1':>6} {'R@10':>6} {'MRR@10':>7} {'nDCG@10':>8} {'p50 ms':>7}")
    for name, ds in query_sets(records, gen).items():
        res = evaluate(coll, ds, "text_vec", modes=MODES)
        out[name] = {"queries": len(ds.queries), "example": ds.queries[0]["text"], "results": res}
        for mode in MODES:
            r = res[mode]
            print(f"{name:<18} {mode:<16} {r['queries']:>3} {r['R@1']:6.3f} {r['R@10']:6.3f} {r['MRR@10']:7.3f} "
                  f"{r['nDCG@10']:8.3f} {r['p50_ms']:7.2f}")
    qbe = query_by_example(coll, [r for r in records if r["id"].startswith("p")])
    print("query by example:", qbe)
    save("quality_demo", {"collection": coll.stats()["docs"], "vector_field": "text_vec", "sets": out,
                          "query_by_example": qbe})
    figure(out)


def figure(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from benchmarks.plots import INK2, S1, S2, S3, save as save_fig, style

    style()
    sets = list(out)
    modes = [("keyword", S1, "keyword (BM25)"), ("vector", S2, "vector (n-gram)"), ("hybrid-rrf", S3, "hybrid RRF"),
             ("hybrid-weighted", "#8b5cf6", "hybrid weighted"), ("hybrid-auto", INK2, "hybrid AUTO")]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    w = 0.16
    x = np.arange(len(sets))
    for i, (m, color, label) in enumerate(modes):
        vals = [out[s]["results"][m]["nDCG@10"] for s in sets]
        ax.bar(x + (i - 2) * w, vals, w, color=color, label=label)
    ax.set_xticks(x, [f"{s}\n(n={out[s]['queries']})" for s in sets])
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=9)
    ax.set_title("Retrieval quality on the demo collection (268 docs, classic offline encoders)", pad=26)
    print("figure:", save_fig(fig, "quality_demo"))


if __name__ == "__main__":
    main()
