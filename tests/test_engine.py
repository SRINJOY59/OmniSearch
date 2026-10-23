"""End-to-end tests through the Collection API with real indexes."""
import numpy as np
import pytest
from PIL import Image

from omnisearch.engine import Collection, Engine
from omnisearch.query.cost import CostModel
from omnisearch.schema import AttributeField, MediaField, Schema, TextField, VectorField

COLORS = ["red", "green", "blue"]
CATS = ["shoes", "shirts", "mugs"]


@pytest.fixture(scope="module")
def coll(tmp_path_factory):
    root = tmp_path_factory.mktemp("coll")
    imgdir = root / "img"
    imgdir.mkdir()
    for c, rgb in {"red": (200, 30, 30), "green": (30, 160, 60), "blue": (30, 60, 200)}.items():
        Image.new("RGB", (16, 16), rgb).save(imgdir / f"{c}.png")
    schema = Schema("shop", [
        MediaField("image", "image"),
        TextField("text", sources=["title", "description"]),
        VectorField("img", 32, "fake", source="image", index="hnsw"),
        VectorField("emb", 16, "precomputed", source="emb", index="ivf", params={"nlist": 8}),
        AttributeField("price", "float"), AttributeField("category", "category"), AttributeField("tags", "tags"),
    ])
    rng = np.random.default_rng(1)
    c = Collection.create(root / "shop", schema)
    recs = []
    for i in range(600):
        col, cat = COLORS[i % 3], CATS[(i // 3) % 3]
        recs.append({"id": f"d{i}", "title": f"{col} {cat} number {i}", "description": "plain item" if i % 4 else "",
                     "image": f"img/{col}.png", "emb": rng.normal(size=16).tolist(),
                     "price": None if i % 50 == 0 else float(i % 100), "category": cat,
                     "tags": [col, "sale"] if i % 7 == 0 else [col]})
    c.add(recs, base_dir=root)
    c.build()
    return c


def ids(res):
    return [h.id for h in res.hits]


def test_keyword_and_filters(coll):
    res = coll.query("SEARCH shop WHERE MATCH(text, 'red shoes') AND price < 20 LIMIT 50")
    assert res.hits and all(h.record["price"] < 20 for h in res.hits)
    # BM25 ranks any doc matching a term; docs matching both terms come first
    assert all(set(h.matched_terms) & {"red", "shoe"} for h in res.hits)
    assert set(res.hits[0].matched_terms) == {"red", "shoe"}
    both = [len(h.matched_terms) == 2 for h in res.hits]
    assert both == sorted(both, reverse=True)


def test_three_valued_logic(coll):
    neg = coll.query("SEARCH shop WHERE NOT (price < 50) LIMIT 1000")
    assert all(h.record["price"] is not None and h.record["price"] >= 50 for h in neg.hits)
    nulls = coll.query("SEARCH shop WHERE price IS NULL LIMIT 1000")
    assert len(nulls.hits) == 12


def test_cross_modal_text_to_image(coll):
    res = coll.query("SEARCH shop WHERE SIMILAR(img, 'green') AND category = 'mugs' LIMIT 10")
    assert all(h.record["title"].startswith("green mugs") for h in res.hits)


def test_image_query_and_vector_arithmetic(coll, tmp_path):
    Image.new("RGB", (8, 8), (190, 40, 40)).save(tmp_path / "q.png")
    res = coll.query(f"SEARCH shop WHERE SIMILAR(img, IMAGE('{tmp_path / 'q.png'}')) LIMIT 5")
    assert all(h.record["title"].startswith("red") for h in res.hits)
    res = coll.query(f"SEARCH shop WHERE SIMILAR(img, IMAGE('{tmp_path / 'q.png'}') - 'red' + 'blue') LIMIT 5")
    assert all(h.record["title"].startswith("blue") for h in res.hits)


@pytest.mark.parametrize("fuse", ["RRF", "WEIGHTED", "AUTO"])
def test_hybrid_fusions(coll, fuse):
    res = coll.query(f"SEARCH shop WHERE MATCH(text, 'blue shirts') AND SIMILAR(img, 'blue') FUSE {fuse} LIMIT 10")
    assert len(res.hits) == 10
    assert all(h.record["title"].startswith("blue") for h in res.hits)
    assert set(res.hits[0].sources) == {"text", "img"}


@pytest.mark.parametrize("algo", ["full", "ta", "nra", "exhaustive"])
def test_topk_algorithms_agree(coll, algo):
    q = "SEARCH shop WHERE MATCH(text, 'red number') AND SIMILAR(emb, DOC('d5')) FUSE WEIGHTED LIMIT 5 WITH (topk='{}', strategy='exact')"
    ref = coll.query(q.format("exhaustive"))
    res = coll.query(q.format(algo))
    assert res.plan.fuse.algorithm == algo
    assert ids(res) == ids(ref)


def test_strategy_override_and_explain(coll):
    for s in ("exact", "index", "postfilter"):
        res = coll.query(f"SEARCH shop WHERE SIMILAR(img, 'red') AND tags = 'sale' LIMIT 5 WITH (strategy='{s}')")
        assert res.plan.branches[0].strategy == s
        assert all("sale" in h.record["tags"] for h in res.hits)
    text = coll.explain("SEARCH shop WHERE MATCH(text, 'red') AND SIMILAR(img, 'red') AND price < 30 LIMIT 3", analyze=True)
    assert "Fuse RRF" in text and "Filter: price < 30" in text and "actual" in text


def test_optimizer_prefers_exact_scan_for_selective_filters(coll):
    coll.cost = CostModel()  # defaults, so the test is machine independent
    res = coll.explain("EXPLAIN SEARCH shop WHERE SIMILAR(img, 'red') AND price = 7 LIMIT 5")
    assert "Pre-filter + Exact Vector Scan" in res


def test_contains_filter(coll):
    res = coll.query("SEARCH shop WHERE CONTAINS(text, 'plain item') AND category = 'shoes' LIMIT 1000")
    assert res.hits and all(h.record["description"] == "plain item" for h in res.hits)


def test_filter_only_query(coll):
    res = coll.query("SEARCH shop WHERE tags = 'sale' AND category IN ('mugs', 'shoes') LIMIT 3")
    assert len(res.hits) == 3 and res.plan.fuse.algorithm == "none"


def test_delete_save_reopen(coll, tmp_path):
    before = ids(coll.query("SEARCH shop WHERE MATCH(text, 'number 42') LIMIT 3"))
    assert "d42" in before
    assert coll.delete(["d42"]) == 1
    assert "d42" not in ids(coll.query("SEARCH shop WHERE MATCH(text, 'number 42') LIMIT 3"))
    coll.save()
    again = Collection.open(coll.path)
    assert again.n_alive == 599
    q = "SEARCH shop WHERE SIMILAR(emb, DOC('d7')) LIMIT 5"
    assert ids(again.query(q)) == ids(coll.query(q))
    assert ids(again.query("SEARCH shop WHERE MATCH(text, 'red shoes') LIMIT 5")) == \
        ids(coll.query("SEARCH shop WHERE MATCH(text, 'red shoes') LIMIT 5"))


def test_duplicate_ids_rejected(coll):
    with pytest.raises(ValueError):
        coll.add([{"id": "d1", "title": "dup"}])


def test_engine_routes_by_collection(coll):
    eng = Engine(coll.path.parent)
    assert "shop" in eng.names()
    assert eng.query("SEARCH shop WHERE MATCH(text, 'mugs') LIMIT 2").hits


def test_calibrate_produces_positive_costs(coll):
    cm = coll.calibrate()
    assert cm.calibrated and cm.scan_per_dim > 0 and cm.posting > 0 and cm.hnsw_expansion > 0
