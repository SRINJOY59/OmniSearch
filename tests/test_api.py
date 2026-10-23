import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from omnisearch.api.builder import build_query
from omnisearch.api.server import create_app
from omnisearch.engine import Collection
from omnisearch.schema import AttributeField, MediaField, Schema, TextField, VectorField


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("api")
    (root / "img").mkdir()
    for c, rgb in {"red": (200, 30, 30), "blue": (30, 60, 200)}.items():
        Image.new("RGB", (16, 16), rgb).save(root / "img" / f"{c}.png")
    schema = Schema("shop", [
        MediaField("image", "image"), TextField("text", sources=["title"]),
        VectorField("img", 32, "fake", source="image", index="flat"),
        AttributeField("price", "float"), AttributeField("category", "category"),
    ])
    c = Collection.create(root / "cols" / "shop", schema)
    c.add([{"id": f"d{i}", "title": f"{col} item {i}", "image": f"img/{col}.png", "price": i, "category": cat}
           for i, (col, cat) in enumerate([("red", "a"), ("blue", "b")] * 10)], base_dir=root)
    c.save()
    return TestClient(create_app(root / "cols"))


def test_list_and_describe(client):
    assert client.get("/api/collections").json() == [{"name": "shop", "docs": 20}]
    d = client.get("/api/collections/shop").json()
    assert dict(d["facets"]["category"]) == {"a": 10, "b": 10}
    assert d["ranges"]["price"] == [0.0, 19.0]


def test_query_endpoint_and_errors(client):
    r = client.post("/api/collections/shop/query", json={"query": "SEARCH shop WHERE MATCH(text, 'red') LIMIT 3"})
    assert r.status_code == 200 and len(r.json()["hits"]) == 3 and "explain" in r.json()
    bad = client.post("/api/collections/shop/query", json={"query": "SEARCH shop WHERE"})
    assert bad.status_code == 400 and "error" in bad.json()
    assert client.post("/api/collections/nope/query", json={"query": "SEARCH nope"}).status_code == 404


def test_server_files_cannot_be_used_as_query_media(client):
    r = client.post("/api/collections/shop/query",
                    json={"query": "SEARCH shop WHERE SIMILAR(img, IMAGE('/etc/hostname')) LIMIT 3"})
    assert r.status_code == 400


def test_search_form_with_image_and_filters(client):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (40, 60, 190)).save(buf, format="PNG")
    r = client.post("/api/collections/shop/search",
                    data={"text": "", "filters": json.dumps({"price": {"max": 9}}), "fusion": "rrf", "limit": "5"},
                    files={"image": ("q.png", buf.getvalue(), "image/png")})
    body = r.json()
    assert r.status_code == 200, body
    assert "IMAGE(" in body["omniql"] and "price <= 9" in body["omniql"]
    assert all(h["record"]["title"].startswith("blue") and h["record"]["price"] <= 9 for h in body["hits"])


def test_media_and_document_crud(client):
    assert client.get("/api/collections/shop/media/d0/image").headers["content-type"] == "image/png"
    assert client.get("/api/collections/shop/media/d0/title").status_code == 404
    r = client.post("/api/collections/shop/documents", data={"metadata": json.dumps({"id": "new", "title": "green thing"})})
    assert r.json() == {"ids": ["new"]}
    assert client.get("/api/collections/shop/documents/new").json()["title"] == "green thing"
    assert client.delete("/api/collections/shop/documents/new").status_code == 200
    assert client.get("/api/collections/shop/documents/new").status_code == 404


def test_builder_quotes_and_ranges(client):
    coll = client.app.state.engine.collection("shop")
    q = build_query(coll, "it's red", None, {"price": {"min": 1, "max": 5}, "category": ["a"]}, "auto", 4)
    assert "MATCH(text, 'it''s red')" in q and "price BETWEEN 1 AND 5" in q and "category IN ('a')" in q
    assert "SIMILAR(img, 'it''s red')" in q and q.endswith("LIMIT 4")
