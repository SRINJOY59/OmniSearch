import csv
import json

import pytest

from omnisearch.datasets import clotho, karpathy, msrvtt
from omnisearch.datasets.base import RetrievalDataset
from omnisearch.engine import Collection
from omnisearch.evaluation import evaluate, mode_query, ndcg_at, mrr_at, recall_at
from omnisearch.schema import Schema, TextField, VectorField


def test_karpathy_loader_does_not_leak_captions(tmp_path):
    data = {"images": [
        {"filepath": "val2014", "filename": "a.jpg", "cocoid": 1, "split": "test",
         "sentences": [{"raw": "A dog runs."}, {"raw": "A brown dog."}]},
        {"filepath": "train2014", "filename": "b.jpg", "cocoid": 2, "split": "train", "sentences": [{"raw": "x"}]},
    ]}
    (tmp_path / "d.json").write_text(json.dumps(data))
    ds = karpathy(tmp_path / "d.json", tmp_path, "coco")
    assert ds.records == [{"id": "1", "image": "val2014/a.jpg"}]
    assert [q["text"] for q in ds.queries] == ["A dog runs.", "A brown dog."]
    assert ds.qrels["1_0"] == {"1": 1}


def test_msrvtt_and_clotho_loaders(tmp_path):
    with open(tmp_path / "MSRVTT_JSFUSION_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "vid_key", "video_id", "sentence"])
        w.writerow(["ret0", "msr9770", "video9770", "a person is cooking"])
    ds = msrvtt(tmp_path)
    assert ds.records[0]["video"] == "videos/all/video9770.mp4" and ds.queries[0]["text"] == "a person is cooking"
    with open(tmp_path / "clotho_captions_evaluation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name"] + [f"caption_{i}" for i in range(1, 6)])
        w.writerow(["rain.wav", "rain falls", "heavy rain", "water", "storm", "drops"])
    ds = clotho(tmp_path)
    assert ds.records[0] == {"id": "rain", "audio": "evaluation/rain.wav"} and len(ds.queries) == 5


def test_metrics():
    assert recall_at(["a", "b", "c"], {"c": 1}, 2) == 0.0 and recall_at(["a", "b", "c"], {"c": 1}, 3) == 1.0
    assert mrr_at(["a", "b"], {"b": 1}) == 0.5
    assert ndcg_at(["x", "y"], {"x": 1}) == 1.0
    assert ndcg_at(["y", "x"], {"x": 1}) == pytest.approx(1 / 1.585, rel=1e-3)


def test_evaluate_runs_all_modes(tmp_path):
    titles = ["red running shoe", "blue ceramic mug", "green cotton shirt", "black leather wallet"]
    ds = RetrievalDataset("toy", "product",
                          records=[{"id": f"d{i}", "title": t} for i, t in enumerate(titles)],
                          queries=[{"qid": "q0", "text": "running shoes"}, {"qid": "q1", "text": "mugg"}],
                          qrels={"q0": {"d0": 1}, "q1": {"d1": 1}})
    schema = Schema("toy", [TextField("text", sources=["title"]),
                            VectorField("vec", 512, "hashed-ngram", source="title", index="flat")])
    coll = Collection.create(tmp_path / "toy", schema)
    coll.add(ds.records)
    res = evaluate(coll, ds, "vec")
    assert set(res) == {"keyword", "vector", "hybrid-rrf", "hybrid-weighted", "hybrid-auto"}
    assert res["keyword"]["R@1"] == 0.5  # BM25 cannot match the typo "mugg"
    assert res["vector"]["R@1"] == 1.0  # character n-grams can
    assert "FUSE RRF" in mode_query("hybrid-rrf", "toy", "text", "vec", "a", 10)
