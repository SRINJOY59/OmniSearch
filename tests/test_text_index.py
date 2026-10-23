import numpy as np
import pytest

from omnisearch.index.text import Analyzer, InvertedIndex, PostingList, vbyte_decode, vbyte_encode
from omnisearch.index.text.analyzer import stem


def test_vbyte_roundtrip(rng):
    v = np.concatenate([rng.integers(0, 2**40, 2000), [0, 1, 127, 128, 16383, 16384, 2**35]])
    assert np.array_equal(vbyte_decode(vbyte_encode(v)), v)


def test_vbyte_small_values_use_one_byte():
    assert len(vbyte_encode(np.arange(128))) == 128


def test_posting_list_append_after_flush(rng):
    docs = np.sort(rng.choice(10**6, 4000, replace=False))
    pl = PostingList()
    for i, d in enumerate(docs):
        pl.add(int(d), i % 5 + 1)
        if i == 1500:
            pl.flush()
    d, tf = pl.arrays()
    assert np.array_equal(d, docs)
    assert np.array_equal(tf, np.arange(4000) % 5 + 1)
    with pytest.raises(ValueError):
        pl.add(int(docs[-1]), 1)


def test_analyzer_and_stemmer():
    a = Analyzer()
    assert a("The Running Shoes, 2 PAIRS!") == ["run", "shoe", "2", "pair"]
    assert stem("sneakers") == "sneaker"
    assert stem("batteries") == "battery"
    assert "no" in a("no parking")  # negations are kept


@pytest.fixture
def index(rng):
    vocab = [f"w{i}" for i in range(200)]
    p = 1 / np.arange(1, 201)
    p /= p.sum()
    idx = InvertedIndex()
    for d in range(1500):
        idx.add(d, " ".join(rng.choice(vocab, size=int(rng.integers(5, 30)), p=p)))
    return idx


def brute_bm25(idx, terms, doc):
    s = 0.0
    for t, qtf in terms:
        docs, tfs = idx.postings[t].arrays()
        hit = np.flatnonzero(docs == doc)
        if len(hit):
            tf = tfs[hit[0]]
            dl = idx.doc_len[doc]
            s += qtf * idx.idf(t) * tf * (idx.k1 + 1) / (tf + idx.k1 * (1 - idx.b + idx.b * dl / idx.avgdl))
    return s


def test_bm25_matches_formula(index):
    terms = index.query_terms("w3 w17 w150")
    ids, scores, _ = index.search(terms, 5)
    for d, s in zip(ids, scores):
        assert s == pytest.approx(brute_bm25(index, terms, d), rel=1e-4)


def test_wand_equals_exhaustive(index, rng):
    allow = rng.random(1500) < 0.4
    for q in ["w1 w9 w60", "w2 w3", "w100 w101 w102 w5"]:
        terms = index.query_terms(q)
        for a in (None, allow):
            i1, s1, st1 = index.search(terms, 10, allow=a)
            i2, s2, st2 = index.search(terms, 10, allow=a, algo="wand")
            assert np.allclose(s1, s2, atol=1e-5)
            assert st2["evaluated"] <= st1["evaluated"]
            if a is not None:
                assert a[i2].all()


def test_random_access_matches_search(index):
    terms = index.query_terms("w4 w44")
    ids, scores, _ = index.search(terms, 20)
    assert np.allclose(index.score_docs(terms, ids), scores, atol=1e-5)
    assert index.score_docs(terms, [10**6 - 1]).tolist() == [0.0]


def test_contains_all(index):
    ids = index.contains_all("w1 w2")
    for d in ids[:20]:
        assert {"w1", "w2"} <= set(index.analyzer(" ".join(t for t in index.postings if d in index.postings[t].arrays()[0])))
    assert len(index.contains_all("w1 nosuchterm")) == 0


def test_save_load(index, tmp_path):
    index.save(tmp_path / "t.osx")
    j = InvertedIndex.load(tmp_path / "t.osx")
    q = "w7 w70"
    assert np.array_equal(index.search(q, 10)[0], j.search(q, 10)[0])
    assert j.stats()["compressed_bytes"] < j.stats()["uncompressed_bytes"]
