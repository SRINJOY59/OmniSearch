import numpy as np
import pytest

from omnisearch.index.vector import HNSWIndex, MultiVectorIndex, ProductQuantizer, ScalarQuantizer, VectorFieldIndex, VectorStore
from omnisearch.schema import VectorField


def exact_topk(x, q, k, allow=None):
    xs = x / np.linalg.norm(x, axis=1, keepdims=True)
    s = xs @ (q / np.linalg.norm(q))
    if allow is not None:
        s = np.where(allow, s, -np.inf)
    return set(np.argsort(-s)[:k].tolist())


def make(x, **kw):
    f = VectorFieldIndex(VectorField("v", x.shape[1], "precomputed", "v", **kw))
    f.add(np.arange(len(x)), x)
    f.build()
    return f


@pytest.mark.parametrize("spec", [dict(index="hnsw"), dict(index="ivf"), dict(index="flat"),
                                  dict(index="ivf", quantization="sq8"), dict(index="ivf", quantization="pq8"),
                                  dict(index="flat", quantization="pq16")])
def test_recall(clustered, spec):
    x, qs = clustered
    f = make(x, **spec)
    hits = 0
    for q in qs:
        ids, _, _ = f.search(f.prepare_query(q), 10, strategy="index")
        hits += len(exact_topk(x, q, 10) & set(ids.tolist()))
    assert hits / (10 * len(qs)) >= 0.85


@pytest.mark.parametrize("strategy", ["exact", "index", "postfilter"])
def test_filtered_strategies_only_return_allowed(clustered, rng, strategy):
    x, qs = clustered
    f = make(x, index="hnsw")
    allow = rng.random(len(x)) < 0.2
    for q in qs[:10]:
        ids, _, _ = f.search(f.prepare_query(q), 10, allow=allow, strategy=strategy, overfetch=200)
        assert allow[ids].all()
        if strategy == "exact":
            assert set(ids.tolist()) == exact_topk(x, q, 10, allow)


def test_scores_are_exact_for_random_access(clustered):
    x, qs = clustered
    f = make(x, index="hnsw")
    q = f.prepare_query(qs[0])
    ids = np.array([0, 5, 9])
    xs = x[ids] / np.linalg.norm(x[ids], axis=1, keepdims=True)
    assert np.allclose(f.score(q, ids), xs @ q, atol=1e-5)
    assert f.score(q, [10**6])[0] == -np.inf


def test_hnsw_graph_is_bounded_and_persists(clustered, tmp_path):
    x, qs = clustered
    store = VectorStore(32)
    store.add(x)
    h = HNSWIndex(store, M=8, ef_construction=64)
    h.sync()
    assert all(len(l[0]) <= 16 for l in h.links)
    assert all(len(lst) <= 8 for l in h.links for lst in l[1:])
    h.save(tmp_path / "h.npz")
    h2 = HNSWIndex.load(tmp_path / "h.npz", store)
    q = store.query(qs[0])
    assert np.array_equal(h.search(q, 10)[0], h2.search(q, 10)[0])


def test_field_index_save_load_with_quantization(clustered, tmp_path):
    x, qs = clustered
    for spec in [dict(index="ivf", quantization="pq8"), dict(index="flat", quantization="sq8"), dict(index="hnsw")]:
        f = make(x, **spec)
        f.save(tmp_path / "v.npz")
        g = VectorFieldIndex(f.field)
        g.load(tmp_path / "v.npz")
        q = f.prepare_query(qs[1])
        assert np.array_equal(f.search(q, 10)[0], g.search(q, 10)[0])


def test_quantizer_memory_and_error(clustered):
    x, _ = clustered
    sq, pq = ScalarQuantizer(32), ProductQuantizer(32, 8)
    sq.train(x)
    pq.train(x)
    sq.add(x)
    pq.add(x)
    assert sq.nbytes() == len(x) * 32 and pq.nbytes() == len(x) * 8
    err_sq = np.abs(sq.decode(sq.codes.data) - x).mean()
    err_pq = np.abs(pq.decode(pq.codes.data) - x).mean()
    assert err_sq < err_pq < np.abs(x).mean()


def test_multivector_maxsim_and_moment(rng):
    mv = MultiVectorIndex(8)
    vecs = rng.normal(size=(30, 8)).astype(np.float32)
    for d in range(10):
        mv.add(d, vecs[d * 3 : d * 3 + 3], [0.0, 2.0, 4.0])
    q = mv.store.query(vecs[7])  # doc 2's second frame
    ids, scores, _ = mv.search(q, 3)
    assert ids[0] == 2 and scores[0] == pytest.approx(1.0, abs=1e-5)
    assert mv.best_moment(q, 2) == 2.0
    allow = np.ones(10, bool)
    allow[2] = False
    assert 2 not in mv.search(q, 3, allow=allow)[0]
    assert mv.score(q, [2])[0] == pytest.approx(1.0, abs=1e-5)
