import numpy as np
import pytest

from omnisearch.query.fusion import rrf
from omnisearch.query.topk import Source, nra, threshold_algorithm
from omnisearch.util import topk


def make_sources(rng, n=2000, m=3, corr=0.5, weights=None):
    base = rng.random(n)
    cols = [corr * base + (1 - corr) * rng.random(n) for _ in range(m)]
    weights = weights or [1.0] * m

    def src(j):
        col = cols[j]
        order = np.argsort(-col)
        fetch = lambda k: (order[:k], col[order[:k]])
        return Source(f"s{j}", weights[j], fetch, lambda ids: col[ids], norm="max")
    return cols, [src(j) for j in range(m)], weights


def reference(cols, weights, k):
    total = sum(w * c / c.max() for w, c in zip(weights, cols))
    return topk(total, k).tolist()


@pytest.mark.parametrize("corr", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("k", [1, 10, 50])
def test_ta_is_exact(rng, corr, k):
    cols, sources, w = make_sources(rng, corr=corr, weights=[0.5, 0.3, 0.2])
    ids, scores, st = threshold_algorithm(sources, k)
    assert ids == reference(cols, w, k)
    assert st["docs_seen"] < 2000 or corr == 0.0


@pytest.mark.parametrize("corr", [0.5, 0.9])
def test_nra_finds_topk_set(rng, corr):
    cols, sources, w = make_sources(rng, corr=corr)
    ids, _, st = nra(sources, 10)
    assert set(ids) == set(reference(cols, w, 10))
    assert st["random_accesses"] == 0


def test_ta_reads_less_when_sources_agree(rng):
    _, s_lo, _ = make_sources(np.random.default_rng(1), corr=0.1)
    _, s_hi, _ = make_sources(np.random.default_rng(1), corr=0.95)
    lo = threshold_algorithm(s_lo, 10)[2]["depth"]
    hi = threshold_algorithm(s_hi, 10)[2]["depth"]
    assert hi < lo


def test_rrf_formula():
    lists = {"a": (np.array([1, 2, 3]), None), "b": (np.array([3, 1]), None)}
    fused, ranks = rrf(lists, {"a": 1.0, "b": 2.0}, k=60)
    assert fused[1] == pytest.approx(1 / 61 + 2 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 2 / 61)
    assert ranks[2] == {"a": 2}
