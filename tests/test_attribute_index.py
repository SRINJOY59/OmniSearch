import numpy as np
import pytest

from omnisearch.index.attribute import BPlusTree, CategoryIndex, ColumnStats, NumericIndex


def brute(vals, lo, hi, li, hi_inc):
    m = np.ones(len(vals), bool)
    if lo is not None:
        m &= vals >= lo if li else vals > lo
    if hi is not None:
        m &= vals <= hi if hi_inc else vals < hi
    return set(np.flatnonzero(m).tolist())


@pytest.mark.parametrize("bulk", [False, True])
def test_bptree_ranges(rng, bulk):
    vals = rng.integers(0, 300, 5000).astype(float)
    t = BPlusTree(order=6)
    if bulk:
        t.bulk_load(zip(vals.tolist(), range(len(vals))))
    else:
        for i, v in enumerate(vals):
            t.insert(float(v), i)
    assert len(t) == 5000
    for args in [(10, 20, True, True), (10, 20, False, False), (None, 3, True, False),
                 (290, None, False, True), (150, 150, True, True), (None, None, True, True)]:
        assert set(t.range(*args)) == brute(vals, *args)


def test_bptree_ids_sorted_within_value():
    t = BPlusTree(order=4)
    for i in [5, 1, 9, 3]:
        t.insert(7.0, i)
    assert list(t.range(7, 7)) == [1, 3, 5, 9]


def test_histogram_on_skewed_data(rng):
    vals = np.concatenate([rng.exponential(40, 18000), np.full(2000, 12.0)])
    st = ColumnStats(vals)
    for lo, hi in [(0, 5), (5, 50), (50, 200), (12, 12), (None, 12)]:
        actual = ((vals >= (lo if lo is not None else -np.inf)) & (vals <= hi)).mean()
        assert st.range(lo, hi) == pytest.approx(actual, abs=0.02)
    assert st.eq(12.0) == pytest.approx(0.1, abs=0.01)


def test_numeric_index_scan_equals_tree(rng):
    ni = NumericIndex()
    vals = rng.normal(50, 20, 2000)
    for i, v in enumerate(vals):
        ni.add(i, None if i % 17 == 0 else v)
    n = 2000
    assert np.array_equal(ni.tree_range(30, 60, True, True, n), ni.scan_between(30, 60, n))
    assert np.array_equal(ni.tree_range(None, 40, True, False, n), ni.scan("<", 40, n))
    assert not ni.known(n)[0] and ni.stats.null_frac == pytest.approx(118 / 2000, abs=1e-3)


def test_category_bitmaps_and_tags():
    ci = CategoryIndex(multi=True)
    for i in range(6):
        ci.add(i, ["red", "sale"] if i % 2 else "blue")
    ci.add(6, None)
    assert ci.eq("sale", 7).tolist() == [False, True, False, True, False, True, False]
    assert ci.isin(["red", "blue"], 7).sum() == 6
    assert not ci.known(7)[6]
    restored = CategoryIndex()
    restored.load_arrays(ci.to_arrays())
    assert np.array_equal(restored.eq("red", 7), ci.eq("red", 7))
