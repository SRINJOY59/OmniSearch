"""Attribute indexes used for filtering.

* ``CategoryIndex`` - a bitmap per distinct value (also multi-valued "tags").
  Equality / IN predicates are bitmap ORs; exact counts come for free.
* ``NumericIndex`` - a dense column (for vectorised scans and random access),
  a B+-tree (for selective range lookups) and ``ColumnStats`` (for estimates).

Both return predicate results as boolean masks over doc ids.
"""
from __future__ import annotations

import numpy as np

from omnisearch.index.attribute.bptree import BPlusTree
from omnisearch.index.attribute.histogram import ColumnStats
from omnisearch.util import GrowableArray


class CategoryIndex:
    kind = "category"

    def __init__(self, multi: bool = False):
        self.multi = multi
        self.bitmaps: dict[str, GrowableArray] = {}
        self.present = GrowableArray(bool, fill=False)
        self.n = 0

    def add(self, doc_id: int, value):
        self.n = max(self.n, doc_id + 1)
        self.present.resize(self.n)
        if value is None:
            return
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for v in values:
            bm = self.bitmaps.get(str(v))
            if bm is None:
                bm = self.bitmaps[str(v)] = GrowableArray(bool, fill=False)
            bm.resize(self.n)
            bm[doc_id] = True
        self.present[doc_id] = True

    def pad(self, n: int):
        self.n = max(self.n, n)
        self.present.resize(self.n)

    def _mask(self, value, n) -> np.ndarray:
        out = np.zeros(n, dtype=bool)
        bm = self.bitmaps.get(str(value))
        if bm is not None:
            m = min(n, len(bm))
            out[:m] = bm.data[:m]
        return out

    def known(self, n) -> np.ndarray:
        out = np.zeros(n, dtype=bool)
        m = min(n, len(self.present))
        out[:m] = self.present.data[:m]
        return out

    def eq(self, value, n) -> np.ndarray:
        return self._mask(value, n)

    def isin(self, values, n) -> np.ndarray:
        out = np.zeros(n, dtype=bool)
        for v in values:
            out |= self._mask(v, n)
        return out

    def count(self, value) -> int:
        bm = self.bitmaps.get(str(value))
        return int(bm.data.sum()) if bm is not None else 0

    def selectivity(self, values) -> float:
        if not self.n:
            return 0.0
        return min(1.0, sum(self.count(v) for v in values) / self.n)

    def values(self) -> list[str]:
        return sorted(self.bitmaps)

    def nbytes(self) -> int:
        return sum((len(b) + 7) // 8 for b in self.bitmaps.values())  # packed size

    def to_arrays(self) -> dict:
        keys = sorted(self.bitmaps)
        return {"keys": np.array(keys, dtype=np.str_),
                "bits": np.stack([np.packbits(self._mask(k, self.n)) for k in keys]) if keys else np.empty((0, 0), np.uint8),
                "present": np.packbits(self.known(self.n)), "n": np.array([self.n])}

    def load_arrays(self, z):
        self.n = int(z["n"][0])
        self.present = GrowableArray.from_array(np.unpackbits(z["present"])[: self.n].astype(bool), fill=False)
        for k, bits in zip(z["keys"], z["bits"]):
            self.bitmaps[str(k)] = GrowableArray.from_array(np.unpackbits(bits)[: self.n].astype(bool), fill=False)


class NumericIndex:
    kind = "numeric"

    def __init__(self, order: int = 64):
        self.values = GrowableArray(np.float64, fill=np.nan)
        self.tree = BPlusTree(order)
        self._stats: ColumnStats | None = None
        self._stats_rows = 0

    def add(self, doc_id: int, value):
        self.values.resize(doc_id + 1)
        if value is None:
            return
        v = float(value)
        self.values[doc_id] = v
        self.tree.insert(v, doc_id)

    def pad(self, n: int):
        self.values.resize(n)

    def rebuild_tree(self):
        vals = self.values.data
        ok = np.flatnonzero(~np.isnan(vals))
        self.tree.bulk_load(zip(vals[ok].tolist(), ok.tolist()))

    @property
    def stats(self) -> ColumnStats:
        """Statistics, re-analyzed when the table grew by more than 10%."""
        n = len(self.values)
        if self._stats is None or n > 1.1 * self._stats_rows:
            self.analyze()
        return self._stats

    def analyze(self):
        self._stats = ColumnStats(self.values.data)
        self._stats_rows = len(self.values)

    def _col(self, n) -> np.ndarray:
        out = np.full(n, np.nan)
        m = min(n, len(self.values))
        out[:m] = self.values.data[:m]
        return out

    def known(self, n) -> np.ndarray:
        return ~np.isnan(self._col(n))

    def scan(self, op: str, value, n) -> np.ndarray:
        """Sequential (vectorised) column scan."""
        col = self._col(n)
        with np.errstate(invalid="ignore"):
            if op == "=":
                return col == value
            if op == "!=":
                return (col != value) & ~np.isnan(col)
            if op == "<":
                return col < value
            if op == "<=":
                return col <= value
            if op == ">":
                return col > value
            if op == ">=":
                return col >= value
        raise ValueError(op)

    def scan_between(self, lo, hi, n) -> np.ndarray:
        col = self._col(n)
        with np.errstate(invalid="ignore"):
            return (col >= lo) & (col <= hi)

    def tree_range(self, lo, hi, lo_inclusive, hi_inclusive, n) -> np.ndarray:
        """Index scan through the B+-tree."""
        out = np.zeros(n, dtype=bool)
        ids = np.fromiter(self.tree.range(lo, hi, lo_inclusive, hi_inclusive), dtype=np.int64)
        out[ids[ids < n]] = True
        return out

    def nbytes(self) -> int:
        return len(self.values) * 8
