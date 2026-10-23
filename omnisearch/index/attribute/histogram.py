"""Column statistics for selectivity estimation (what ANALYZE builds in Postgres).

* most-common values (MCV) with exact frequencies,
* an equi-depth histogram over the remaining values: every bucket holds the
  same number of rows, so skewed data gets narrow buckets where it is dense,
* null fraction and distinct count.

Range selectivity = MCV mass inside the range + histogram mass inside the
range, assuming values are uniform within a bucket.
"""
from __future__ import annotations

import numpy as np


class ColumnStats:
    def __init__(self, values: np.ndarray, buckets: int = 64, n_mcv: int = 16):
        values = np.asarray(values, dtype=np.float64)
        self.n = len(values)
        known = values[~np.isnan(values)]
        self.null_frac = 1.0 - len(known) / self.n if self.n else 0.0
        uniq, counts = np.unique(known, return_counts=True)
        self.n_distinct = len(uniq)
        order = np.argsort(-counts, kind="stable")[:n_mcv]
        # a value is "common" only if it is noticeably more frequent than average
        avg = len(known) / max(len(uniq), 1)
        common = order[counts[order] > 1.25 * avg] if len(uniq) > n_mcv else order
        self.mcv = {float(uniq[i]): counts[i] / self.n for i in common}
        rest = known[~np.isin(known, list(self.mcv))] if self.mcv else known
        self.rest_frac = len(rest) / self.n if self.n else 0.0
        self.rest_distinct = max(self.n_distinct - len(self.mcv), 1)
        if len(rest):
            self.bounds = np.quantile(rest, np.linspace(0, 1, min(buckets, len(rest)) + 1))
        else:
            self.bounds = np.empty(0)

    def eq(self, v) -> float:
        v = float(v)
        if v in self.mcv:
            return self.mcv[v]
        if not len(self.bounds) or v < self.bounds[0] or v > self.bounds[-1]:
            return 0.0
        return self.rest_frac / self.rest_distinct

    def _hist_le(self, v: float) -> float:
        """Fraction of non-MCV rows with value <= v (interpolated)."""
        b = self.bounds
        if not len(b) or v < b[0]:
            return 0.0
        if v >= b[-1]:
            return 1.0
        i = int(np.searchsorted(b, v, side="right")) - 1
        width = b[i + 1] - b[i]
        within = (v - b[i]) / width if width > 0 else 1.0
        return (i + within) / (len(b) - 1)

    def range(self, lo=None, hi=None, lo_inclusive=True, hi_inclusive=True) -> float:
        lo_v = -np.inf if lo is None else float(lo)
        hi_v = np.inf if hi is None else float(hi)
        sel = 0.0
        for v, f in self.mcv.items():
            if (v > lo_v or (lo_inclusive and v == lo_v)) and (v < hi_v or (hi_inclusive and v == hi_v)):
                sel += f
        hist = self._hist_le(hi_v) - (self._hist_le(lo_v) if lo is not None else 0.0)
        sel += max(hist, 0.0) * self.rest_frac
        return float(min(max(sel, 0.0), 1.0))
