"""Attribute filtering: B+-tree index scan vs vectorised column scan, and the
accuracy of selectivity estimates (equi-depth histogram + MCVs vs a naive
equi-width histogram).

    python -m benchmarks.bench_filters [--n 200000]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from benchmarks.common import save
from omnisearch.index.attribute import ColumnStats, NumericIndex
from omnisearch.query.cost import CostModel


class EquiWidth:
    def __init__(self, values, buckets=64):
        self.counts, self.edges = np.histogram(values, bins=buckets)
        self.n = len(values)

    def range(self, lo, hi):
        sel = 0.0
        for c, a, b in zip(self.counts, self.edges[:-1], self.edges[1:]):
            overlap = max(0.0, min(hi, b) - max(lo, a))
            if b > a:
                sel += c * overlap / (b - a)
        return sel / self.n


def qerror(est, act, floor=1e-4):
    est, act = max(est, floor), max(act, floor)
    return max(est / act, act / est)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    vals = np.round(rng.lognormal(3.5, 0.9, args.n), 2)
    vals[rng.random(args.n) < 0.08] = 19.99  # a heavy hitter, like a common price point

    idx = NumericIndex()
    idx.values.extend(vals)
    idx.rebuild_tree()
    cm = CostModel()
    rows = []
    sorted_vals = np.sort(vals)
    for sel in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3]:
        lo = sorted_vals[int(0.4 * args.n)]
        hi = sorted_vals[min(args.n - 1, int((0.4 + sel) * args.n))]
        reps = 5
        t = time.perf_counter()
        for _ in range(reps):
            a = idx.tree_range(lo, hi, True, True, args.n)
        tree_ms = (time.perf_counter() - t) / reps * 1e3
        t = time.perf_counter()
        for _ in range(reps):
            b = idx.scan_between(lo, hi, args.n)
        scan_ms = (time.perf_counter() - t) / reps * 1e3
        assert np.array_equal(a, b)
        actual = float(b.mean())
        est_tree = (cm.op_overhead + actual * args.n * cm.btree_row) * 1e3
        est_scan = args.n * cm.col_row * 1e3
        rows.append({"selectivity": actual, "tree_ms": tree_ms, "scan_ms": scan_ms,
                     "optimizer_choice": "index" if est_tree < est_scan else "scan"})
        print(f"sel={actual:.5f} btree={tree_ms:.3f}ms scan={scan_ms:.3f}ms -> {rows[-1]['optimizer_choice']}")

    stats, width = ColumnStats(vals), EquiWidth(vals)
    qe_depth, qe_width = [], []
    for _ in range(500):
        a, b = np.sort(rng.choice(vals, 2))
        actual = float(((vals >= a) & (vals <= b)).mean())
        if actual == 0:
            continue
        qe_depth.append(qerror(stats.range(a, b), actual))
        qe_width.append(qerror(width.range(a, b), actual))
    for _ in range(200):  # narrow ranges, where estimators struggle most
        a = rng.choice(vals)
        b = a + rng.uniform(0.1, 3)
        actual = float(((vals >= a) & (vals <= b)).mean())
        if actual == 0:
            continue
        qe_depth.append(qerror(stats.range(a, b), actual))
        qe_width.append(qerror(width.range(a, b), actual))

    def dist(q):
        q = np.asarray(q)
        return {"median": float(np.median(q)), "p90": float(np.percentile(q, 90)), "max": float(q.max())}

    est = {"equi_depth_mcv": dist(qe_depth), "equi_width": dist(qe_width)}
    print("q-error", est)
    save("filters", {"n": args.n, "access_paths": rows, "estimation_qerror": est,
                     "data": "log-normal prices with an 8% heavy hitter"})


if __name__ == "__main__":
    main()
