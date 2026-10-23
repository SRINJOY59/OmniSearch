"""Run every systems benchmark (no models or datasets needed), then plot.

    python -m benchmarks.run_all            # default sizes (~15-25 min on 4 cores)
    python -m benchmarks.run_all --quick    # small sizes (~2 min)
"""
from __future__ import annotations

import argparse
import time

from benchmarks import bench_ann, bench_filters, bench_optimizer, bench_quantization, bench_scale, bench_text, bench_topk

FULL = {
    "ann": [],
    "optimizer": [],
    "topk": [],
    "text": [],
    "filters": [],
    "quantization": [],
    "scale": [],
}
QUICK = {
    "ann": ["--n", "4000", "--queries", "50"],
    "optimizer": ["--n", "5000", "--queries", "15"],
    "topk": ["--n", "5000", "--queries", "10"],
    "text": ["--docs", "10000", "--queries", "50"],
    "filters": ["--n", "50000"],
    "quantization": ["--n", "10000", "--queries", "30"],
    "scale": ["--sizes", "2000", "5000", "--queries", "10"],
}
MODULES = {"ann": bench_ann, "optimizer": bench_optimizer, "topk": bench_topk, "text": bench_text,
           "filters": bench_filters, "quantization": bench_quantization, "scale": bench_scale}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", choices=sorted(MODULES))
    args = ap.parse_args(argv)
    cfg = QUICK if args.quick else FULL
    for name in args.only or MODULES:
        t0 = time.perf_counter()
        print(f"\n===== {name} =====", flush=True)
        MODULES[name].main(cfg[name])
        print(f"({name} took {time.perf_counter() - t0:.0f}s)", flush=True)
    from benchmarks import plots

    plots.main([])


if __name__ == "__main__":
    main()
