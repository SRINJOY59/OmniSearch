"""Shared helpers for benchmark scripts."""
from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULTS = Path(os.environ.get("OMNISEARCH_RESULTS", "results"))


def env_info() -> dict:
    import numpy

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def save(name: str, data: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps({"benchmark": name, "env": env_info(), **data}, indent=2, default=_default), encoding="utf-8")
    print(f"wrote {path}")
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1e3


def exact_topk(x: np.ndarray, q: np.ndarray, k: int, allow: np.ndarray | None = None) -> list[set]:
    """Ground truth by brute force (x, q normalised)."""
    s = q @ x.T
    if allow is not None:
        s[:, ~allow] = -np.inf
    k = min(k, x.shape[0])
    idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
    return [set(int(i) for i in r if np.isfinite(s[row, i])) for row, r in enumerate(idx)]


def recall(found, truth: set) -> float:
    return len(set(map(int, found)) & truth) / max(len(truth), 1)


def summary(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95))}


def vec_literal(v) -> str:
    return "VECTOR[" + ", ".join(f"{x:.6f}" for x in v) + "]"
