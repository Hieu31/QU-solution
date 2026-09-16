from __future__ import annotations

import math
import statistics
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def latency_summary(values: list[float], *, official: bool, note: str = "") -> dict:
    total_seconds = sum(values) / 1000
    return {
        "samples": len(values), "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95), "p99_ms": percentile(values, 0.99),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "queries_per_second": len(values) / total_seconds if total_seconds else 0.0,
        "official": official, "note": note,
    }


def artifact_size(path: str | Path) -> int:
    root = Path(path)
    return root.stat().st_size if root.is_file() else sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
