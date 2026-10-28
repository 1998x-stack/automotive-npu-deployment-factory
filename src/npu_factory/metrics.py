"""Pure numerical comparison and latency statistics."""
from __future__ import annotations

import math
from statistics import mean

import numpy as np


def numerical_difference(reference: np.ndarray, candidate: np.ndarray) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} != {candidate.shape}")
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite model output")
    diff = np.abs(a - b)
    return {
        "max_absolute_error": float(diff.max()),
        "mean_absolute_error": float(diff.mean()),
        "max_relative_error_atol_1e-6": float((diff / np.maximum(np.abs(a), 1e-6)).max()),
        "top1_equal": bool(np.argmax(a, axis=-1).tolist() == np.argmax(b, axis=-1).tolist()),
    }


def classification(scores: list[np.ndarray], labels: list[int]) -> dict:
    if not scores or len(scores) != len(labels):
        raise ValueError("Nonempty aligned scores and labels are required")
    top1 = top5 = 0
    for values, label in zip(scores, labels):
        v = np.asarray(values).reshape(-1)
        if v.size < 5 or not np.isfinite(v).all():
            raise ValueError("Expected at least five finite class scores")
        if label < 0 or label >= v.size:
            raise ValueError("Label outside class score range")
        top1 += int(int(np.argmax(v)) == label)
        top5 += int(label in np.argsort(v)[-5:])
    return {"count": len(labels), "top1": top1 / len(labels), "top5": top5 / len(labels)}


def latency_stats(milliseconds: list[float]) -> dict:
    if not milliseconds or any(x < 0 or not math.isfinite(x) for x in milliseconds):
        raise ValueError("Expected nonempty, nonnegative finite latency observations")
    return {
        "samples": len(milliseconds),
        "mean_ms": mean(milliseconds),
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "p99_ms": float(np.percentile(milliseconds, 99)),
        "min_ms": min(milliseconds),
        "max_ms": max(milliseconds),
    }
