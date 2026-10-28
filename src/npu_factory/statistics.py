"""Paired inference comparisons: bootstrap accuracy-difference CI and exact McNemar test.

The unit of resampling is an image, not a model-inference timing sample.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def paired_accuracy_evidence(
    reference_correct: Sequence[bool],
    candidate_correct: Sequence[bool],
    *,
    seed: int = 2026,
    bootstrap_replicates: int = 5000,
    confidence: float = 0.95,
) -> dict:
    a = np.asarray(reference_correct, dtype=np.bool_)
    b = np.asarray(candidate_correct, dtype=np.bool_)
    if a.ndim != 1 or a.size < 2 or a.shape != b.shape:
        raise ValueError("Need paired 1D correctness for at least two images")
    if not 0 < confidence < 1 or bootstrap_replicates < 100:
        raise ValueError("Invalid confidence or number of bootstrap replicates")
    diff = b.astype(np.int8) - a.astype(np.int8)
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.empty(bootstrap_replicates, dtype=np.float64)
    for i in range(bootstrap_replicates):
        indices = rng.integers(0, n, size=n)
        boot[i] = float(diff[indices].mean())
    alpha = 1 - confidence
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    # Discordant counts: candidate exclusively wins vs reference exclusively wins.
    candidate_only = int(np.count_nonzero(b & ~a))
    reference_only = int(np.count_nonzero(a & ~b))
    discordant = candidate_only + reference_only
    if discordant == 0:
        p_value = 1.0
    else:
        extreme = min(candidate_only, reference_only)
        p_value = min(1.0, 2 * sum(
            math.comb(discordant, k) for k in range(extreme + 1)
        ) / (2 ** discordant))
    return {
        "count": n,
        "reference_accuracy": float(a.mean()),
        "candidate_accuracy": float(b.mean()),
        "candidate_minus_reference_pp": 100 * float(diff.mean()),
        "paired_bootstrap_confidence": confidence,
        "paired_bootstrap_replicates": bootstrap_replicates,
        "paired_bootstrap_low_pp": 100 * float(low),
        "paired_bootstrap_high_pp": 100 * float(high),
        "mcnemar_exact_two_sided_p": p_value,
        "discordant_candidate_only": candidate_only,
        "discordant_reference_only": reference_only,
        "bootstrap_seed": seed,
        "methodological_note": (
            "Percentile paired bootstrap over images; confidence interval is "
            "conditional on the fixed test set and model training run."
        ),
    }
