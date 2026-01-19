"""Statistical helper functions for estimators."""

from __future__ import annotations

import math

import numpy as np


def mean_stderr(values: np.ndarray) -> float:
    """Compute standard error of the mean.

    Estimand:
        Not applicable.
    Assumptions:
        Values are finite samples.
    Inputs:
        values: Array of sample values.
    Outputs:
        Standard error scalar.
    Failure modes:
        Returns 0 for n <= 1.
    """

    v = np.asarray(values, dtype=float)
    n = v.size
    if n <= 1:
        return 0.0
    return float(np.std(v, ddof=1) / math.sqrt(n))


def weighted_mean_and_stderr(
    values: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Compute weighted mean and an ESS-based standard error.

    Estimand:
        Not applicable.
    Assumptions:
        Weights are non-negative.
    Inputs:
        values: Array of sample values.
        weights: Array of weights.
    Outputs:
        Weighted mean and standard error.
    Failure modes:
        Returns 0 stderr for empty or zero-weight inputs.
    """

    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0:
        return 0.0, 0.0
    w_sum = w.sum()
    if w_sum <= 0:
        return float(np.mean(v)), 0.0
    w_norm = w / w_sum
    mean = float(np.sum(w_norm * v))
    ess = 1.0 / np.sum(w_norm**2)
    var = float(np.sum(w_norm * (v - mean) ** 2))
    stderr = math.sqrt(var / ess) if ess > 0 else 0.0
    return mean, stderr
