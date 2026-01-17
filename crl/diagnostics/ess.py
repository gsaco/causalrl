"""Effective sample size diagnostics."""

from __future__ import annotations

import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute effective sample size from non-negative weights.

    Estimand:
        Not applicable.
    Assumptions:
        Weights are non-negative.
    Inputs:
        weights: Array of importance weights.
    Outputs:
        Effective sample size scalar.
    Failure modes:
        Returns 0 if weights are empty or sum to zero.
    """

    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return 0.0
    w_sum = w.sum()
    if w_sum <= 0:
        return 0.0
    w_norm = w / w_sum
    return float(1.0 / np.sum(w_norm**2))


def ess_ratio(weights: np.ndarray) -> float:
    """Compute ESS divided by sample count.

    Estimand:
        Not applicable.
    Assumptions:
        Weights are non-negative.
    Inputs:
        weights: Array of importance weights.
    Outputs:
        ESS ratio scalar.
    Failure modes:
        Returns 0 if weights are empty or sum to zero.
    """

    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return 0.0
    return effective_sample_size(w) / w.size
