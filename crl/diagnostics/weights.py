"""Weight diagnostics for importance sampling."""

from __future__ import annotations

from typing import Any

import numpy as np

from crl.diagnostics.ess import effective_sample_size


def weight_tail_stats(
    weights: np.ndarray, quantile: float = 0.99, threshold: float = 10.0
) -> dict[str, Any]:
    """Compute weight tail statistics.

    Estimand:
        Not applicable.
    Assumptions:
        Weights are non-negative.
    Inputs:
        weights: Array of importance weights.
        quantile: Quantile level for tail summary.
        threshold: Threshold to count extreme weights.
    Outputs:
        Dictionary with tail metrics.
    Failure modes:
        None (returns zeros for empty input).
    """

    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "q95": 0.0,
            "q99": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
            "tail_fraction": 0.0,
        }

    mean = float(np.mean(w))
    std = float(np.std(w))
    q95 = float(np.quantile(w, 0.95))
    q99 = float(np.quantile(w, quantile))
    if std > 0:
        centered = (w - mean) / std
        skew = float(np.mean(centered**3))
        kurtosis = float(np.mean(centered**4) - 3.0)
    else:
        skew = 0.0
        kurtosis = 0.0

    return {
        "min": float(np.min(w)),
        "max": float(np.max(w)),
        "mean": mean,
        "std": std,
        "q95": q95,
        "q99": q99,
        "skew": skew,
        "kurtosis": kurtosis,
        "tail_fraction": float(np.mean(w > threshold)),
    }


def weight_time_diagnostics(
    weights: np.ndarray, mask: np.ndarray | None = None
) -> dict[str, list[float]]:
    """Summarize weight behavior over time (per timestep)."""

    w = np.asarray(weights, dtype=float)
    if w.ndim != 2:
        raise ValueError("weights must have shape (n, t) for time diagnostics.")
    t = w.shape[1]
    sums: list[float] = []
    variances: list[float] = []
    ess_vals: list[float] = []
    for step in range(t):
        if mask is not None:
            valid = mask[:, step]
            w_step = w[valid, step]
        else:
            w_step = w[:, step]
        sums.append(float(np.sum(w_step)))
        variances.append(float(np.var(w_step)) if w_step.size else 0.0)
        ess_vals.append(float(effective_sample_size(w_step)))
    return {"weight_sum": sums, "weight_variance": variances, "ess": ess_vals}
