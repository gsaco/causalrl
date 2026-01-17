"""Weight diagnostics for importance sampling."""

from __future__ import annotations

from typing import Any

import numpy as np


def weight_tail_stats(weights: np.ndarray, quantile: float = 0.99, threshold: float = 10.0) -> dict[str, Any]:
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
        return {"max": 0.0, "q99": 0.0, "tail_fraction": 0.0}
    return {
        "max": float(np.max(w)),
        "q99": float(np.quantile(w, quantile)),
        "tail_fraction": float(np.mean(w > threshold)),
    }
