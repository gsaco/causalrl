"""Overlap diagnostics for logged data."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_overlap_metrics(
    target_action_probs: np.ndarray,
    behavior_action_probs: np.ndarray,
    mask: np.ndarray | None = None,
    threshold: float = 1e-3,
) -> dict[str, Any]:
    """Compute overlap diagnostics between target and behavior policies.

    Estimand:
        Not applicable.
    Assumptions:
        Logged propensities are accurate and non-zero for observed actions.
    Inputs:
        target_action_probs: Array of probabilities for observed actions.
        behavior_action_probs: Array of behavior propensities.
        mask: Optional boolean mask for valid steps.
        threshold: Minimum acceptable behavior probability.
    Outputs:
        Dictionary of overlap metrics.
    Failure modes:
        If arrays contain zeros, ratios may be infinite.
    """

    target = np.asarray(target_action_probs, dtype=float)
    behavior = np.asarray(behavior_action_probs, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        target = target[mask]
        behavior = behavior[mask]

    ratio = np.divide(
        target, behavior, out=np.full_like(target, np.inf), where=behavior > 0
    )
    metrics = {
        "min_behavior_prob": float(np.min(behavior)) if behavior.size else 0.0,
        "min_target_prob": float(np.min(target)) if target.size else 0.0,
        "fraction_behavior_below_threshold": float(np.mean(behavior < threshold))
        if behavior.size
        else 0.0,
        "fraction_target_below_threshold": float(np.mean(target < threshold))
        if target.size
        else 0.0,
        "ratio_min": float(np.min(ratio)) if ratio.size else 0.0,
        "ratio_max": float(np.max(ratio)) if ratio.size else 0.0,
        "ratio_q50": float(np.quantile(ratio, 0.5)) if ratio.size else 0.0,
        "ratio_q90": float(np.quantile(ratio, 0.9)) if ratio.size else 0.0,
        "ratio_q99": float(np.quantile(ratio, 0.99)) if ratio.size else 0.0,
        "support_violations": int(np.sum((target > 0) & (behavior <= threshold))),
    }
    return metrics
