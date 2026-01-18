"""Diagnostics for behavior policy estimation."""

from __future__ import annotations

from typing import Any

import numpy as np


def calibration_curve(
    probs: np.ndarray, actions: np.ndarray, num_bins: int = 10
) -> dict[str, Any]:
    """Compute calibration curve for predicted action confidences."""

    probs = np.asarray(probs, dtype=float)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = predictions == actions

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.digitize(confidences, bin_edges, right=True) - 1

    acc: list[float] = []
    conf: list[float] = []
    counts: list[int] = []
    ece = 0.0
    for b in range(num_bins):
        mask = bin_ids == b
        if not np.any(mask):
            acc.append(0.0)
            conf.append(0.0)
            counts.append(0)
            continue
        acc_b = float(np.mean(correct[mask]))
        conf_b = float(np.mean(confidences[mask]))
        count = int(np.sum(mask))
        acc.append(acc_b)
        conf.append(conf_b)
        counts.append(count)
        ece += abs(acc_b - conf_b) * (count / actions.size)

    return {
        "bin_edges": bin_edges.tolist(),
        "accuracy": acc,
        "confidence": conf,
        "counts": counts,
        "ece": float(ece),
    }


def propensity_stats(
    propensities: np.ndarray, clip_min: float = 1e-3
) -> dict[str, Any]:
    """Summarize propensity distribution and clipping rates."""

    prop = np.asarray(propensities, dtype=float)
    prop = prop[np.isfinite(prop)]
    if prop.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "clip_min": clip_min,
            "fraction_below_clip": 0.0,
        }
    return {
        "min": float(np.min(prop)),
        "max": float(np.max(prop)),
        "mean": float(np.mean(prop)),
        "std": float(np.std(prop)),
        "clip_min": float(clip_min),
        "fraction_below_clip": float(np.mean(prop < clip_min)),
    }


def support_stats(propensities: np.ndarray, min_prob: float = 1e-3) -> dict[str, Any]:
    """Compute support diagnostics for estimated propensities."""

    prop = np.asarray(propensities, dtype=float)
    return {
        "min_prob": float(min_prob),
        "fraction_below_min": float(np.mean(prop < min_prob)),
    }


def behavior_diagnostics(
    action_probs: np.ndarray,
    actions: np.ndarray,
    propensities: np.ndarray,
    *,
    clip_min: float = 1e-3,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Aggregate diagnostics for behavior policy estimation."""

    return {
        "calibration": calibration_curve(action_probs, actions, num_bins=num_bins),
        "propensity": propensity_stats(propensities, clip_min=clip_min),
        "support": support_stats(propensities, min_prob=clip_min),
    }
