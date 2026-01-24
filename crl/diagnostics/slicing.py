"""Slicing diagnostics for overlap and weights."""

from __future__ import annotations

from typing import Any

import numpy as np


def action_overlap_slices(
    actions: np.ndarray,
    behavior_action_probs: np.ndarray,
    target_action_probs: np.ndarray,
    *,
    action_space_n: int,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Summarize overlap metrics per action."""

    actions = np.asarray(actions).reshape(-1)
    behavior = np.asarray(behavior_action_probs).reshape(-1)
    target = np.asarray(target_action_probs).reshape(-1)

    slices: list[dict[str, Any]] = []
    for action in range(action_space_n):
        mask = actions == action
        if not np.any(mask):
            continue
        slices.append(
            {
                "action": int(action),
                "count": int(np.sum(mask)),
                "behavior_prob_min": float(np.min(behavior[mask])),
                "behavior_prob_mean": float(np.mean(behavior[mask])),
                "ratio_mean": float(np.mean(target[mask] / np.clip(behavior[mask], 1e-12, None))),
            }
        )
    slices.sort(key=lambda row: row["behavior_prob_min"])
    return slices[:top_k]


def timestep_weight_slices(
    ratios: np.ndarray,
    mask: np.ndarray,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Summarize importance ratios by timestep."""

    ratios = np.asarray(ratios)
    mask = np.asarray(mask, dtype=bool)
    horizon = ratios.shape[1]
    slices: list[dict[str, Any]] = []
    for t in range(horizon):
        valid = mask[:, t]
        if not np.any(valid):
            continue
        vals = ratios[:, t][valid]
        slices.append(
            {
                "timestep": int(t),
                "count": int(np.sum(valid)),
                "ratio_mean": float(np.mean(vals)),
                "ratio_max": float(np.max(vals)),
            }
        )
    slices.sort(key=lambda row: row["ratio_mean"], reverse=True)
    return slices[:top_k]


__all__ = ["action_overlap_slices", "timestep_weight_slices"]
