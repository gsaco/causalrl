"""Sensitivity analysis for contextual bandits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from crl.data.datasets import LoggedBanditDataset
from crl.policies.base import Policy


@dataclass
class SensitivityBounds:
    """Sensitivity bounds for bandit policy value."""

    gammas: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gammas": self.gammas,
            "lower": self.lower,
            "upper": self.upper,
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        return f"SensitivityBounds(num_points={self.gammas.size})"


def sensitivity_bounds(
    dataset: LoggedBanditDataset,
    policy: Policy,
    gammas: np.ndarray,
) -> SensitivityBounds:
    """Compute multiplicative propensity sensitivity bounds."""

    gammas = np.asarray(gammas, dtype=float)
    if np.any(gammas < 1.0):
        raise ValueError("gamma values must be >= 1.")
    if dataset.behavior_action_probs is None:
        raise ValueError("behavior_action_probs required for sensitivity bounds.")

    target_probs = policy.action_prob(dataset.contexts, dataset.actions)
    base_weights = target_probs / dataset.behavior_action_probs
    rewards = dataset.rewards

    lower = np.zeros_like(gammas)
    upper = np.zeros_like(gammas)

    for i, gamma in enumerate(gammas):
        low_adj = np.where(rewards >= 0, 1.0 / gamma, gamma)
        high_adj = np.where(rewards >= 0, gamma, 1.0 / gamma)
        lower[i] = float(np.mean(base_weights * low_adj * rewards))
        upper[i] = float(np.mean(base_weights * high_adj * rewards))

    return SensitivityBounds(
        gammas=gammas,
        lower=lower,
        upper=upper,
        metadata={"method": "bandit_sensitivity"},
    )
