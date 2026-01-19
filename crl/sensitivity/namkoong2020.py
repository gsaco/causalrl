"""Sensitivity bounds for sequential OPE under bounded confounding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crl.data.trajectory import TrajectoryDataset
from crl.estimators.utils import compute_action_probs, compute_trajectory_returns
from crl.policies.base import Policy
from crl.sensitivity.bandits import SensitivityBounds


@dataclass
class GammaSensitivityModel:
    """Gamma sensitivity model for confounded sequential OPE."""

    gamma: float

    def adjustments(self, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute multiplicative adjustments for lower/upper bounds."""

        low_adj = np.where(returns >= 0, 1.0 / self.gamma, self.gamma)
        high_adj = np.where(returns >= 0, self.gamma, 1.0 / self.gamma)
        return low_adj, high_adj


def confounded_ope_bounds(
    dataset: TrajectoryDataset,
    policy: Policy,
    gammas: np.ndarray,
) -> SensitivityBounds:
    """Compute sensitivity bounds over gamma values for trajectories."""

    gammas = np.asarray(gammas, dtype=float)
    if np.any(gammas < 1.0):
        raise ValueError("gamma values must be >= 1.")
    if dataset.behavior_action_probs is None:
        raise ValueError("behavior_action_probs required for sensitivity bounds.")

    target_probs = compute_action_probs(policy, dataset.observations, dataset.actions)
    ratios = np.where(dataset.mask, target_probs / dataset.behavior_action_probs, 1.0)
    weights = np.prod(ratios, axis=1)
    returns = compute_trajectory_returns(
        dataset.rewards, dataset.mask, dataset.discount
    )

    lower = np.zeros_like(gammas)
    upper = np.zeros_like(gammas)
    for i, gamma in enumerate(gammas):
        model = GammaSensitivityModel(gamma)
        low_adj, high_adj = model.adjustments(returns)
        lower[i] = float(np.mean(weights * low_adj * returns))
        upper[i] = float(np.mean(weights * high_adj * returns))

    return SensitivityBounds(
        gammas=gammas,
        lower=lower,
        upper=upper,
        metadata={"method": "namkoong2020_sequential"},
    )


__all__ = ["GammaSensitivityModel", "confounded_ope_bounds"]
