"""Sensitivity analysis for bandit propensities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from crl.data.datasets import LoggedBanditDataset
from crl.estimands.policy_value import PolicyValueEstimand


@dataclass
class SensitivityCurve:
    """Sensitivity curve output for robustness analysis.

    Estimand:
        Policy value under bounded propensity perturbations.
    Assumptions:
        Bounded confounding via multiplicative propensity shifts.
    Inputs:
        gammas: Array of gamma values.
        lower: Lower bound array.
        upper: Upper bound array.
        metadata: Optional metadata.
    Outputs:
        Sensitivity curve with lower/upper bounds.
    Failure modes:
        None.
    """

    gammas: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return {
            "gammas": self.gammas,
            "lower": self.lower,
            "upper": self.upper,
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        return f"SensitivityCurve(num_points={self.gammas.size})"


class BanditPropensitySensitivity:
    """Bandit sensitivity analysis using multiplicative propensity bounds.

    Estimand:
        Policy value under bounded confounding.
    Assumptions:
        Bounded confounding and overlap in logged propensities.
    Inputs:
        estimand: PolicyValueEstimand for the target policy.
    Outputs:
        SensitivityCurve with lower and upper bounds.
    Failure modes:
        Bounds are heuristic and may be conservative or loose.
    """

    required_assumptions = ["bounded_confounding", "overlap"]

    def __init__(self, estimand: PolicyValueEstimand) -> None:
        self.estimand = estimand
        self.estimand.require(self.required_assumptions)

    def curve(self, data: LoggedBanditDataset, gammas: np.ndarray) -> SensitivityCurve:
        """Compute a robustness curve over gamma values.

        Inputs:
            data: LoggedBanditDataset with propensities.
            gammas: Array of gamma values >= 1.
        Outputs:
            SensitivityCurve with lower and upper bounds per gamma.
        Failure modes:
            Raises ValueError for invalid gamma values.
        """

        gammas = np.asarray(gammas, dtype=float)
        if np.any(gammas < 1.0):
            raise ValueError("gamma values must be >= 1.")

        target_probs = self.estimand.policy.action_prob(data.contexts, data.actions)
        base_weights = target_probs / data.behavior_action_probs
        rewards = data.rewards

        lower = np.zeros_like(gammas)
        upper = np.zeros_like(gammas)

        for i, gamma in enumerate(gammas):
            low_adj = np.where(rewards >= 0, 1.0 / gamma, gamma)
            high_adj = np.where(rewards >= 0, gamma, 1.0 / gamma)
            lower[i] = float(np.mean(base_weights * low_adj * rewards))
            upper[i] = float(np.mean(base_weights * high_adj * rewards))

        return SensitivityCurve(
            gammas=gammas,
            lower=lower,
            upper=upper,
            metadata={"method": "bandit_propensity_sensitivity"},
        )
