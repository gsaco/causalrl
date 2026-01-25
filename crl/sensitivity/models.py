"""Sensitivity model objects for bounded confounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from crl.data.datasets import LoggedBanditDataset, TrajectoryDataset
from crl.policies.base import Policy
from crl.sensitivity.bandits import SensitivityBounds, sensitivity_bounds
from crl.sensitivity.namkoong2020 import confounded_ope_bounds


@dataclass(frozen=True)
class GammaSensitivityModel:
    """Gamma sensitivity model for bandits or sequential OPE."""

    gamma: float
    method: Literal["bandit", "sequential"] = "sequential"

    def bounds(
        self, dataset: LoggedBanditDataset | TrajectoryDataset, policy: Policy
    ) -> SensitivityBounds:
        gammas = np.asarray([self.gamma], dtype=float)
        if isinstance(dataset, LoggedBanditDataset) or self.method == "bandit":
            return sensitivity_bounds(dataset, policy, gammas)
        return confounded_ope_bounds(dataset, policy, gammas)


__all__ = ["GammaSensitivityModel"]
