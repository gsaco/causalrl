"""Sensitivity estimands for policy value under bounded confounding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crl.assumptions import AssumptionSet
from crl.core.policy import Policy
from crl.data.datasets import LoggedBanditDataset, TrajectoryDataset
from crl.sensitivity.bandits import SensitivityBounds, sensitivity_bounds
from crl.sensitivity.namkoong2020 import confounded_ope_bounds


@dataclass(frozen=True)
class SensitivityPolicyValueEstimand:
    """Policy value estimand under a gamma-bounded confounding model."""

    policy: Policy
    discount: float
    horizon: int | None
    gammas: np.ndarray
    assumptions: AssumptionSet

    def require(self, names: list[str]) -> None:
        """Require that assumptions include the specified names."""

        self.assumptions.require(names)

    def compute_bounds(
        self, dataset: LoggedBanditDataset | TrajectoryDataset
    ) -> SensitivityBounds:
        """Compute sensitivity bounds for the provided dataset."""

        gammas = np.asarray(self.gammas, dtype=float)
        if isinstance(dataset, LoggedBanditDataset):
            return sensitivity_bounds(dataset, self.policy, gammas)
        return confounded_ope_bounds(dataset, self.policy, gammas)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": type(self.policy).__name__,
            "discount": self.discount,
            "horizon": self.horizon,
            "gammas": self.gammas.tolist(),
            "assumptions": self.assumptions.names(),
        }

    def __repr__(self) -> str:
        return (
            "SensitivityPolicyValueEstimand(policy="
            f"{type(self.policy).__name__}, discount={self.discount}, horizon={self.horizon})"
        )


__all__ = ["SensitivityPolicyValueEstimand"]
