"""High-confidence off-policy evaluation bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from crl.data.datasets import LoggedBanditDataset, TrajectoryDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.base import (
    DiagnosticsConfig,
    EstimatorReport,
    OPEEstimator,
    UncertaintySummary,
)
from crl.estimators.diagnostics import run_diagnostics
from crl.estimators.hc_bounds import select_hcope_bound
from crl.estimators.utils import compute_action_probs, compute_trajectory_returns


@dataclass
class HighConfidenceISConfig:
    """Configuration for high-confidence lower bounds."""

    delta: float = 0.05
    reward_bound: float | None = None
    clip_grid: list[float] | None = None
    bound: Literal["empirical_bernstein", "hoeffding"] = "empirical_bernstein"


HighConfidenceConfig = HighConfidenceISConfig


class HighConfidenceISEstimator(OPEEstimator):
    """High-confidence lower bound based on IS (Thomas et al., 2015)."""

    required_assumptions = ["sequential_ignorability", "overlap", "bounded_rewards"]
    required_fields = ["behavior_action_probs"]
    diagnostics_keys = ["overlap", "ess", "weights", "max_weight", "model"]

    def __init__(
        self,
        estimand: PolicyValueEstimand,
        run_diagnostics: bool = True,
        diagnostics_config: DiagnosticsConfig | None = None,
        config: HighConfidenceISConfig | None = None,
        bootstrap: bool = False,
        bootstrap_config: Any | None = None,
    ) -> None:
        super().__init__(
            estimand,
            run_diagnostics,
            diagnostics_config,
            bootstrap,
            bootstrap_config,
        )
        self.config = config or HighConfidenceConfig()
        self._bootstrap_params.update(
            {
                "config": self.config,
                "bootstrap": False,
                "bootstrap_config": None,
            }
        )

    def estimate(
        self, data: LoggedBanditDataset | TrajectoryDataset
    ) -> EstimatorReport:
        self._validate_dataset(data)
        if isinstance(data, LoggedBanditDataset):
            returns, weights, target_probs, behavior_probs = self._bandit_values(data)
        else:
            returns, weights, target_probs, behavior_probs = self._trajectory_values(
                data
            )

        bound = self.config.reward_bound
        warnings: list[str] = []
        if bound is None:
            bound = float(np.max(np.abs(returns))) if returns.size else 0.0
            warnings.append(
                "reward_bound was inferred from data; provide a theoretical bound for coverage guarantees."
            )

        clip_grid = self.config.clip_grid
        if clip_grid is None:
            if weights.size == 0:
                clip_grid = [1.0]
            else:
                quantiles = np.quantile(weights, [0.5, 0.75, 0.9, 0.95, 0.99])
                clip_grid = sorted({1.0, float(np.max(weights)), *quantiles.tolist()})

        best, all_results = select_hcope_bound(
            returns=returns,
            weights=weights,
            reward_bound=bound,
            delta=self.config.delta,
            clip_grid=list(clip_grid),
            bound_kind=self.config.bound,
        )
        lcb = best.lower_bound
        is_estimate = float(np.mean(weights * returns)) if weights.size else 0.0
        diagnostics: dict[str, Any] = {}
        if self.run_diagnostics:
            mask = data.mask if isinstance(data, TrajectoryDataset) else None
            diagnostics, diag_warnings = run_diagnostics(
                weights, target_probs, behavior_probs, mask, self.diagnostics_config
            )
            warnings.extend(diag_warnings)
        uncertainty = UncertaintySummary(
            kind=self.config.bound,
            level=1.0 - self.config.delta,
            interval=None,
            lower_bound=lcb,
            upper_bound=is_estimate,
            notes=[
                f"clip={best.clip}",
                f"bias_term={best.bias_term:.4f}",
            ],
        )

        return self._build_report(
            value=lcb,
            stderr=None,
            ci=(lcb, is_estimate),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={
                "estimator": "HCOPE",
                "delta": self.config.delta,
                "reward_bound": bound,
                "bound": self.config.bound,
                "clip_grid": list(clip_grid),
                "clip": best.clip,
                "bias_term": best.bias_term,
                "clipped_mean": best.clipped_mean,
                "grid_results": [
                    {
                        "clip": result.clip,
                        "lower_bound": result.lower_bound,
                        "bias_term": result.bias_term,
                        "clipped_mean": result.clipped_mean,
                    }
                    for result in all_results
                ],
            },
            data=data,
            lower_bound=lcb,
            upper_bound=is_estimate,
            uncertainty=uncertainty,
        )

    def _bandit_values(self, data: LoggedBanditDataset):
        if data.behavior_action_probs is None:
            raise ValueError("behavior_action_probs required for HCOPE on bandits.")
        target_probs = self.estimand.policy.action_prob(data.contexts, data.actions)
        weights = target_probs / data.behavior_action_probs
        returns = data.rewards
        return returns, weights, target_probs, data.behavior_action_probs

    def _trajectory_values(self, data: TrajectoryDataset):
        if data.behavior_action_probs is None:
            raise ValueError(
                "behavior_action_probs required for HCOPE on trajectories."
            )
        target_probs = compute_action_probs(
            self.estimand.policy, data.observations, data.actions
        )
        ratios = np.where(data.mask, target_probs / data.behavior_action_probs, 1.0)
        weights = np.prod(ratios, axis=1)
        returns = compute_trajectory_returns(data.rewards, data.mask, data.discount)
        return returns, weights, target_probs, data.behavior_action_probs


__all__ = [
    "HighConfidenceISConfig",
    "HighConfidenceConfig",
    "HighConfidenceISEstimator",
]
