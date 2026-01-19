"""High-confidence off-policy evaluation bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from crl.data.datasets import LoggedBanditDataset, TrajectoryDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.base import DiagnosticsConfig, EstimatorReport, OPEEstimator
from crl.estimators.diagnostics import run_diagnostics
from crl.estimators.utils import compute_action_probs, compute_trajectory_returns


@dataclass
class HighConfidenceConfig:
    """Configuration for high-confidence lower bounds."""

    delta: float = 0.05
    reward_bound: float | None = None


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
        config: HighConfidenceConfig | None = None,
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
            values, weights, target_probs, behavior_probs = self._bandit_values(data)
        else:
            values, weights, target_probs, behavior_probs = self._trajectory_values(
                data
            )

        bound = self.config.reward_bound
        warnings: list[str] = []
        if bound is None:
            bound = float(np.max(np.abs(values))) if values.size else 0.0
            warnings.append(
                "reward_bound was inferred from data; provide a theoretical bound for coverage guarantees."
            )

        lcb = empirical_bernstein_lower_bound(values, bound, self.config.delta)
        diagnostics: dict[str, Any] = {}
        if self.run_diagnostics:
            mask = data.mask if isinstance(data, TrajectoryDataset) else None
            diagnostics, diag_warnings = run_diagnostics(
                weights, target_probs, behavior_probs, mask, self.diagnostics_config
            )
            warnings.extend(diag_warnings)

        return self._build_report(
            value=lcb,
            stderr=None,
            ci=(lcb, float(np.mean(values))),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={
                "estimator": "HCOPE",
                "delta": self.config.delta,
                "reward_bound": bound,
            },
            data=data,
            lower_bound=lcb,
            upper_bound=float(np.mean(values)),
        )

    def _bandit_values(self, data: LoggedBanditDataset):
        if data.behavior_action_probs is None:
            raise ValueError("behavior_action_probs required for HCOPE on bandits.")
        target_probs = self.estimand.policy.action_prob(data.contexts, data.actions)
        weights = target_probs / data.behavior_action_probs
        values = weights * data.rewards
        return values, weights, target_probs, data.behavior_action_probs

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
        values = weights * returns
        return values, weights, target_probs, data.behavior_action_probs


def empirical_bernstein_lower_bound(
    values: np.ndarray, bound: float, delta: float
) -> float:
    """Empirical Bernstein lower bound for bounded random variables."""

    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(v.mean())
    mean = float(np.mean(v))
    var = float(np.var(v, ddof=1))
    log_term = np.log(2.0 / delta)
    term1 = np.sqrt(2.0 * var * log_term / n)
    term2 = 7.0 * bound * log_term / (3.0 * (n - 1))
    return mean - term1 - term2
