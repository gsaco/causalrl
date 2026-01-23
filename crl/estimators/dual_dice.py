"""DualDICE estimator for behavior-agnostic OPE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from crl.data.datasets import TrajectoryDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.base import (
    DiagnosticsConfig,
    EstimatorReport,
    OPEEstimator,
    compute_ci,
)
from crl.estimators.stats import mean_stderr


@dataclass
class DualDICEConfig:
    """Configuration for DualDICE."""

    ridge: float = 1e-3
    normalize: bool = True


class DualDICEEstimator(OPEEstimator):
    """DualDICE estimator (Nachum et al., 2019) for discrete MDPs."""

    required_assumptions = ["sequential_ignorability", "markov"]
    required_fields = ["state_space_n"]
    diagnostics_keys: list[str] = ["density_ratio"]

    def __init__(
        self,
        estimand: PolicyValueEstimand,
        run_diagnostics: bool = True,
        diagnostics_config: DiagnosticsConfig | None = None,
        config: DualDICEConfig | None = None,
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
        self.config = config or DualDICEConfig()
        self._bootstrap_params.update(
            {
                "config": self.config,
                "bootstrap": False,
                "bootstrap_config": None,
            }
        )

    def estimate(self, data: TrajectoryDataset) -> EstimatorReport:
        self._validate_dataset(data)
        if data.state_space_n is None:
            raise ValueError("DualDICE requires discrete state_space_n.")

        obs = data.observations
        actions = data.actions
        rewards = data.rewards
        mask = data.mask
        num_states = data.state_space_n
        num_actions = data.action_space_n
        gamma = data.discount

        num_features = num_states * num_actions
        idx = obs.astype(int) * num_actions + actions.astype(int)
        idx = idx[mask]

        time_steps = np.tile(np.arange(data.horizon), (data.num_trajectories, 1))[mask]
        discounts = gamma**time_steps

        phi = np.zeros((idx.shape[0], num_features), dtype=float)
        phi[np.arange(idx.shape[0]), idx] = 1.0

        next_obs = data.next_observations[mask].astype(int)
        policy_probs_next = self.estimand.policy.action_probs(next_obs)
        phi_next = np.zeros((idx.shape[0], num_features), dtype=float)
        for a in range(num_actions):
            phi_next[np.arange(idx.shape[0]), next_obs * num_actions + a] = (
                policy_probs_next[:, a]
            )

        weights = discounts / max(np.mean(discounts), 1e-8)
        a_mat = (phi.T * weights) @ (phi - gamma * phi_next) / phi.shape[0]
        b_vec = self._initial_feature_mean(num_states, num_actions, obs[:, 0])
        b_vec = (1.0 - gamma) * b_vec

        theta = np.linalg.solve(a_mat + self.config.ridge * np.eye(num_features), b_vec)
        density_ratio = phi @ theta
        density_ratio = np.maximum(density_ratio, 0.0)

        rewards_flat = rewards[mask]
        values_flat = discounts * density_ratio * rewards_flat
        traj_values = self._aggregate_by_trajectory(values_flat, mask, data)

        if self.config.normalize:
            traj_values = traj_values / max(1.0 - gamma, 1e-6)

        value = float(np.mean(traj_values))
        stderr = mean_stderr(traj_values)

        diagnostics: dict[str, Any] = {
            "density_ratio": _ratio_stats(density_ratio)
        }
        warnings: list[str] = [
            "Uncertainty for density-ratio estimators may be unreliable; interpret CI cautiously."
        ]

        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={"estimator": "DualDICE", "config": self.config.__dict__},
            data=data,
        )

    def _initial_feature_mean(
        self, num_states: int, num_actions: int, initial_states: np.ndarray
    ) -> np.ndarray:
        phi0 = np.zeros(num_states * num_actions, dtype=float)
        policy_probs = self.estimand.policy.action_probs(initial_states)
        for s in range(num_states):
            mask = initial_states.astype(int) == s
            if not np.any(mask):
                continue
            probs = policy_probs[mask].mean(axis=0)
            for a in range(num_actions):
                phi0[s * num_actions + a] = probs[a]
        return phi0

    def _aggregate_by_trajectory(
        self, values_flat: np.ndarray, mask: np.ndarray, data: TrajectoryDataset
    ) -> np.ndarray:
        values = np.zeros((data.num_trajectories, data.horizon), dtype=float)
        values[mask] = values_flat
        return values.sum(axis=1)


def _ratio_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "q95": 0.0,
            "q99": 0.0,
        }
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
    }
