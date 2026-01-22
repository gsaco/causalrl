"""MAGIC estimator (Thomas & Brunskill, 2016)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from crl.data.datasets import TrajectoryDataset
from crl.diagnostics.weights import weight_time_diagnostics
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.base import (
    DiagnosticsConfig,
    EstimatorReport,
    OPEEstimator,
    compute_ci,
)
from crl.estimators.diagnostics import run_diagnostics
from crl.estimators.dr import LinearQModel
from crl.estimators.dr_core import discounted_powers
from crl.estimators.stats import mean_stderr
from crl.estimators.utils import compute_action_probs


@dataclass
class MAGICConfig:
    """Configuration for MAGIC."""

    num_iterations: int = 5
    ridge: float = 1e-3
    mixing_horizons: tuple[int, ...] | None = None


class MAGICEstimator(OPEEstimator):
    """MAGIC estimator that mixes truncated DR estimators."""

    required_assumptions = ["sequential_ignorability", "overlap", "markov"]
    required_fields = ["behavior_action_probs"]
    diagnostics_keys = [
        "overlap",
        "ess",
        "weights",
        "max_weight",
        "model",
        "weight_time",
    ]

    def __init__(
        self,
        estimand: PolicyValueEstimand,
        run_diagnostics: bool = True,
        diagnostics_config: DiagnosticsConfig | None = None,
        config: MAGICConfig | None = None,
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
        self.config = config or MAGICConfig()
        self._bootstrap_params.update(
            {
                "config": self.config,
                "bootstrap": False,
                "bootstrap_config": None,
            }
        )

    def estimate(self, data: TrajectoryDataset) -> EstimatorReport:
        self._validate_dataset(data)
        if data.behavior_action_probs is None:
            raise ValueError("behavior_action_probs are required for MAGIC.")

        q_model = self._fit_q_model(data)
        per_traj_values, weights = self._magic_values(data, q_model)

        value = float(np.mean(per_traj_values))
        stderr = mean_stderr(per_traj_values)

        diagnostics: dict[str, Any] = {}
        warnings: list[str] = []
        if self.run_diagnostics:
            target_probs = compute_action_probs(
                self.estimand.policy, data.observations, data.actions
            )
            ratios = np.where(data.mask, target_probs / data.behavior_action_probs, 1.0)
            weights_diag = np.prod(ratios, axis=1)
            diagnostics, warnings = run_diagnostics(
                weights_diag,
                target_probs,
                data.behavior_action_probs,
                data.mask,
                self.diagnostics_config,
            )
            diagnostics["model"] = {"q_model_mse": q_model.train_mse}
            diagnostics["weight_time"] = weight_time_diagnostics(
                np.cumprod(ratios, axis=1), data.mask
            )

        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={"estimator": "MAGIC", "weights": weights.tolist()},
            data=data,
        )

    def _fit_q_model(self, data: TrajectoryDataset) -> LinearQModel:
        obs = data.observations[data.mask]
        next_obs = data.next_observations[data.mask]
        actions = data.actions[data.mask]
        rewards = data.rewards[data.mask]
        policy_probs_next = self.estimand.policy.action_probs(next_obs)

        q_model = LinearQModel(
            action_space_n=data.action_space_n,
            state_space_n=data.state_space_n,
            ridge=self.config.ridge,
        )
        q_model.fit(
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            policy_probs_next=policy_probs_next,
            discount=data.discount,
            num_iterations=self.config.num_iterations,
        )
        return q_model

    def _magic_values(
        self, data: TrajectoryDataset, q_model: LinearQModel
    ) -> tuple[np.ndarray, np.ndarray]:
        obs = data.observations
        next_obs = data.next_observations
        actions = data.actions
        rewards = data.rewards
        mask = data.mask

        target_probs = compute_action_probs(self.estimand.policy, obs, actions)
        ratios = np.where(mask, target_probs / data.behavior_action_probs, 1.0)
        cumulative = np.cumprod(ratios, axis=1)

        obs_flat = obs[mask]
        next_obs_flat = next_obs[mask]
        actions_flat = actions[mask]
        policy_probs_flat = self.estimand.policy.action_probs(obs_flat)
        policy_probs_next = self.estimand.policy.action_probs(next_obs_flat)

        q_hat = q_model.predict_q(obs_flat, actions_flat)
        v_hat = q_model.predict_v(obs_flat, policy_probs_flat)
        v_hat_next = q_model.predict_v(next_obs_flat, policy_probs_next)

        td_residual = rewards[mask] + data.discount * v_hat_next - q_hat

        td_matrix = np.zeros_like(rewards, dtype=float)
        v_matrix = np.zeros_like(rewards, dtype=float)
        td_matrix[mask] = td_residual
        v_matrix[mask] = v_hat

        horizons = self.config.mixing_horizons
        if horizons is None:
            horizons = tuple(range(0, data.horizon + 1))

        candidate_values = []
        discounts = discounted_powers(data.discount, data.horizon)
        for m in horizons:
            if m == 0:
                values = v_matrix[:, 0]
            else:
                values = v_matrix[:, 0] + np.sum(
                    cumulative[:, :m] * td_matrix[:, :m] * discounts[:m], axis=1
                )
            candidate_values.append(values)

        candidate_values_arr = np.vstack(candidate_values)
        variances = np.var(candidate_values_arr, axis=1, ddof=1)
        weights = np.where(variances > 0, 1.0 / variances, 0.0)
        if weights.sum() == 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        mixed_values = np.average(candidate_values_arr, axis=0, weights=weights)
        return mixed_values, weights
