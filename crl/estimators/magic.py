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
from crl.estimators.crossfit import make_folds
from crl.estimators.diagnostics import run_diagnostics
from crl.estimators.dr import LinearQModel
from crl.estimators.dr_core import discounted_powers
from crl.estimators.stats import mean_stderr
from crl.estimators.utils import compute_action_probs


@dataclass
class MAGICConfig:
    """Configuration for MAGIC."""

    num_folds: int = 1
    num_iterations: int = 5
    ridge: float = 1e-3
    mixing_horizons: tuple[int, ...] | None = None
    seed: int = 0


class MAGICEstimator(OPEEstimator):
    """MAGIC estimator that mixes truncated DR estimators."""

    required_assumptions = [
        "sequential_ignorability",
        "overlap",
        "markov",
        "behavior_policy_known",
    ]
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

        weights = np.array([])
        model_mse: list[float] = []

        if self.config.num_folds <= 1:
            q_model = self._fit_q_model(data)
            if q_model.train_mse is not None:
                model_mse.append(q_model.train_mse)
            per_traj_values, weights = self._magic_values(data, q_model)
        else:
            indices = np.arange(data.num_trajectories)
            folds = make_folds(
                data.num_trajectories, self.config.num_folds, self.config.seed
            )
            per_traj_values = np.zeros(data.num_trajectories, dtype=float)
            weight_list: list[np.ndarray] = []
            for fold_idx in folds:
                train_idx = np.setdiff1d(indices, fold_idx)
                q_model = self._fit_q_model(data, train_idx)
                if q_model.train_mse is not None:
                    model_mse.append(q_model.train_mse)
                values, fold_weights = self._magic_values(
                    data, q_model, indices=fold_idx
                )
                per_traj_values[fold_idx] = values
                weight_list.append(fold_weights)
            if weight_list:
                weights = np.mean(np.stack(weight_list, axis=0), axis=0)

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
            if model_mse:
                diagnostics["model"] = {"q_model_mse": float(np.mean(model_mse))}
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

    def _fit_q_model(
        self, data: TrajectoryDataset, indices: np.ndarray | None = None
    ) -> LinearQModel:
        if indices is None:
            obs = data.observations[data.mask]
            next_obs = data.next_observations[data.mask]
            actions = data.actions[data.mask]
            rewards = data.rewards[data.mask]
        else:
            obs = data.observations[indices][data.mask[indices]]
            next_obs = data.next_observations[indices][data.mask[indices]]
            actions = data.actions[indices][data.mask[indices]]
            rewards = data.rewards[indices][data.mask[indices]]
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
        self,
        data: TrajectoryDataset,
        q_model: LinearQModel,
        *,
        indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        obs = data.observations if indices is None else data.observations[indices]
        next_obs = (
            data.next_observations if indices is None else data.next_observations[indices]
        )
        actions = data.actions if indices is None else data.actions[indices]
        rewards = data.rewards if indices is None else data.rewards[indices]
        mask = data.mask if indices is None else data.mask[indices]

        behavior_probs = (
            data.behavior_action_probs
            if indices is None
            else data.behavior_action_probs[indices]
        )
        target_probs = compute_action_probs(self.estimand.policy, obs, actions)
        ratios = np.where(mask, target_probs / behavior_probs, 1.0)
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
