"""Double Reinforcement Learning estimator for discrete MDPs."""

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
from crl.estimators.crossfit import make_folds
from crl.estimators.diagnostics import run_diagnostics
from crl.estimators.dr import LinearQModel
from crl.estimators.dr_core import discounted_powers
from crl.estimators.stats import mean_stderr
from crl.estimators.utils import compute_action_probs
from crl.utils.cache import get_or_set


@dataclass
class DRLConfig:
    """Configuration for Double Reinforcement Learning (DRL)."""

    num_folds: int = 2
    num_iterations: int = 5
    ridge: float = 1e-3
    seed: int = 0
    min_prob: float = 1e-3
    clip_ratio: float | None = 10.0


class DRLEstimator(OPEEstimator):
    """Double Reinforcement Learning estimator for discrete MDPs.

    Estimand:
        PolicyValueEstimand for the target policy.
    Assumptions:
        Sequential ignorability, overlap, Markov property.
    Inputs:
        TrajectoryDataset with discrete state_space_n.
    Outputs:
        EstimatorReport with value and diagnostics.
    Failure modes:
        Requires adequate state-action coverage to estimate occupancy ratios.
    """

    required_assumptions = ["sequential_ignorability", "overlap", "markov"]
    required_fields = ["state_space_n"]
    diagnostics_keys = ["overlap", "ess", "weights", "max_weight", "model"]

    def __init__(
        self,
        estimand: PolicyValueEstimand,
        run_diagnostics: bool = True,
        diagnostics_config: DiagnosticsConfig | None = None,
        config: DRLConfig | None = None,
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
        self.config = config or DRLConfig()
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
            raise ValueError("DRL requires discrete state_space_n.")

        indices = np.arange(data.num_trajectories)
        folds = make_folds(
            data.num_trajectories, self.config.num_folds, self.config.seed
        )

        fold_values = np.zeros(data.num_trajectories, dtype=float)
        model_mse: list[float] = []

        for fold_idx in folds:
            train_idx = np.setdiff1d(indices, fold_idx)
            q_model = self._fit_q_model(data, train_idx)
            if q_model.train_mse is not None:
                model_mse.append(q_model.train_mse)
            ratio_table = self._estimate_time_ratios(data, train_idx)
            fold_values[fold_idx] = self._drl_values(
                data, fold_idx, q_model, ratio_table
            )

        value = float(np.mean(fold_values))
        stderr = mean_stderr(fold_values)

        diagnostics: dict[str, Any] = {}
        warnings: list[str] = []
        if self.run_diagnostics and data.behavior_action_probs is not None:
            target_probs = compute_action_probs(
                self.estimand.policy, data.observations, data.actions
            )
            ratios = np.where(data.mask, target_probs / data.behavior_action_probs, 1.0)
            weights = np.prod(ratios, axis=1)
            diagnostics, warnings = run_diagnostics(
                weights,
                target_probs,
                data.behavior_action_probs,
                data.mask,
                self.diagnostics_config,
            )
        if model_mse:
            diagnostics.setdefault("model", {})
            diagnostics["model"]["q_model_mse"] = float(np.mean(model_mse))

        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={"estimator": "DRL", "config": self.config.__dict__},
            data=data,
        )

    def _fit_q_model(
        self, data: TrajectoryDataset, indices: np.ndarray
    ) -> LinearQModel:
        cache_key = (
            "drl_q_model",
            self.config.num_iterations,
            self.config.ridge,
            indices.tobytes(),
        )

        def _build() -> LinearQModel:
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

        return get_or_set(data, cache_key, _build)

    def _estimate_time_ratios(
        self, data: TrajectoryDataset, indices: np.ndarray
    ) -> np.ndarray:
        """Estimate per-time state-action density ratios for target vs behavior."""

        obs = data.observations[indices]
        actions = data.actions[indices]
        mask = data.mask[indices]
        num_states = data.state_space_n or 0
        num_actions = data.action_space_n
        horizon = data.horizon

        # Behavior action probabilities per time step.
        behavior_sa = np.zeros((horizon, num_states, num_actions), dtype=float)
        behavior_s = np.zeros((horizon, num_states), dtype=float)
        for t in range(horizon):
            mask_t = mask[:, t]
            if not np.any(mask_t):
                continue
            states_t = obs[mask_t, t].astype(int)
            actions_t = actions[mask_t, t].astype(int)
            for s, a in zip(states_t, actions_t, strict=True):
                behavior_sa[t, s, a] += 1.0
                behavior_s[t, s] += 1.0
        denom = np.maximum(behavior_s[:, :, None], 1.0)
        behavior_prob = behavior_sa / denom
        behavior_prob = np.clip(behavior_prob, self.config.min_prob, 1.0)

        states = np.arange(num_states)
        pi_table = self.estimand.policy.action_probs(states)
        ratio_time = np.zeros_like(behavior_prob)
        for t in range(horizon):
            ratio_time[t] = pi_table / behavior_prob[t]
        if self.config.clip_ratio is not None:
            ratio_time = np.minimum(ratio_time, self.config.clip_ratio)
        return ratio_time

    def _drl_values(
        self,
        data: TrajectoryDataset,
        indices: np.ndarray,
        q_model: LinearQModel,
        ratio_time: np.ndarray,
    ) -> np.ndarray:
        obs = data.observations[indices]
        next_obs = data.next_observations[indices]
        actions = data.actions[indices]
        rewards = data.rewards[indices]
        mask = data.mask[indices]

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

        weights = np.zeros_like(rewards, dtype=float)
        for t in range(data.horizon):
            mask_t = mask[:, t]
            if not np.any(mask_t):
                continue
            states_t = obs[mask_t, t].astype(int)
            actions_t = actions[mask_t, t].astype(int)
            weights[mask_t, t] = ratio_time[t, states_t, actions_t]

        discounts = discounted_powers(data.discount, data.horizon)
        return v_matrix[:, 0] + np.sum(weights * td_matrix * discounts, axis=1)


__all__ = ["DRLConfig", "DRLEstimator"]
