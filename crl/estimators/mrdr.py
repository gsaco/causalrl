"""Minimally randomized doubly robust estimator."""

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
from crl.estimators.dr_core import dr_values_from_qv
from crl.estimators.stats import mean_stderr
from crl.estimators.utils import compute_action_probs
from crl.utils.cache import get_or_set


@dataclass
class MRDRConfig:
    """Configuration for MRDR."""

    num_folds: int = 2
    num_iterations: int = 5
    ridge: float = 1e-3
    seed: int = 0


class WeightedLinearQModel(LinearQModel):
    """Linear Q-model with weighted ridge regression."""

    def fit(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        policy_probs_next: np.ndarray,
        discount: float,
        num_iterations: int,
        sample_weights: np.ndarray | None = None,
    ) -> None:
        obs_features = self._features(observations)
        next_features = self._features(next_observations)
        feature_dim = obs_features.shape[1]
        self.weights = np.zeros((self.action_space_n, feature_dim), dtype=float)
        if sample_weights is None:
            sample_weights = np.ones_like(rewards, dtype=float)

        for _ in range(num_iterations):
            next_q = self._predict_all(next_features)
            next_v = np.sum(policy_probs_next * next_q, axis=1)
            targets = rewards + discount * next_v

            for action in range(self.action_space_n):
                mask = actions == action
                if not np.any(mask):
                    continue
                x = obs_features[mask]
                y = targets[mask]
                w = sample_weights[mask].reshape(-1, 1)
                xtw = x.T * w.T
                xtx = xtw @ x + self.ridge * np.eye(feature_dim)
                xty = xtw @ y
                self.weights[action] = np.linalg.solve(xtx, xty)

        preds = self.predict_q(observations, actions)
        self.train_mse = float(np.mean((preds - targets) ** 2))


class MRDREstimator(OPEEstimator):
    """MRDR estimator (Farajtabar et al., 2018)."""

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
        config: MRDRConfig | None = None,
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
        self.config = config or MRDRConfig()
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
            raise ValueError("behavior_action_probs are required for MRDR.")
        behavior_action_probs = data.behavior_action_probs

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
            fold_values[fold_idx] = self._dr_values(data, fold_idx, q_model)

        value = float(np.mean(fold_values))
        stderr = mean_stderr(fold_values)

        diagnostics: dict[str, Any] = {}
        warnings: list[str] = []
        if self.run_diagnostics:
            target_probs = compute_action_probs(
                self.estimand.policy, data.observations, data.actions
            )
            ratios = np.where(data.mask, target_probs / behavior_action_probs, 1.0)
            weights = np.prod(ratios, axis=1)
            diagnostics, warnings = run_diagnostics(
                weights,
                target_probs,
                behavior_action_probs,
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
            metadata={"estimator": "MRDR", "config": self.config.__dict__},
            data=data,
        )

    def _fit_q_model(
        self, data: TrajectoryDataset, indices: np.ndarray
    ) -> WeightedLinearQModel:
        cache_key = (
            "mrdr_q_model",
            self.config.num_iterations,
            self.config.ridge,
            indices.tobytes(),
        )

        def _build() -> WeightedLinearQModel:
            obs = data.observations[indices][data.mask[indices]]
            next_obs = data.next_observations[indices][data.mask[indices]]
            actions = data.actions[indices][data.mask[indices]]
            rewards = data.rewards[indices][data.mask[indices]]
            policy_probs_next = self.estimand.policy.action_probs(next_obs)
            target_probs = self.estimand.policy.action_prob(obs, actions)
            behavior_action_probs = data.behavior_action_probs
            assert behavior_action_probs is not None
            sample_weights = (
                target_probs / behavior_action_probs[indices][data.mask[indices]]
            )

            q_model = WeightedLinearQModel(
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
                sample_weights=sample_weights,
            )
            return q_model

        return get_or_set(data, cache_key, _build)

    def _dr_values(
        self, data: TrajectoryDataset, indices: np.ndarray, q_model: LinearQModel
    ) -> np.ndarray:
        obs = data.observations[indices]
        next_obs = data.next_observations[indices]
        actions = data.actions[indices]
        rewards = data.rewards[indices]
        mask = data.mask[indices]
        behavior_action_probs = data.behavior_action_probs
        assert behavior_action_probs is not None

        target_probs = compute_action_probs(self.estimand.policy, obs, actions)
        ratios = np.where(mask, target_probs / behavior_action_probs[indices], 1.0)
        cumulative = np.cumprod(ratios, axis=1)

        obs_flat = obs[mask]
        next_obs_flat = next_obs[mask]
        actions_flat = actions[mask]
        policy_probs_flat = self.estimand.policy.action_probs(obs_flat)
        policy_probs_next = self.estimand.policy.action_probs(next_obs_flat)

        q_hat = q_model.predict_q(obs_flat, actions_flat)
        v_hat = q_model.predict_v(obs_flat, policy_probs_flat)
        v_hat_next = q_model.predict_v(next_obs_flat, policy_probs_next)

        q_matrix = np.zeros_like(rewards, dtype=float)
        v_matrix = np.zeros((rewards.shape[0], rewards.shape[1] + 1), dtype=float)

        q_matrix[mask] = q_hat
        v_matrix[:, :-1][mask] = v_hat
        v_matrix[:, 1:][mask] = v_hat_next

        return dr_values_from_qv(
            rewards=rewards,
            mask=mask,
            discount=data.discount,
            cumulative_rho=cumulative,
            v_hat=v_matrix,
            q_hat=q_matrix,
        )
