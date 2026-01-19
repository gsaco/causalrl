"""Doubly robust estimator with cross-fitting."""

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
from crl.estimators.stats import mean_stderr
from crl.estimators.utils import compute_action_probs


@dataclass
class DRCrossFitConfig:
    """Configuration for cross-fitting.

    Estimand:
        Not applicable.
    Assumptions:
        None.
    Inputs:
        num_folds: Number of cross-fitting folds.
        num_iterations: Bellman iteration count for linear Q.
        ridge: Ridge regularization strength.
        seed: RNG seed for fold splitting.
    Outputs:
        Configuration object.
    Failure modes:
        None.
    """

    num_folds: int = 2
    num_iterations: int = 5
    ridge: float = 1e-3
    seed: int = 0


class LinearQModel:
    """Linear Q-function model with per-action ridge regression."""

    def __init__(
        self, action_space_n: int, state_space_n: int | None, ridge: float
    ) -> None:
        self.action_space_n = action_space_n
        self.state_space_n = state_space_n
        self.ridge = ridge
        self.weights: np.ndarray | None = None
        self.train_mse: float | None = None

    def _features(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations)
        if obs.ndim == 1:
            if self.state_space_n is not None:
                return self._one_hot(obs.astype(int), self.state_space_n)
            return obs.reshape(-1, 1).astype(float)
        if obs.ndim == 2:
            if obs.shape[1] == 1 and self.state_space_n is not None:
                return self._one_hot(obs.reshape(-1).astype(int), self.state_space_n)
            return obs.astype(float)
        raise ValueError("Observations must be 1D or 2D after flattening.")

    @staticmethod
    def _one_hot(values: np.ndarray, num_classes: int) -> np.ndarray:
        values = values.astype(int)
        one_hot = np.zeros((values.shape[0], num_classes), dtype=float)
        one_hot[np.arange(values.shape[0]), values] = 1.0
        return one_hot

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
                if sample_weights is None:
                    xtx = x.T @ x + self.ridge * np.eye(feature_dim)
                    xty = x.T @ y
                else:
                    w = sample_weights[mask].reshape(-1, 1)
                    xtw = x.T * w.T
                    xtx = xtw @ x + self.ridge * np.eye(feature_dim)
                    xty = xtw @ y
                self.weights[action] = np.linalg.solve(xtx, xty)

        preds = self.predict_q(observations, actions)
        self.train_mse = float(np.mean((preds - targets) ** 2))

    def _predict_all(self, obs_features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Q model is not fit.")
        return obs_features @ self.weights.T

    def predict_q(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        obs_features = self._features(observations)
        q_all = self._predict_all(obs_features)
        return q_all[np.arange(obs_features.shape[0]), actions]

    def predict_v(
        self, observations: np.ndarray, policy_probs: np.ndarray
    ) -> np.ndarray:
        obs_features = self._features(observations)
        q_all = self._predict_all(obs_features)
        return np.sum(policy_probs * q_all, axis=1)


class DoublyRobustEstimator(OPEEstimator):
    """Doubly robust estimator with cross-fitting.

    Estimand:
        PolicyValueEstimand for the target policy.
    Assumptions:
        Sequential ignorability, overlap, and correct model specification.
    Inputs:
        TrajectoryDataset (n, t).
    Outputs:
        EstimatorReport with value and diagnostics.
    Failure modes:
        Bias if both the Q model and propensities are misspecified.
    """

    required_assumptions = ["sequential_ignorability", "overlap", "markov"]
    required_fields = ["behavior_action_probs"]
    diagnostics_keys = ["overlap", "ess", "weights", "max_weight", "model"]

    def __init__(
        self,
        estimand: PolicyValueEstimand,
        run_diagnostics: bool = True,
        diagnostics_config: DiagnosticsConfig | None = None,
        config: DRCrossFitConfig | None = None,
    ) -> None:
        super().__init__(estimand, run_diagnostics, diagnostics_config)
        self.config = config or DRCrossFitConfig()

    def estimate(self, data: TrajectoryDataset) -> EstimatorReport:
        """Estimate policy value via cross-fitted DR."""

        self._validate_dataset(data)
        if data.behavior_action_probs is None:
            raise ValueError("behavior_action_probs are required for DR.")
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

        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={"estimator": "DR", "config": self.config.__dict__},
            data=data,
        )

    def _fit_q_model(
        self, data: TrajectoryDataset, indices: np.ndarray
    ) -> LinearQModel:
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

        td_residual = rewards[mask] + data.discount * v_hat_next - q_hat

        td_matrix = np.zeros_like(rewards, dtype=float)
        v_matrix = np.zeros_like(rewards, dtype=float)
        td_matrix[mask] = td_residual
        v_matrix[mask] = v_hat

        dr_values = v_matrix[:, 0] + np.sum(cumulative * td_matrix, axis=1)
        return dr_values
