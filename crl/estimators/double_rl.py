"""Double reinforcement learning estimator utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from crl.data.datasets import LoggedBanditDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.base import DiagnosticsConfig, EstimatorReport, OPEEstimator, compute_ci
from crl.estimators.crossfit import make_folds
from crl.estimators.diagnostics import run_diagnostics
from crl.estimators.stats import mean_stderr


@dataclass
class DoubleRLConfig:
    """Configuration for Double RL cross-fitting."""

    num_folds: int = 2
    ridge: float = 1e-3
    seed: int = 0
    min_prob: float = 1e-6
    reward_model_factory: Callable[[int], Any] | None = None
    behavior_model_factory: Callable[[int, int], Any] | None = None


class LinearRewardModel:
    """Linear reward model with per-action ridge regression."""

    def __init__(self, num_actions: int, ridge: float) -> None:
        self.num_actions = num_actions
        self.ridge = ridge
        self.weights: np.ndarray | None = None

    def fit(self, contexts: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> None:
        features = _features(contexts)
        dim = features.shape[1]
        self.weights = np.zeros((self.num_actions, dim), dtype=float)
        for action in range(self.num_actions):
            mask = actions == action
            if not np.any(mask):
                continue
            x = features[mask]
            y = rewards[mask]
            xtx = x.T @ x + self.ridge * np.eye(dim)
            xty = x.T @ y
            self.weights[action] = np.linalg.solve(xtx, xty)

    def predict_all(self, contexts: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Reward model not fit.")
        features = _features(contexts)
        return features @ self.weights.T


class TabularBehaviorModel:
    """Tabular behavior policy estimator for discrete contexts."""

    def __init__(self, num_contexts: int, num_actions: int, min_prob: float) -> None:
        self.num_contexts = num_contexts
        self.num_actions = num_actions
        self.min_prob = min_prob
        self.table = np.ones((num_contexts, num_actions), dtype=float) / num_actions

    def fit(self, contexts: np.ndarray, actions: np.ndarray) -> None:
        counts = np.zeros_like(self.table)
        for c, a in zip(contexts, actions, strict=True):
            counts[int(c), int(a)] += 1.0
        row_sums = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        probs = counts / row_sums
        self.table = np.clip(probs, self.min_prob, 1.0)
        self.table = self.table / self.table.sum(axis=1, keepdims=True)

    def predict_proba(self, contexts: np.ndarray) -> np.ndarray:
        contexts = contexts.astype(int)
        return self.table[contexts]


class DoubleRLEstimator(OPEEstimator):
    """Double RL estimator for contextual bandits (Kallus & Uehara, 2020)."""

    required_assumptions = ["sequential_ignorability", "overlap"]

    def __init__(
        self,
        estimand: PolicyValueEstimand,
        run_diagnostics: bool = True,
        diagnostics_config: DiagnosticsConfig | None = None,
        config: DoubleRLConfig | None = None,
    ) -> None:
        super().__init__(estimand, run_diagnostics, diagnostics_config)
        self.config = config or DoubleRLConfig()

    def estimate(self, data: LoggedBanditDataset) -> EstimatorReport:
        self._validate_dataset(data)
        indices = np.arange(data.num_samples)
        folds = make_folds(data.num_samples, self.config.num_folds, self.config.seed)

        values = np.zeros(data.num_samples, dtype=float)

        for fold_idx in folds:
            train_idx = np.setdiff1d(indices, fold_idx)
            reward_model = (
                self.config.reward_model_factory(data.action_space_n)
                if self.config.reward_model_factory is not None
                else LinearRewardModel(data.action_space_n, self.config.ridge)
            )
            reward_model.fit(
                data.contexts[train_idx], data.actions[train_idx], data.rewards[train_idx]
            )

            behavior_probs = data.behavior_action_probs
            if behavior_probs is None:
                if data.contexts.ndim != 1 or not np.issubdtype(data.contexts.dtype, np.integer):
                    raise ValueError("behavior_action_probs missing and contexts are not discrete.")
                num_contexts = int(np.max(data.contexts)) + 1
                behavior_model = (
                    self.config.behavior_model_factory(num_contexts, data.action_space_n)
                    if self.config.behavior_model_factory is not None
                    else TabularBehaviorModel(
                        num_contexts, data.action_space_n, self.config.min_prob
                    )
                )
                behavior_model.fit(data.contexts[train_idx], data.actions[train_idx])
                behavior_probs = behavior_model.predict_proba(data.contexts[fold_idx])[
                    np.arange(fold_idx.size), data.actions[fold_idx]
                ]
            else:
                behavior_probs = behavior_probs[fold_idx]

            q_hat = reward_model.predict_all(data.contexts[fold_idx])
            pi_probs = self.estimand.policy.action_probs(data.contexts[fold_idx])
            mu_hat = np.clip(behavior_probs, self.config.min_prob, 1.0)
            mu_hat_pi = np.sum(pi_probs * q_hat, axis=1)
            q_hat_actions = q_hat[np.arange(fold_idx.size), data.actions[fold_idx]]
            pi_actions = pi_probs[np.arange(fold_idx.size), data.actions[fold_idx]]

            values[fold_idx] = mu_hat_pi + (pi_actions / mu_hat) * (
                data.rewards[fold_idx] - q_hat_actions
            )

        value = float(np.mean(values))
        stderr = mean_stderr(values)

        diagnostics: dict[str, Any] = {}
        warnings: list[str] = []
        if self.run_diagnostics and data.behavior_action_probs is not None:
            target_probs = self.estimand.policy.action_prob(data.contexts, data.actions)
            ratios = target_probs / data.behavior_action_probs
            diagnostics, warnings = run_diagnostics(
                ratios, target_probs, data.behavior_action_probs, None, self.diagnostics_config
            )

        return EstimatorReport(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={"estimator": "DoubleRL", "config": self.config.__dict__},
        )


def _features(contexts: np.ndarray) -> np.ndarray:
    contexts = np.asarray(contexts)
    if contexts.ndim == 1:
        return np.column_stack([np.ones_like(contexts, dtype=float), contexts.astype(float)])
    if contexts.ndim == 2:
        ones = np.ones((contexts.shape[0], 1), dtype=float)
        return np.concatenate([ones, contexts.astype(float)], axis=1)
    raise ValueError("contexts must be 1D or 2D for feature construction.")
