"""Proximal OPE for confounded bandits (simplified)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from crl.policies.base import Policy
from crl.data.datasets import LoggedBanditDataset


@dataclass
class ProximalBanditDataset:
    """Bandit dataset with proxy variables for proximal OPE."""

    contexts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    proxy_treatment: np.ndarray
    proxy_outcome: np.ndarray
    behavior_action_probs: np.ndarray | None
    action_space_n: int
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contexts": self.contexts,
            "actions": self.actions,
            "rewards": self.rewards,
            "proxy_treatment": self.proxy_treatment,
            "proxy_outcome": self.proxy_outcome,
            "behavior_action_probs": self.behavior_action_probs,
            "action_space_n": self.action_space_n,
            "metadata": self.metadata or {},
        }

    def __repr__(self) -> str:
        return f"ProximalBanditDataset(num_samples={self.actions.shape[0]})"

    def to_logged_dataset(self) -> LoggedBanditDataset:
        """Convert to a LoggedBanditDataset for standard OPE baselines."""

        return LoggedBanditDataset(
            contexts=self.contexts,
            actions=self.actions,
            rewards=self.rewards,
            behavior_action_probs=self.behavior_action_probs,
            action_space_n=self.action_space_n,
            metadata=self.metadata,
        )


@dataclass
class ProximalConfig:
    """Configuration for proximal OPE."""

    ridge: float = 1e-3


class ProximalBanditEstimator:
    """Simplified proximal OPE using linear bridge functions."""

    def __init__(self, policy: Policy, config: ProximalConfig | None = None) -> None:
        self.policy = policy
        self.config = config or ProximalConfig()

    def estimate(self, data: ProximalBanditDataset) -> float:
        contexts = np.asarray(data.contexts)
        actions = np.asarray(data.actions).astype(int)
        rewards = np.asarray(data.rewards, dtype=float)
        proxy_t = np.asarray(data.proxy_treatment, dtype=float)
        proxy_o = np.asarray(data.proxy_outcome, dtype=float)

        phi = _bridge_features(proxy_o, actions, contexts)
        psi = _instrument_features(proxy_t, actions, contexts)

        xtx = psi.T @ phi + self.config.ridge * np.eye(phi.shape[1])
        xty = psi.T @ rewards
        theta = np.linalg.solve(xtx, xty)

        policy_probs = self.policy.action_probs(contexts)
        expected_phi = _expected_bridge_features(proxy_o, contexts, policy_probs, data.action_space_n)
        return float(np.mean(expected_phi @ theta))


def _bridge_features(proxy_o: np.ndarray, actions: np.ndarray, contexts: np.ndarray) -> np.ndarray:
    actions = actions.reshape(-1)
    contexts = contexts.reshape(-1)
    proxy_o = proxy_o.reshape(-1)
    return np.column_stack(
        [
            np.ones_like(proxy_o),
            proxy_o,
            actions,
            contexts,
            proxy_o * actions,
            proxy_o * contexts,
        ]
    )


def _instrument_features(proxy_t: np.ndarray, actions: np.ndarray, contexts: np.ndarray) -> np.ndarray:
    actions = actions.reshape(-1)
    contexts = contexts.reshape(-1)
    proxy_t = proxy_t.reshape(-1)
    return np.column_stack(
        [
            np.ones_like(proxy_t),
            proxy_t,
            actions,
            contexts,
            proxy_t * actions,
            proxy_t * contexts,
        ]
    )


def _expected_bridge_features(
    proxy_o: np.ndarray,
    contexts: np.ndarray,
    policy_probs: np.ndarray,
    action_space_n: int,
) -> np.ndarray:
    proxy_o = proxy_o.reshape(-1)
    contexts = contexts.reshape(-1)
    num_samples = proxy_o.shape[0]
    features = np.zeros((num_samples, 6), dtype=float)
    for a in range(action_space_n):
        probs = policy_probs[:, a]
        actions = np.full(num_samples, a)
        phi = _bridge_features(proxy_o, actions, contexts)
        features += probs[:, None] * phi
    return features
