"""Confounded bandit benchmark with proxy variables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crl.confounding.proximal import ProximalBanditDataset
from crl.policies.tabular import TabularPolicy


@dataclass
class ConfoundedBanditConfig:
    """Configuration for the confounded bandit benchmark."""

    p_z: float = 0.8
    p_w: float = 0.8
    behavior_prob_u0: float = 0.2
    behavior_prob_u1: float = 0.8
    target_prob_z0: float = 0.3
    target_prob_z1: float = 0.7
    reward_base: tuple[float, float] = (0.0, 1.0)
    confounder_effect: float = 0.5
    reward_noise_std: float = 0.1
    seed: int = 0


class ConfoundedBandit:
    """Binary-action confounded bandit with proxies."""

    def __init__(self, config: ConfoundedBanditConfig) -> None:
        self.config = config
        self.target_policy = TabularPolicy(
            np.array(
                [
                    [1 - config.target_prob_z0, config.target_prob_z0],
                    [1 - config.target_prob_z1, config.target_prob_z1],
                ]
            )
        )

    def sample(self, num_samples: int, seed: int | None = None) -> ProximalBanditDataset:
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        u = rng.integers(0, 2, size=num_samples)
        z_flip = rng.random(num_samples) > self.config.p_z
        w_flip = rng.random(num_samples) > self.config.p_w
        z = np.where(z_flip, 1 - u, u)
        w = np.where(w_flip, 1 - u, u)

        p_a1 = np.where(u == 1, self.config.behavior_prob_u1, self.config.behavior_prob_u0)
        actions = rng.binomial(1, p_a1, size=num_samples)
        rewards = (
            np.array(self.config.reward_base)[actions]
            + self.config.confounder_effect * u
            + rng.normal(0.0, self.config.reward_noise_std, size=num_samples)
        )

        behavior_action_probs = self._behavior_propensity(z, actions)

        return ProximalBanditDataset(
            contexts=z,
            actions=actions,
            rewards=rewards,
            proxy_treatment=z,
            proxy_outcome=w,
            behavior_action_probs=behavior_action_probs,
            action_space_n=2,
            metadata={"benchmark": "confounded_bandit"},
        )

    def true_policy_value(self, policy: TabularPolicy, num_mc: int = 50000) -> float:
        rng = np.random.default_rng(self.config.seed + 123)
        u = rng.integers(0, 2, size=num_mc)
        z_flip = rng.random(num_mc) > self.config.p_z
        z = np.where(z_flip, 1 - u, u)
        pi = policy.action_probs(z)
        actions = np.array([rng.choice(2, p=p) for p in pi])
        rewards = (
            np.array(self.config.reward_base)[actions]
            + self.config.confounder_effect * u
            + rng.normal(0.0, self.config.reward_noise_std, size=num_mc)
        )
        return float(np.mean(rewards))

    def _behavior_propensity(self, z: np.ndarray, actions: np.ndarray) -> np.ndarray:
        p_u1_given_z = np.where(z == 1, self.config.p_z, 1.0 - self.config.p_z)
        p_a1 = (
            self.config.behavior_prob_u1 * p_u1_given_z
            + self.config.behavior_prob_u0 * (1.0 - p_u1_given_z)
        )
        probs = np.where(actions == 1, p_a1, 1.0 - p_a1)
        return probs
