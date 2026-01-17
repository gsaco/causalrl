"""Synthetic contextual bandit benchmark with ground-truth value."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crl.data.datasets import LoggedBanditDataset
from crl.policies.tabular import TabularPolicy


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


@dataclass
class SyntheticBanditConfig:
    """Configuration for the synthetic bandit benchmark.

    Estimand:
        Not applicable.
    Assumptions:
        None.
    Inputs:
        num_contexts: Number of discrete contexts.
        num_actions: Number of discrete actions.
        reward_noise_std: Reward noise standard deviation.
        seed: Random seed for benchmark generation.
    Failure modes:
        None.
    """

    num_contexts: int = 5
    num_actions: int = 4
    reward_noise_std: float = 0.1
    seed: int = 0


class SyntheticBandit:
    """Synthetic bandit with discrete contexts and known reward means.

    Estimand:
        Policy value under intervention for a target policy.
    Assumptions:
        None (ground-truth generator).
    Inputs:
        config: SyntheticBanditConfig.
    Outputs:
        Methods provide sampled datasets and ground-truth values.
    Failure modes:
        None.
    """

    def __init__(self, config: SyntheticBanditConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.context_probs = rng.dirichlet(np.ones(config.num_contexts))
        self.reward_means = rng.normal(0.0, 1.0, size=(config.num_contexts, config.num_actions))

        behavior_logits = rng.normal(size=(config.num_contexts, config.num_actions))
        target_logits = rng.normal(size=(config.num_contexts, config.num_actions))
        self.behavior_policy = TabularPolicy(_softmax(behavior_logits))
        self.target_policy = TabularPolicy(_softmax(target_logits))

    def sample(self, num_samples: int, seed: int | None = None) -> LoggedBanditDataset:
        """Sample a logged bandit dataset.

        Inputs:
            num_samples: Number of logged samples.
            seed: Optional RNG seed.
        Outputs:
            LoggedBanditDataset with propensities.
        Failure modes:
            None.
        """

        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        contexts = rng.choice(self.config.num_contexts, size=num_samples, p=self.context_probs)
        behavior_probs = self.behavior_policy.action_probs(contexts)
        actions = np.array([
            rng.choice(self.config.num_actions, p=prob) for prob in behavior_probs
        ])
        rewards = (
            self.reward_means[contexts, actions]
            + rng.normal(0.0, self.config.reward_noise_std, size=num_samples)
        )
        behavior_action_probs = behavior_probs[np.arange(num_samples), actions]
        return LoggedBanditDataset(
            contexts=contexts,
            actions=actions,
            rewards=rewards,
            behavior_action_probs=behavior_action_probs,
            action_space_n=self.config.num_actions,
            metadata={"benchmark": "synthetic_bandit"},
        )

    def true_policy_value(self, policy: TabularPolicy) -> float:
        """Compute the ground-truth value for a policy.

        Inputs:
            policy: TabularPolicy.
        Outputs:
            Expected reward scalar.
        Failure modes:
            None.
        """

        expected_reward = np.sum(
            self.context_probs[:, None] * policy.table * self.reward_means
        )
        return float(expected_reward)
