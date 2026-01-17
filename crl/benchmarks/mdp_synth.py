"""Synthetic finite-horizon MDP benchmark with ground-truth value."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crl.data.datasets import TrajectoryDataset
from crl.policies.tabular import TabularPolicy


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


@dataclass
class SyntheticMDPConfig:
    """Configuration for the synthetic MDP benchmark.

    Estimand:
        Not applicable.
    Assumptions:
        None.
    Inputs:
        num_states: Number of discrete states.
        num_actions: Number of discrete actions.
        horizon: Episode horizon.
        discount: Discount factor.
        reward_noise_std: Reward noise standard deviation.
        seed: Random seed for benchmark generation.
    Failure modes:
        None.
    """

    num_states: int = 6
    num_actions: int = 3
    horizon: int = 5
    discount: float = 0.95
    reward_noise_std: float = 0.1
    transition_concentration: float = 1.0
    behavior_scale: float = 1.0
    target_scale: float = 1.0
    seed: int = 0


class SyntheticMDP:
    """Synthetic finite-horizon MDP with tabular dynamics.

    Estimand:
        Policy value under intervention for a target policy.
    Assumptions:
        None (ground-truth generator).
    Inputs:
        config: SyntheticMDPConfig.
    Outputs:
        Methods provide sampled datasets and ground-truth values.
    Failure modes:
        None.
    """

    def __init__(self, config: SyntheticMDPConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.initial_state_probs = rng.dirichlet(np.ones(config.num_states))
        self.transition_probs = rng.dirichlet(
            np.ones(config.num_states) * config.transition_concentration,
            size=(config.num_states, config.num_actions),
        )
        self.reward_means = rng.normal(0.0, 1.0, size=(config.num_states, config.num_actions))

        behavior_logits = rng.normal(size=(config.num_states, config.num_actions)) * config.behavior_scale
        target_logits = rng.normal(size=(config.num_states, config.num_actions)) * config.target_scale
        self.behavior_policy = TabularPolicy(_softmax(behavior_logits))
        self.target_policy = TabularPolicy(_softmax(target_logits))

    def sample(self, num_trajectories: int, seed: int | None = None) -> TrajectoryDataset:
        """Sample trajectories from the behavior policy.

        Inputs:
            num_trajectories: Number of trajectories.
            seed: Optional RNG seed.
        Outputs:
            TrajectoryDataset with propensities.
        Failure modes:
            None.
        """

        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        obs = np.zeros((num_trajectories, self.config.horizon), dtype=int)
        actions = np.zeros((num_trajectories, self.config.horizon), dtype=int)
        rewards = np.zeros((num_trajectories, self.config.horizon), dtype=float)
        next_obs = np.zeros((num_trajectories, self.config.horizon), dtype=int)
        behavior_action_probs = np.zeros((num_trajectories, self.config.horizon), dtype=float)
        mask = np.zeros((num_trajectories, self.config.horizon), dtype=bool)

        for i in range(num_trajectories):
            state = rng.choice(self.config.num_states, p=self.initial_state_probs)
            for t in range(self.config.horizon):
                mask[i, t] = True
                obs[i, t] = state
                action_probs = self.behavior_policy.action_probs(np.array([state]))[0]
                action = rng.choice(self.config.num_actions, p=action_probs)
                actions[i, t] = action
                behavior_action_probs[i, t] = action_probs[action]
                rewards[i, t] = (
                    self.reward_means[state, action]
                    + rng.normal(0.0, self.config.reward_noise_std)
                )
                next_state = rng.choice(
                    self.config.num_states, p=self.transition_probs[state, action]
                )
                next_obs[i, t] = next_state
                state = next_state

        return TrajectoryDataset(
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            behavior_action_probs=behavior_action_probs,
            mask=mask,
            discount=self.config.discount,
            action_space_n=self.config.num_actions,
            state_space_n=self.config.num_states,
            metadata={"benchmark": "synthetic_mdp"},
        )

    def true_policy_value(self, policy: TabularPolicy) -> float:
        """Compute the ground-truth policy value via dynamic programming.

        Inputs:
            policy: TabularPolicy.
        Outputs:
            Expected discounted return.
        Failure modes:
            None.
        """

        horizon = self.config.horizon
        discount = self.config.discount
        v = np.zeros((horizon + 1, self.config.num_states), dtype=float)

        for t in range(horizon - 1, -1, -1):
            for s in range(self.config.num_states):
                q = np.zeros(self.config.num_actions, dtype=float)
                for a in range(self.config.num_actions):
                    q[a] = self.reward_means[s, a] + discount * np.sum(
                        self.transition_probs[s, a] * v[t + 1]
                    )
                v[t, s] = np.sum(policy.table[s] * q)

        return float(np.sum(self.initial_state_probs * v[0]))
