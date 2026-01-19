"""Transition dataset contract for (s, a, r, s', done) tuples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from crl.data.base import ensure_1d
from crl.data.trajectory import TrajectoryDataset
from crl.utils.validation import (
    require_finite,
    require_in_unit_interval,
    require_ndarray,
    require_same_length,
)


@dataclass
class TransitionDataset:
    """Transition dataset with optional episode ids and timesteps."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    behavior_action_probs: np.ndarray | None
    discount: float
    action_space_n: int | None = None
    episode_ids: np.ndarray | None = None
    timesteps: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate shapes, types, and ranges."""

        require_ndarray("states", self.states)
        require_ndarray("actions", self.actions)
        require_ndarray("rewards", self.rewards)
        require_ndarray("next_states", self.next_states)
        require_ndarray("dones", self.dones)
        if self.behavior_action_probs is not None:
            require_ndarray("behavior_action_probs", self.behavior_action_probs)
        if self.episode_ids is not None:
            require_ndarray("episode_ids", self.episode_ids)
        if self.timesteps is not None:
            require_ndarray("timesteps", self.timesteps)

        require_finite("states", self.states)
        require_finite("rewards", self.rewards)
        require_finite("next_states", self.next_states)
        if self.behavior_action_probs is not None:
            require_finite("behavior_action_probs", self.behavior_action_probs)

        if self.rewards.ndim != 1:
            raise ValueError(f"rewards must be 1D, got shape {self.rewards.shape}.")
        if self.dones.ndim != 1:
            raise ValueError(f"dones must be 1D, got shape {self.dones.shape}.")

        if self.states.shape[0] == 0:
            raise ValueError("dataset must contain at least one transition.")

        require_same_length(
            ["states", "actions", "rewards", "next_states", "dones"],
            [self.states, self.actions, self.rewards, self.next_states, self.dones],
        )

        if self.actions.ndim not in (1, 2):
            raise ValueError(
                f"actions must have shape (n,) or (n, d), got {self.actions.shape}."
            )

        if self.behavior_action_probs is not None:
            if self.behavior_action_probs.ndim != 1:
                raise ValueError(
                    "behavior_action_probs must be 1D, got "
                    f"{self.behavior_action_probs.shape}."
                )
            require_same_length(
                ["actions", "behavior_action_probs"],
                [self.actions, self.behavior_action_probs],
            )
            if self.action_space_n is not None:
                require_in_unit_interval(
                    "behavior_action_probs", self.behavior_action_probs
                )
            elif np.any(self.behavior_action_probs <= 0.0):
                raise ValueError(
                    "behavior_action_probs must be positive for densities."
                )

        if self.discount < 0.0 or self.discount > 1.0:
            raise ValueError("discount must be within [0, 1].")

        if self.action_space_n is not None:
            if self.action_space_n <= 0:
                raise ValueError("action_space_n must be positive.")
            if self.actions.ndim != 1:
                raise ValueError("actions must be 1D when action_space_n is provided.")
            if not np.issubdtype(self.actions.dtype, np.integer):
                raise ValueError(
                    "actions must be integer indices for discrete action spaces."
                )
            if np.any(self.actions < 0) or np.any(self.actions >= self.action_space_n):
                raise ValueError(
                    "actions must be within [0, action_space_n), got "
                    f"min={int(self.actions.min())}, max={int(self.actions.max())}."
                )

        if self.episode_ids is not None:
            ep_ids = ensure_1d("episode_ids", self.episode_ids)
            require_same_length(["episode_ids", "states"], [ep_ids, self.states])
            if not np.issubdtype(ep_ids.dtype, np.integer):
                raise ValueError("episode_ids must be integer identifiers.")
            if np.any(ep_ids < 0):
                raise ValueError("episode_ids must be non-negative.")

        if self.timesteps is not None:
            tsteps = ensure_1d("timesteps", self.timesteps)
            require_same_length(["timesteps", "states"], [tsteps, self.states])
            if not np.issubdtype(tsteps.dtype, np.integer):
                raise ValueError("timesteps must be integer indices.")
            if np.any(tsteps < 0):
                raise ValueError("timesteps must be non-negative.")

    @property
    def num_steps(self) -> int:
        """Return the number of transitions."""

        return int(self.rewards.shape[0])

    @property
    def horizon(self) -> int:
        """Return horizon inferred from timesteps if available."""

        if self.timesteps is None:
            return 1
        return int(np.max(self.timesteps)) + 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataset to a dictionary of arrays."""

        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "dones": self.dones,
            "behavior_action_probs": self.behavior_action_probs,
            "discount": self.discount,
            "action_space_n": self.action_space_n,
            "episode_ids": self.episode_ids,
            "timesteps": self.timesteps,
            "metadata": self.metadata or {},
        }

    def describe(self) -> dict[str, Any]:
        """Return summary statistics for the dataset."""

        summary: dict[str, Any] = {
            "type": "transition",
            "num_steps": self.num_steps,
            "discount": float(self.discount),
            "action_space_n": None
            if self.action_space_n is None
            else int(self.action_space_n),
            "behavior_action_probs_present": self.behavior_action_probs is not None,
            "episode_ids_present": self.episode_ids is not None,
            "timesteps_present": self.timesteps is not None,
            "done_fraction": float(np.mean(self.dones.astype(float))),
            "reward_mean": float(np.mean(self.rewards)),
            "reward_std": float(np.std(self.rewards)),
            "reward_min": float(np.min(self.rewards)),
            "reward_max": float(np.max(self.rewards)),
        }
        if self.behavior_action_probs is not None:
            summary.update(
                {
                    "behavior_prob_min": float(np.min(self.behavior_action_probs)),
                    "behavior_prob_max": float(np.max(self.behavior_action_probs)),
                }
            )
        return summary

    def to_trajectory(self) -> TrajectoryDataset:
        """Convert transitions to a TrajectoryDataset if episodes are known."""

        if self.episode_ids is None or self.timesteps is None:
            raise ValueError(
                "episode_ids and timesteps are required to build trajectories."
            )
        if self.action_space_n is None:
            raise ValueError("action_space_n required to build discrete trajectories.")

        episode_ids = ensure_1d("episode_ids", self.episode_ids)
        timesteps = ensure_1d("timesteps", self.timesteps)
        unique_eps = np.unique(episode_ids)
        horizon = int(np.max(timesteps)) + 1

        obs_shape = self.states.shape[1:]
        next_shape = self.next_states.shape[1:]

        obs = np.zeros((unique_eps.size, horizon, *obs_shape), dtype=self.states.dtype)
        next_obs = np.zeros(
            (unique_eps.size, horizon, *next_shape), dtype=self.next_states.dtype
        )
        actions = np.zeros((unique_eps.size, horizon), dtype=self.actions.dtype)
        rewards = np.zeros((unique_eps.size, horizon), dtype=self.rewards.dtype)
        mask = np.zeros((unique_eps.size, horizon), dtype=bool)
        behavior_action_probs_src = self.behavior_action_probs
        behavior_action_probs = (
            np.zeros((unique_eps.size, horizon), dtype=behavior_action_probs_src.dtype)
            if behavior_action_probs_src is not None
            else None
        )

        index_map = {int(ep): idx for idx, ep in enumerate(unique_eps)}
        for idx in range(self.num_steps):
            ep = index_map[int(episode_ids[idx])]
            t = int(timesteps[idx])
            obs[ep, t] = self.states[idx]
            next_obs[ep, t] = self.next_states[idx]
            actions[ep, t] = self.actions[idx]
            rewards[ep, t] = self.rewards[idx]
            mask[ep, t] = True
            if behavior_action_probs is not None:
                assert behavior_action_probs_src is not None
                behavior_action_probs[ep, t] = behavior_action_probs_src[idx]

        state_space_n = None
        if np.issubdtype(self.states.dtype, np.integer) and self.states.ndim == 1:
            state_space_n = int(np.max(self.states)) + 1

        return TrajectoryDataset(
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            behavior_action_probs=behavior_action_probs,
            mask=mask,
            discount=self.discount,
            action_space_n=int(self.action_space_n),
            state_space_n=state_space_n,
            metadata=self.metadata,
        )

    def __repr__(self) -> str:
        return f"TransitionDataset(num_steps={self.num_steps})"
