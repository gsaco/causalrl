"""Dataset objects for logged bandit and trajectory data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from crl.utils.validation import (
    require_finite,
    require_in_unit_interval,
    require_ndarray,
    require_same_length,
    require_shape,
)


@dataclass
class LoggedBanditDataset:
    """Logged contextual bandit dataset.

    Estimand:
        Not applicable.
    Assumptions:
        None.
    Inputs:
        contexts: Array with shape (n, d) or (n,).
        actions: Array with shape (n,) of integer action indices.
        rewards: Array with shape (n,) of observed rewards.
        behavior_action_probs: Array with shape (n,) of propensities for actions.
        action_space_n: Number of discrete actions.
        metadata: Optional dictionary for provenance.
    Outputs:
        Dataset instance with validated fields.
    Failure modes:
        Raises ValueError if shapes mismatch or probabilities are invalid.
    """

    contexts: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    behavior_action_probs: np.ndarray | None
    action_space_n: int
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate shapes and value ranges."""

        require_ndarray("contexts", self.contexts)
        require_ndarray("actions", self.actions)
        require_ndarray("rewards", self.rewards)
        if self.behavior_action_probs is not None:
            require_ndarray("behavior_action_probs", self.behavior_action_probs)

        require_finite("contexts", self.contexts)
        require_finite("rewards", self.rewards)
        if self.behavior_action_probs is not None:
            require_finite("behavior_action_probs", self.behavior_action_probs)

        if self.contexts.ndim not in (1, 2):
            raise ValueError(
                "contexts must have shape (n,) or (n, d), got "
                f"{self.contexts.shape}."
            )
        require_shape("actions", self.actions, 1)
        require_shape("rewards", self.rewards, 1)
        if self.behavior_action_probs is not None:
            require_shape("behavior_action_probs", self.behavior_action_probs, 1)

        if self.actions.shape[0] == 0:
            raise ValueError("dataset must contain at least one sample.")

        if self.behavior_action_probs is not None:
            require_same_length(
                ["contexts", "actions", "rewards", "behavior_action_probs"],
                [self.contexts, self.actions, self.rewards, self.behavior_action_probs],
            )
            require_in_unit_interval("behavior_action_probs", self.behavior_action_probs)
        else:
            require_same_length(
                ["contexts", "actions", "rewards"],
                [self.contexts, self.actions, self.rewards],
            )

        if self.action_space_n <= 0:
            raise ValueError("action_space_n must be positive.")

        if not np.issubdtype(self.actions.dtype, np.integer):
            raise ValueError("actions must be integer indices.")

        if np.any(self.actions < 0) or np.any(self.actions >= self.action_space_n):
            raise ValueError(
                "actions must be within [0, action_space_n), got "
                f"min={int(self.actions.min())}, max={int(self.actions.max())}."
            )

    @property
    def num_samples(self) -> int:
        """Return the number of logged samples."""

        return int(self.actions.shape[0])

    @property
    def states(self) -> np.ndarray:
        """Alias for contexts to match the core Dataset interface."""

        return self.contexts

    @property
    def next_states(self) -> None:
        """Bandit data has no next-state field."""

        return None

    @property
    def dones(self) -> np.ndarray:
        """Bandit data is terminal after each sample."""

        return np.ones_like(self.actions, dtype=bool)

    @property
    def horizon(self) -> int:
        """Return horizon length for bandits (1)."""

        return 1

    @property
    def discount(self) -> float:
        """Return discount factor for bandits (1.0)."""

        return 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataset to a dictionary of arrays."""

        return {
            "contexts": self.contexts,
            "actions": self.actions,
            "rewards": self.rewards,
            "behavior_action_probs": self.behavior_action_probs,
            "action_space_n": self.action_space_n,
            "metadata": self.metadata or {},
        }

    def describe(self) -> dict[str, Any]:
        """Return summary statistics for the dataset."""

        context_dim = 1 if self.contexts.ndim == 1 else int(self.contexts.shape[1])
        summary: dict[str, Any] = {
            "type": "bandit",
            "num_samples": self.num_samples,
            "context_dim": context_dim,
            "action_space_n": int(self.action_space_n),
            "behavior_action_probs_present": self.behavior_action_probs is not None,
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

    def __repr__(self) -> str:
        return (
            "LoggedBanditDataset(num_samples="
            f"{self.num_samples}, action_space_n={self.action_space_n})"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggedBanditDataset":
        """Create a dataset from a serialized dictionary."""

        return cls(
            contexts=np.asarray(data["contexts"]),
            actions=np.asarray(data["actions"]),
            rewards=np.asarray(data["rewards"]),
            behavior_action_probs=(
                np.asarray(data["behavior_action_probs"])
                if data.get("behavior_action_probs") is not None
                else None
            ),
            action_space_n=int(data["action_space_n"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class TrajectoryDataset:
    """Logged finite-horizon trajectory dataset.

    Estimand:
        Not applicable.
    Assumptions:
        None.
    Inputs:
        observations: Array with shape (n, t, d) or (n, t).
        actions: Array with shape (n, t) of integer action indices.
        rewards: Array with shape (n, t) of rewards.
        next_observations: Array with shape matching observations.
        behavior_action_probs: Array with shape (n, t) of propensities.
        mask: Boolean array with shape (n, t) indicating valid steps.
        discount: Discount factor in [0, 1].
        action_space_n: Number of discrete actions.
        state_space_n: Optional number of discrete states for one-hot features.
        metadata: Optional dictionary for provenance.
    Outputs:
        Dataset instance with validated fields.
    Failure modes:
        Raises ValueError if shapes mismatch or probabilities are invalid.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    behavior_action_probs: np.ndarray | None
    mask: np.ndarray
    discount: float
    action_space_n: int
    state_space_n: int | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate shapes and value ranges."""

        require_ndarray("observations", self.observations)
        require_ndarray("actions", self.actions)
        require_ndarray("rewards", self.rewards)
        require_ndarray("next_observations", self.next_observations)
        if self.behavior_action_probs is not None:
            require_ndarray("behavior_action_probs", self.behavior_action_probs)
        require_ndarray("mask", self.mask)

        require_finite("observations", self.observations)
        require_finite("rewards", self.rewards)
        require_finite("next_observations", self.next_observations)
        if self.behavior_action_probs is not None:
            require_finite("behavior_action_probs", self.behavior_action_probs)

        if self.observations.ndim not in (2, 3):
            raise ValueError(
                "observations must have shape (n, t) or (n, t, d), got "
                f"{self.observations.shape}."
            )
        if self.next_observations.shape != self.observations.shape:
            raise ValueError(
                "next_observations must match observations shape, got "
                f"{self.next_observations.shape} vs {self.observations.shape}."
            )

        require_shape("actions", self.actions, 2)
        require_shape("rewards", self.rewards, 2)
        if self.behavior_action_probs is not None:
            require_shape("behavior_action_probs", self.behavior_action_probs, 2)
        require_shape("mask", self.mask, 2)

        if self.actions.shape != self.rewards.shape:
            raise ValueError(
                "actions and rewards must share shape (n, t), got "
                f"{self.actions.shape} vs {self.rewards.shape}."
            )
        if self.behavior_action_probs is not None and self.actions.shape != self.behavior_action_probs.shape:
            raise ValueError(
                "behavior_action_probs must share shape (n, t) with actions, got "
                f"{self.behavior_action_probs.shape} vs {self.actions.shape}."
            )
        if self.actions.shape != self.mask.shape:
            raise ValueError(
                "mask must share shape (n, t) with actions, got "
                f"{self.mask.shape} vs {self.actions.shape}."
            )

        if self.actions.shape[0] == 0:
            raise ValueError("dataset must contain at least one trajectory.")

        lengths = self.mask.sum(axis=1).astype(int)
        if np.any(lengths == 0):
            raise ValueError("mask must mark at least one valid step per trajectory.")
        for row in self.mask:
            false_indices = np.where(~row)[0]
            if false_indices.size == 0:
                continue
            first_false = int(false_indices[0])
            if np.any(row[first_false:]):
                raise ValueError(
                    "mask must be contiguous: valid steps must be a prefix of each trajectory."
                )

        if self.discount < 0.0 or self.discount > 1.0:
            raise ValueError("discount must be within [0, 1].")

        if self.action_space_n <= 0:
            raise ValueError("action_space_n must be positive.")

        if not np.issubdtype(self.actions.dtype, np.integer):
            raise ValueError("actions must be integer indices.")

        if self.mask.dtype != bool:
            raise ValueError("mask must be a boolean array.")

        if np.any(self.actions[self.mask] < 0) or np.any(
            self.actions[self.mask] >= self.action_space_n
        ):
            raise ValueError(
                "actions must be within [0, action_space_n) on valid steps."
            )

        if self.behavior_action_probs is not None:
            valid_probs = self.behavior_action_probs[self.mask]
            require_in_unit_interval("behavior_action_probs (masked)", valid_probs)

    @property
    def num_trajectories(self) -> int:
        """Return the number of trajectories."""

        return int(self.actions.shape[0])

    @property
    def horizon(self) -> int:
        """Return the horizon length (max steps)."""

        return int(self.actions.shape[1])

    @property
    def num_steps(self) -> int:
        """Return the number of valid steps (mask True)."""

        return int(self.mask.sum())

    @property
    def states(self) -> np.ndarray:
        """Alias for observations to match the core Dataset interface."""

        return self.observations

    @property
    def next_states(self) -> np.ndarray:
        """Alias for next_observations to match the core Dataset interface."""

        return self.next_observations

    @property
    def dones(self) -> np.ndarray:
        """Infer terminal flags from the mask (last valid step per trajectory)."""

        mask = np.asarray(self.mask, dtype=bool)
        dones = np.zeros_like(mask, dtype=bool)
        lengths = mask.sum(axis=1).astype(int)
        for idx, length in enumerate(lengths):
            if length > 0:
                dones[idx, length - 1] = True
        return dones

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataset to a dictionary of arrays."""

        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "behavior_action_probs": self.behavior_action_probs,
            "mask": self.mask,
            "discount": self.discount,
            "action_space_n": self.action_space_n,
            "state_space_n": self.state_space_n,
            "metadata": self.metadata or {},
        }

    def describe(self) -> dict[str, Any]:
        """Return summary statistics for the dataset."""

        obs_dim = 1 if self.observations.ndim == 2 else int(self.observations.shape[2])
        lengths = self.mask.sum(axis=1).astype(int)
        valid_rewards = self.rewards[self.mask]
        summary: dict[str, Any] = {
            "type": "trajectory",
            "num_trajectories": self.num_trajectories,
            "horizon": self.horizon,
            "num_steps": self.num_steps,
            "discount": float(self.discount),
            "action_space_n": int(self.action_space_n),
            "state_space_n": None if self.state_space_n is None else int(self.state_space_n),
            "observation_dim": obs_dim,
            "behavior_action_probs_present": self.behavior_action_probs is not None,
            "trajectory_length_min": int(lengths.min()),
            "trajectory_length_max": int(lengths.max()),
            "trajectory_length_mean": float(np.mean(lengths)),
            "reward_mean": float(np.mean(valid_rewards)),
            "reward_std": float(np.std(valid_rewards)),
            "reward_min": float(np.min(valid_rewards)),
            "reward_max": float(np.max(valid_rewards)),
        }
        if self.behavior_action_probs is not None:
            valid_probs = self.behavior_action_probs[self.mask]
            summary.update(
                {
                    "behavior_prob_min": float(np.min(valid_probs)),
                    "behavior_prob_max": float(np.max(valid_probs)),
                }
            )
        return summary

    def __repr__(self) -> str:
        return (
            "TrajectoryDataset(num_trajectories="
            f"{self.num_trajectories}, horizon={self.horizon}, action_space_n={self.action_space_n})"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryDataset":
        """Create a dataset from a serialized dictionary."""

        return cls(
            observations=np.asarray(data["observations"]),
            actions=np.asarray(data["actions"]),
            rewards=np.asarray(data["rewards"]),
            next_observations=np.asarray(data["next_observations"]),
            behavior_action_probs=(
                np.asarray(data["behavior_action_probs"])
                if data.get("behavior_action_probs") is not None
                else None
            ),
            mask=np.asarray(data["mask"], dtype=bool),
            discount=float(data["discount"]),
            action_space_n=int(data["action_space_n"]),
            state_space_n=(
                int(data["state_space_n"]) if data.get("state_space_n") is not None else None
            ),
            metadata=dict(data.get("metadata", {})),
        )


BanditDataset = LoggedBanditDataset


__all__ = ["LoggedBanditDataset", "BanditDataset", "TrajectoryDataset"]
