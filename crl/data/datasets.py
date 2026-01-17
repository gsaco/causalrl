"""Dataset objects for logged bandit and trajectory data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from crl.utils.validation import (
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
    behavior_action_probs: np.ndarray
    action_space_n: int
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate shapes and value ranges."""

        require_ndarray("contexts", self.contexts)
        require_ndarray("actions", self.actions)
        require_ndarray("rewards", self.rewards)
        require_ndarray("behavior_action_probs", self.behavior_action_probs)

        if self.contexts.ndim not in (1, 2):
            raise ValueError(
                "contexts must have shape (n,) or (n, d), got "
                f"{self.contexts.shape}."
            )
        require_shape("actions", self.actions, 1)
        require_shape("rewards", self.rewards, 1)
        require_shape("behavior_action_probs", self.behavior_action_probs, 1)

        require_same_length(
            ["contexts", "actions", "rewards", "behavior_action_probs"],
            [self.contexts, self.actions, self.rewards, self.behavior_action_probs],
        )
        require_in_unit_interval("behavior_action_probs", self.behavior_action_probs)

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoggedBanditDataset":
        """Create a dataset from a serialized dictionary."""

        return cls(
            contexts=np.asarray(data["contexts"]),
            actions=np.asarray(data["actions"]),
            rewards=np.asarray(data["rewards"]),
            behavior_action_probs=np.asarray(data["behavior_action_probs"]),
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
    behavior_action_probs: np.ndarray
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
        require_ndarray("behavior_action_probs", self.behavior_action_probs)
        require_ndarray("mask", self.mask)

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
        require_shape("behavior_action_probs", self.behavior_action_probs, 2)
        require_shape("mask", self.mask, 2)

        if self.actions.shape != self.rewards.shape:
            raise ValueError(
                "actions and rewards must share shape (n, t), got "
                f"{self.actions.shape} vs {self.rewards.shape}."
            )
        if self.actions.shape != self.behavior_action_probs.shape:
            raise ValueError(
                "behavior_action_probs must share shape (n, t) with actions, got "
                f"{self.behavior_action_probs.shape} vs {self.actions.shape}."
            )
        if self.actions.shape != self.mask.shape:
            raise ValueError(
                "mask must share shape (n, t) with actions, got "
                f"{self.mask.shape} vs {self.actions.shape}."
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryDataset":
        """Create a dataset from a serialized dictionary."""

        return cls(
            observations=np.asarray(data["observations"]),
            actions=np.asarray(data["actions"]),
            rewards=np.asarray(data["rewards"]),
            next_observations=np.asarray(data["next_observations"]),
            behavior_action_probs=np.asarray(data["behavior_action_probs"]),
            mask=np.asarray(data["mask"], dtype=bool),
            discount=float(data["discount"]),
            action_space_n=int(data["action_space_n"]),
            state_space_n=(
                int(data["state_space_n"]) if data.get("state_space_n") is not None else None
            ),
            metadata=dict(data.get("metadata", {})),
        )
