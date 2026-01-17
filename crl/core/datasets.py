"""Core dataset interfaces and aliases."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from crl.data.datasets import LoggedBanditDataset as BanditDataset
from crl.data.datasets import TrajectoryDataset as TrajectoryDataset


@runtime_checkable
class Dataset(Protocol):
    """Protocol for datasets used by core pipelines."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray | None
    dones: np.ndarray | None
    behavior_action_probs: np.ndarray | None
    horizon: int
    discount: float

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""


__all__ = ["Dataset", "BanditDataset", "TrajectoryDataset"]
