"""Base dataset protocols and validation helpers."""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BaseDataset(Protocol):
    """Protocol for dataset contracts used across estimators."""

    actions: np.ndarray
    rewards: np.ndarray
    behavior_action_probs: np.ndarray | None
    discount: float
    horizon: int

    def validate(self) -> None:
        """Validate internal fields."""

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

    def describe(self) -> dict[str, Any]:
        """Return a summary dictionary."""


def require_fields(dataset: Any, fields: Iterable[str]) -> None:
    """Ensure dataset exposes required fields and they are not None."""

    missing = []
    for field in fields:
        if not hasattr(dataset, field):
            missing.append(field)
            continue
        value = getattr(dataset, field)
        if value is None:
            missing.append(field)
    if missing:
        raise ValueError(
            "Dataset missing required fields: " + ", ".join(sorted(missing))
        )


def optional_field(dataset: Any, field: str) -> Any | None:
    """Return a dataset attribute if present, otherwise None."""

    return getattr(dataset, field, None)


def ensure_1d(name: str, value: np.ndarray) -> np.ndarray:
    """Ensure an array is 1D for scalar fields."""

    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}.")
    return arr
