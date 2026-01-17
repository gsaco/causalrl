"""Validation helpers for datasets and inputs."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def require_ndarray(name: str, value: np.ndarray) -> None:
    """Validate that a value is a numpy array."""

    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(value).__name__}.")


def require_shape(name: str, value: np.ndarray, ndim: int) -> None:
    """Validate array dimensionality."""

    if value.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimensions, got shape {value.shape}."
        )


def require_same_length(names: Iterable[str], arrays: Iterable[np.ndarray]) -> None:
    """Ensure arrays share the same length on the first dimension."""

    arrays = list(arrays)
    lengths = [array.shape[0] for array in arrays]
    if len(set(lengths)) != 1:
        pairs = ", ".join(
            f"{name}={length}" for name, length in zip(names, lengths, strict=True)
        )
        raise ValueError(f"Arrays must share the same first dimension: {pairs}.")


def require_in_unit_interval(name: str, value: np.ndarray) -> None:
    """Validate that all entries are in (0, 1]."""

    if np.any(value <= 0.0) or np.any(value > 1.0):
        min_val = float(np.min(value))
        max_val = float(np.max(value))
        raise ValueError(
            f"{name} must be in the interval (0, 1], got min={min_val}, max={max_val}."
        )
