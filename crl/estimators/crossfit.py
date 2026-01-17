"""Cross-fitting utilities for nuisance models."""

from __future__ import annotations

import numpy as np


def make_folds(num_samples: int, num_folds: int, seed: int = 0) -> list[np.ndarray]:
    """Return shuffled fold indices for cross-fitting."""

    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    return list(np.array_split(indices, num_folds))
