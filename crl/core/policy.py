"""Policy protocol for CRL core interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    """Protocol for discrete-action policies.

    Implementations may use numpy, torch, or other backends as long as they
    return numpy arrays for probabilities.
    """

    def action_probs(self, observations: np.ndarray) -> np.ndarray:
        """Return action probabilities for each observation."""

    def action_prob(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return action probabilities for selected actions."""

    def sample_action(
        self, observations: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample actions for observations."""
