"""Policy protocol for CRL core interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    """Protocol for policies with discrete or continuous actions."""

    def action_probs(self, observations: np.ndarray) -> np.ndarray:
        """Return action probabilities for each observation (discrete)."""

    def action_prob(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return action probabilities for selected actions (discrete)."""

    def action_density(
        self, observations: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """Return action densities for selected actions (continuous)."""

    def log_prob(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return log-probabilities or log-densities for selected actions."""

    def sample_action(
        self, observations: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample actions for observations (optional)."""
