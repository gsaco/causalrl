"""Policy interfaces."""

from __future__ import annotations

from abc import ABC

import numpy as np


class Policy(ABC):
    """Abstract policy interface for discrete or continuous action spaces.

    Estimand:
        Not applicable.
    Assumptions:
        None.
    Inputs:
        observations: Array with shape (n, d) or (n,) representing states.
    Outputs:
        action_probs: Array with shape (n, a) for discrete policies.
        action_density: Array with shape (n,) for continuous policies.
    Failure modes:
        Implementations should raise ValueError if probabilities are invalid.
    """

    def action_probs(self, observations: np.ndarray) -> np.ndarray:
        """Return action probabilities for each observation (discrete)."""

        raise NotImplementedError("action_probs is not implemented for this policy.")

    def action_prob(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return probabilities for selected actions (discrete)."""

        probs = self.action_probs(observations)
        actions = np.asarray(actions).reshape(-1)
        return probs[np.arange(probs.shape[0]), actions]

    def action_density(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return action densities for selected actions (continuous)."""

        raise NotImplementedError("action_density is not implemented for this policy.")

    def log_prob(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return log-probability or log-density for selected actions."""

        try:
            probs = self.action_prob(observations, actions)
        except NotImplementedError:
            probs = self.action_density(observations, actions)
        probs = np.asarray(probs, dtype=float)
        if np.any(probs <= 0.0):
            raise ValueError("Probabilities/densities must be positive for log_prob.")
        return np.log(probs)

    def sample_action(
        self, observations: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample actions for observations (optional)."""

        raise NotImplementedError("sample_action is not implemented for this policy.")

    def to_dict(self) -> dict[str, object]:
        """Return a dictionary representation."""

        return {"policy_type": type(self).__name__}

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
