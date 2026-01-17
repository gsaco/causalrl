"""Policy interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    """Abstract policy interface for discrete action spaces.

    Estimand:
        Not applicable.
    Assumptions:
        None.
    Inputs:
        observations: Array with shape (n, d) or (n,) representing states.
    Outputs:
        action_probs: Array with shape (n, a) with probabilities for each action.
    Failure modes:
        Implementations should raise ValueError if probabilities are invalid.
    """

    @abstractmethod
    def action_probs(self, observations: np.ndarray) -> np.ndarray:
        """Return action probabilities for each observation."""

    def action_prob(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return probabilities for selected actions."""

        probs = self.action_probs(observations)
        actions = np.asarray(actions).reshape(-1)
        return probs[np.arange(probs.shape[0]), actions]

    @abstractmethod
    def sample_action(self, observations: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample actions for observations."""

    def to_dict(self) -> dict[str, object]:
        """Return a dictionary representation."""

        return {"policy_type": type(self).__name__}

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
