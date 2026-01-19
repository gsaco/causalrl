"""Discrete policy wrappers and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from crl.policies.base import Policy


@dataclass
class StochasticPolicy(Policy):
    """Wrap a callable that returns action probabilities."""

    prob_fn: Callable[[np.ndarray], np.ndarray]
    action_space_n: int
    name: str | None = None

    def action_probs(self, observations: np.ndarray) -> np.ndarray:
        probs = np.asarray(self.prob_fn(observations), dtype=float)
        if probs.ndim != 2 or probs.shape[1] != self.action_space_n:
            raise ValueError(
                f"prob_fn must return shape (n, action_space_n), got {probs.shape}."
            )
        if np.any(probs < 0.0):
            raise ValueError("probabilities must be non-negative.")
        row_sums = probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("probabilities must sum to 1 across actions.")
        return probs

    def sample_action(
        self, observations: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        probs = self.action_probs(observations)
        return np.array([rng.choice(self.action_space_n, p=p) for p in probs])

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_type": "StochasticPolicy",
            "action_space_n": self.action_space_n,
            "name": self.name or "stochastic_policy",
        }


@dataclass
class CallablePolicy(Policy):
    """Wrap a callable that returns actions or action probabilities."""

    action_fn: Callable[[np.ndarray], np.ndarray]
    action_space_n: int
    returns: str = "actions"
    name: str | None = None

    def action_probs(self, observations: np.ndarray) -> np.ndarray:
        output = np.asarray(self.action_fn(observations))
        if self.returns == "actions":
            actions = output.reshape(-1).astype(int)
            probs = np.zeros((actions.size, self.action_space_n), dtype=float)
            probs[np.arange(actions.size), actions] = 1.0
            return probs
        if self.returns == "probs":
            probs = output.astype(float)
            if probs.ndim != 2 or probs.shape[1] != self.action_space_n:
                raise ValueError(
                    f"action_fn must return shape (n, action_space_n), got {probs.shape}."
                )
            if np.any(probs < 0.0):
                raise ValueError("probabilities must be non-negative.")
            row_sums = probs.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                raise ValueError("probabilities must sum to 1 across actions.")
            return probs
        raise ValueError("returns must be 'actions' or 'probs'.")

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_type": "CallablePolicy",
            "action_space_n": self.action_space_n,
            "returns": self.returns,
            "name": self.name or "callable_policy",
        }


class UniformPolicy(Policy):
    """Uniform random policy over discrete actions."""

    def __init__(self, action_space_n: int) -> None:
        if action_space_n <= 0:
            raise ValueError("action_space_n must be positive.")
        self.action_space_n = action_space_n

    def action_probs(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations)
        batch = obs.shape[0] if obs.ndim > 0 else 1
        return np.full((batch, self.action_space_n), 1.0 / self.action_space_n)

    def sample_action(
        self, observations: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        probs = self.action_probs(observations)
        return np.array([rng.choice(self.action_space_n, p=p) for p in probs])

    def to_dict(self) -> dict[str, object]:
        return {"policy_type": "UniformPolicy", "action_space_n": self.action_space_n}


__all__ = ["CallablePolicy", "StochasticPolicy", "UniformPolicy"]
