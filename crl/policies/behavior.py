"""Behavior policy wrappers for known or estimated logging policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from crl.policies.base import Policy


@dataclass
class BehaviorPolicy(Policy):
    """Wrap a policy with metadata about how it was obtained."""

    policy: Policy
    source: str = "known"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def action_probs(self, observations: np.ndarray) -> np.ndarray:
        return self.policy.action_probs(observations)

    def action_prob(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.policy.action_prob(observations, actions)

    def action_density(
        self, observations: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        return self.policy.action_density(observations, actions)

    def log_prob(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.policy.log_prob(observations, actions)

    def sample_action(
        self, observations: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        return self.policy.sample_action(observations, rng)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_type": "BehaviorPolicy",
            "source": self.source,
            "wrapped": type(self.policy).__name__,
            "diagnostics": dict(self.diagnostics),
            "metadata": dict(self.metadata),
        }
