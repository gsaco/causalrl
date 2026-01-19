"""Tabular policy for discrete state-action spaces."""

from __future__ import annotations

import numpy as np

from crl.policies.base import Policy


class TabularPolicy(Policy):
    """Tabular policy defined by a probability table.

    Estimand:
        Not applicable.
    Assumptions:
        Actions are discrete and state indices are valid.
    Inputs:
        table: Array with shape (num_states, num_actions) of action probabilities.
    Outputs:
        action_probs: Array with shape (n, num_actions).
    Failure modes:
        Raises ValueError if rows do not sum to 1 or contain negative values.
    """

    def __init__(self, table: np.ndarray) -> None:
        self.table = np.asarray(table, dtype=float)
        self._validate()

    def _validate(self) -> None:
        if self.table.ndim != 2:
            raise ValueError("table must have shape (num_states, num_actions).")
        if np.any(self.table < 0.0):
            raise ValueError("table entries must be non-negative.")
        row_sums = self.table.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("table rows must sum to 1.")

    @property
    def num_states(self) -> int:
        """Return the number of states."""

        return int(self.table.shape[0])

    @property
    def num_actions(self) -> int:
        """Return the number of actions."""

        return int(self.table.shape[1])

    def action_probs(self, observations: np.ndarray) -> np.ndarray:
        """Return action probabilities for each observation."""

        obs = np.asarray(observations)
        if obs.ndim == 2 and obs.shape[1] == 1:
            obs = obs.reshape(-1)
        if obs.ndim != 1:
            raise ValueError("TabularPolicy expects integer state observations.")
        if not np.issubdtype(obs.dtype, np.integer):
            raise ValueError("Observations must be integer state indices.")
        if np.any(obs < 0) or np.any(obs >= self.num_states):
            raise ValueError("Observations out of bounds for tabular policy.")
        return self.table[obs]

    def sample_action(
        self, observations: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample actions for observations."""

        probs = self.action_probs(observations)
        return np.array([rng.choice(self.num_actions, p=p) for p in probs])

    def to_dict(self) -> dict[str, object]:
        """Return a dictionary representation."""

        return {"policy_type": "TabularPolicy", "table": self.table}

    def __repr__(self) -> str:
        return f"TabularPolicy(num_states={self.num_states}, num_actions={self.num_actions})"
