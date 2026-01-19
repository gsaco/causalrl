"""Utility functions for estimators."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from crl.core.policy import Policy


def flatten_observations(observations: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Flatten (n, t, d) or (n, t) observations to (n*t, d_flat).

    Estimand:
        Not applicable.
    Assumptions:
        Observations have shape (n, t) or (n, t, d).
    Inputs:
        observations: Array of observations.
    Outputs:
        Flattened observations, n, t.
    Failure modes:
        Raises ValueError for invalid shapes.
    """

    obs = np.asarray(observations)
    if obs.ndim == 2:
        n, t = obs.shape
        flat = obs.reshape(n * t, 1)
        return flat, n, t
    if obs.ndim == 3:
        n, t, d = obs.shape
        flat = obs.reshape(n * t, d)
        return flat, n, t
    raise ValueError("observations must have shape (n, t) or (n, t, d).")


def compute_action_probs(
    policy: Policy,
    observations: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    """Compute action probabilities for observed actions.

    Estimand:
        Not applicable.
    Assumptions:
        Policy supports the provided observation representation.
    Inputs:
        policy: Policy to evaluate.
        observations: Array with shape (n, t, d) or (n, t).
        actions: Array with shape (n, t).
    Outputs:
        Array with shape (n, t) of action probabilities.
    Failure modes:
        Raises ValueError if observation shapes are invalid.
    """

    obs_flat, n, t = flatten_observations(observations)
    actions_flat = actions.reshape(n * t)
    probs_flat = policy.action_prob(obs_flat, actions_flat)
    return probs_flat.reshape(n, t)


def compute_trajectory_returns(
    rewards: np.ndarray,
    mask: np.ndarray,
    discount: float,
) -> np.ndarray:
    """Compute discounted returns per trajectory.

    Estimand:
        Not applicable.
    Assumptions:
        Rewards and mask share shape (n, t).
    Inputs:
        rewards: Array with shape (n, t).
        mask: Boolean array with shape (n, t).
        discount: Discount factor.
    Outputs:
        Array with shape (n,) of discounted returns.
    Failure modes:
        Incorrect shapes raise ValueError downstream.
    """

    rewards = np.asarray(rewards, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n, t = rewards.shape
    discounts = discount ** np.arange(t)
    returns = (rewards * mask * discounts).sum(axis=1)
    return returns


def compute_stepwise_returns(
    rewards: np.ndarray,
    mask: np.ndarray,
    discount: float,
) -> np.ndarray:
    """Compute discounted rewards per step.

    Estimand:
        Not applicable.
    Assumptions:
        Rewards and mask share shape (n, t).
    Inputs:
        rewards: Array with shape (n, t).
        mask: Boolean array with shape (n, t).
        discount: Discount factor.
    Outputs:
        Array with shape (n, t) of discounted rewards.
    Failure modes:
        Incorrect shapes raise ValueError downstream.
    """

    rewards = np.asarray(rewards, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    t = rewards.shape[1]
    discounts = discount ** np.arange(t)
    return rewards * mask * discounts
