"""Shared utilities for doubly robust estimators."""

from __future__ import annotations

import numpy as np


def discounted_powers(discount: float, horizon: int) -> np.ndarray:
    """Return gamma^t for t=0..horizon-1."""

    if horizon < 0:
        raise ValueError("horizon must be non-negative.")
    return np.power(float(discount), np.arange(horizon, dtype=float))


def dr_values_from_qv(
    *,
    rewards: np.ndarray,
    mask: np.ndarray,
    discount: float,
    cumulative_rho: np.ndarray,
    v_hat: np.ndarray,
    q_hat: np.ndarray,
) -> np.ndarray:
    """Compute per-trajectory DR values from Q/V predictions.

    Args:
        rewards: Array of shape (n, t).
        mask: Boolean array of shape (n, t) indicating valid steps.
        discount: Discount factor gamma.
        cumulative_rho: Cumulative importance ratios with shape (n, t).
        v_hat: Value predictions with shape (n, t + 1).
        q_hat: Q predictions for logged actions with shape (n, t).
    """

    rewards = np.asarray(rewards, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    cumulative_rho = np.asarray(cumulative_rho, dtype=float)
    v_hat = np.asarray(v_hat, dtype=float)
    q_hat = np.asarray(q_hat, dtype=float)

    if rewards.ndim != 2:
        raise ValueError("rewards must have shape (n, t).")
    n, t = rewards.shape
    if v_hat.shape != (n, t + 1):
        raise ValueError("v_hat must have shape (n, t+1).")
    if q_hat.shape != (n, t):
        raise ValueError("q_hat must have shape (n, t).")
    if cumulative_rho.shape != (n, t):
        raise ValueError("cumulative_rho must have shape (n, t).")

    discounts = discounted_powers(discount, t).reshape(1, t)
    td_residual = rewards + discount * v_hat[:, 1:] - q_hat
    td_residual = np.where(mask, td_residual, 0.0)
    rho = np.where(mask, cumulative_rho, 1.0)

    return v_hat[:, 0] + np.sum(discounts * rho * td_residual, axis=1)


__all__ = ["discounted_powers", "dr_values_from_qv"]
