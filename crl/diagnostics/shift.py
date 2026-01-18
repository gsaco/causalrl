"""Distribution shift diagnostics for state coverage."""

from __future__ import annotations

from typing import Any

import numpy as np

from crl.diagnostics.ess import effective_sample_size


def state_shift_diagnostics(
    states: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    max_samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate state distribution shift using weighted vs. unweighted samples."""

    x = _flatten_states(states)
    rng = np.random.default_rng(seed)

    if weights is None:
        weights = np.ones(x.shape[0], dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.shape[0] != x.shape[0]:
        raise ValueError("weights must be 1D and match number of states.")
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative.")
    w_sum = weights.sum()
    if w_sum <= 0:
        weights = np.ones_like(weights, dtype=float)
        w_sum = weights.sum()
    w_norm = weights / w_sum

    n = x.shape[0]
    sample_size = min(max_samples, n)
    uniform_idx = rng.choice(n, size=sample_size, replace=False)
    weighted_idx = rng.choice(n, size=sample_size, replace=True, p=w_norm)

    x_uniform = x[uniform_idx]
    x_weighted = x[weighted_idx]

    mean_uniform = np.mean(x_uniform, axis=0)
    mean_weighted = np.average(x, axis=0, weights=w_norm)
    mean_shift = float(np.linalg.norm(mean_weighted - mean_uniform))

    cov_uniform = np.cov(x_uniform, rowvar=False)
    cov_weighted = _weighted_covariance(x, w_norm)
    cov_shift = float(np.linalg.norm(cov_weighted - cov_uniform))

    mmd = _mmd_rbf(x_uniform, x_weighted)

    return {
        "mmd_rbf": float(mmd),
        "mean_shift_norm": mean_shift,
        "cov_shift_fro": cov_shift,
        "ess": float(effective_sample_size(weights)),
    }


def _flatten_states(states: np.ndarray) -> np.ndarray:
    x = np.asarray(states)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    return x.reshape(x.shape[0], -1)


def _weighted_covariance(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = np.average(x, axis=0, weights=weights)
    centered = x - mean
    weighted = centered * weights[:, None]
    return weighted.T @ centered


def _mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] == 0 or y.shape[0] == 0:
        return 0.0
    gamma = _median_heuristic(np.vstack([x, y]))
    k_xx = _rbf_kernel(x, x, gamma)
    k_yy = _rbf_kernel(y, y, gamma)
    k_xy = _rbf_kernel(x, y, gamma)
    return float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())


def _median_heuristic(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 1.0
    rng = np.random.default_rng(0)
    idx = rng.choice(x.shape[0], size=min(200, x.shape[0]), replace=False)
    subset = x[idx]
    dists = _pairwise_sq_dists(subset, subset)
    dists = dists[np.triu_indices_from(dists, k=1)]
    median = float(np.median(dists)) if dists.size else 1.0
    if median <= 0.0:
        return 1.0
    return 1.0 / median


def _pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x**2, axis=1)[:, None]
    y_norm = np.sum(y**2, axis=1)[None, :]
    return x_norm + y_norm - 2.0 * (x @ y.T)


def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    dists = _pairwise_sq_dists(x, y)
    return np.exp(-gamma * dists)


__all__ = ["state_shift_diagnostics"]
