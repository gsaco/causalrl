"""High-confidence lower bound utilities for OPE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class HCOPEBoundResult:
    """Result for a single clipping parameter."""

    clip: float
    lower_bound: float
    clipped_mean: float
    bias_term: float


def empirical_bernstein_lower_bound(
    values: np.ndarray, bound: float, delta: float
) -> float:
    """Empirical Bernstein lower bound for bounded random variables."""

    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(v.mean())
    mean = float(np.mean(v))
    var = float(np.var(v, ddof=1))
    log_term = np.log(2.0 / delta)
    term1 = np.sqrt(2.0 * var * log_term / n)
    term2 = 7.0 * bound * log_term / (3.0 * (n - 1))
    return mean - term1 - term2


def hoeffding_lower_bound(values: np.ndarray, bound: float, delta: float) -> float:
    """Hoeffding lower bound for bounded random variables."""

    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return 0.0
    mean = float(np.mean(v))
    log_term = np.log(2.0 / delta)
    term = bound * np.sqrt(log_term / (2.0 * n))
    return mean - term


def hcope_lower_bound(
    *,
    returns: np.ndarray,
    weights: np.ndarray,
    reward_bound: float,
    delta: float,
    clip: float,
    bound_kind: Literal["empirical_bernstein", "hoeffding"],
) -> HCOPEBoundResult:
    """Compute a clipped IS lower bound with bias correction."""

    weights = np.asarray(weights, dtype=float)
    returns = np.asarray(returns, dtype=float)
    clipped = np.minimum(weights, clip)
    values = clipped * returns
    bound = float(clip * reward_bound)
    if bound_kind == "hoeffding":
        lcb = hoeffding_lower_bound(values, bound, delta)
    else:
        lcb = empirical_bernstein_lower_bound(values, bound, delta)

    # Conservative bias correction for clipping.
    bias_term = float(reward_bound * np.mean(np.maximum(weights - clip, 0.0)))
    return HCOPEBoundResult(
        clip=float(clip),
        lower_bound=float(lcb - bias_term),
        clipped_mean=float(np.mean(values)),
        bias_term=bias_term,
    )


def select_hcope_bound(
    *,
    returns: np.ndarray,
    weights: np.ndarray,
    reward_bound: float,
    delta: float,
    clip_grid: list[float],
    bound_kind: Literal["empirical_bernstein", "hoeffding"],
) -> tuple[HCOPEBoundResult, list[HCOPEBoundResult]]:
    """Evaluate a grid of clipping values and return the best bound."""

    results: list[HCOPEBoundResult] = []
    for clip in clip_grid:
        if clip <= 0:
            continue
        results.append(
            hcope_lower_bound(
                returns=returns,
                weights=weights,
                reward_bound=reward_bound,
                delta=delta,
                clip=clip,
                bound_kind=bound_kind,
            )
        )
    if not results:
        raise ValueError("clip_grid must contain at least one positive value.")
    best = max(results, key=lambda item: item.lower_bound)
    return best, results


__all__ = [
    "HCOPEBoundResult",
    "empirical_bernstein_lower_bound",
    "hoeffding_lower_bound",
    "hcope_lower_bound",
    "select_hcope_bound",
]
