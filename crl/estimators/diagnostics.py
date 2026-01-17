"""Shared diagnostics for OPE estimators."""

from __future__ import annotations

from typing import Any

import numpy as np

from crl.diagnostics.ess import effective_sample_size
from crl.diagnostics.overlap import compute_overlap_metrics
from crl.diagnostics.weights import weight_tail_stats
from crl.estimators.base import DiagnosticsConfig


def run_diagnostics(
    weights: np.ndarray,
    target_action_probs: np.ndarray,
    behavior_action_probs: np.ndarray,
    mask: np.ndarray | None,
    config: DiagnosticsConfig,
) -> tuple[dict[str, Any], list[str]]:
    """Run overlap and weight diagnostics and return warnings.

    Estimand:
        Not applicable.
    Assumptions:
        Weights and propensities are non-negative.
    Inputs:
        weights: Importance weights per sample or trajectory.
        target_action_probs: Target policy probabilities for observed actions.
        behavior_action_probs: Behavior propensities for observed actions.
        mask: Optional mask for valid steps.
        config: DiagnosticsConfig thresholds.
    Outputs:
        Tuple of diagnostics dict and warnings list.
    Failure modes:
        Returns empty diagnostics if inputs are empty.
    """

    overlap = compute_overlap_metrics(
        target_action_probs,
        behavior_action_probs,
        mask=mask,
        threshold=config.min_behavior_prob,
    )
    ess = effective_sample_size(weights)
    ess_ratio = ess / weights.size if weights.size else 0.0
    tail = weight_tail_stats(
        weights, quantile=config.weight_tail_quantile, threshold=config.weight_tail_threshold
    )
    diagnostics = {
        "overlap": overlap,
        "ess": {"ess": ess, "ess_ratio": ess_ratio},
        "weights": tail,
    }

    warnings: list[str] = []
    if overlap["support_violations"] > 0:
        warnings.append("Detected overlap violations: target actions with low behavior support.")
    if overlap["fraction_behavior_below_threshold"] > 0.0:
        warnings.append("Behavior policy probabilities below minimum threshold.")
    if ess_ratio < config.ess_threshold:
        warnings.append("Effective sample size ratio below threshold; estimates may be unstable.")
    if tail["tail_fraction"] > 0.01:
        warnings.append("Heavy-tailed importance weights detected.")
    return diagnostics, warnings
