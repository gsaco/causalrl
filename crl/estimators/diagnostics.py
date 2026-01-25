"""Shared diagnostics for OPE estimators."""

from __future__ import annotations

from typing import Any

import numpy as np

from crl.diagnostics.registry import run_suite
from crl.estimators.base import DiagnosticsConfig


def run_diagnostics(
    weights: np.ndarray,
    target_action_probs: np.ndarray,
    behavior_action_probs: np.ndarray,
    mask: np.ndarray | None,
    config: DiagnosticsConfig,
    model_metrics: dict[str, Any] | None = None,
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

    diagnostics, warnings = run_suite(
        ["overlap", "ess", "weights"],
        weights=weights,
        target_action_probs=target_action_probs,
        behavior_action_probs=behavior_action_probs,
        mask=mask,
        config=config,
        contexts=None,
    )
    tail = diagnostics.get("weights", {})
    diagnostics["max_weight"] = tail.get("max", 0.0)
    diagnostics["model"] = model_metrics or {}
    return diagnostics, warnings
