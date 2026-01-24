"""Diagnostics registry and runners."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from crl.diagnostics.ess import effective_sample_size
from crl.diagnostics.overlap import compute_overlap_metrics
from crl.diagnostics.shift import state_shift_diagnostics
from crl.diagnostics.weights import weight_tail_stats
from crl.estimators.base import DiagnosticsConfig

DiagnosticFn = Callable[..., tuple[dict[str, Any], list[str]]]

REGISTRY: dict[str, DiagnosticFn] = {}


def register(name: str, fn: DiagnosticFn) -> None:
    REGISTRY[name] = fn


def run_suite(
    suites: list[str],
    *,
    weights: np.ndarray,
    target_action_probs: np.ndarray,
    behavior_action_probs: np.ndarray,
    mask: np.ndarray | None,
    config: DiagnosticsConfig,
    contexts: np.ndarray | None = None,
) -> tuple[dict[str, Any], list[str]]:
    diagnostics: dict[str, Any] = {}
    warnings: list[str] = []
    for name in suites:
        fn = REGISTRY.get(name)
        if fn is None:
            continue
        metrics, warn = fn(
            weights=weights,
            target_action_probs=target_action_probs,
            behavior_action_probs=behavior_action_probs,
            mask=mask,
            config=config,
            contexts=contexts,
        )
        diagnostics[name] = metrics
        warnings.extend(warn)
    return diagnostics, warnings


def _overlap(
    *,
    target_action_probs: np.ndarray,
    behavior_action_probs: np.ndarray,
    mask: np.ndarray | None,
    config: DiagnosticsConfig,
    **_: Any,
) -> tuple[dict[str, Any], list[str]]:
    metrics = compute_overlap_metrics(
        target_action_probs,
        behavior_action_probs,
        mask=mask,
        threshold=config.min_behavior_prob,
    )
    warnings: list[str] = []
    if metrics.get("support_violations", 0) > 0:
        warnings.append(
            "Detected overlap violations: target actions with low behavior support."
        )
    if metrics.get("fraction_behavior_below_threshold", 0.0) > 0.0:
        warnings.append("Behavior policy probabilities below minimum threshold.")
    return metrics, warnings


def _ess(
    *,
    weights: np.ndarray,
    config: DiagnosticsConfig,
    **_: Any,
) -> tuple[dict[str, Any], list[str]]:
    ess = effective_sample_size(weights)
    ess_ratio = ess / weights.size if weights.size else 0.0
    metrics = {"ess": ess, "ess_ratio": ess_ratio}
    warnings: list[str] = []
    if ess_ratio < config.ess_threshold:
        warnings.append(
            "Effective sample size ratio below threshold; estimates may be unstable."
        )
    return metrics, warnings


def _weights(
    *,
    weights: np.ndarray,
    config: DiagnosticsConfig,
    **_: Any,
) -> tuple[dict[str, Any], list[str]]:
    metrics = weight_tail_stats(
        weights,
        quantile=config.weight_tail_quantile,
        threshold=config.weight_tail_threshold,
    )
    warnings: list[str] = []
    if metrics.get("tail_fraction", 0.0) > 0.01:
        warnings.append("Heavy-tailed importance weights detected.")
    if config.max_weight is not None:
        max_weight = metrics.get("max", 0.0)
        if max_weight > config.max_weight:
            warnings.append("Maximum importance weight exceeds threshold.")
    return metrics, warnings


def _shift(
    *,
    contexts: np.ndarray | None,
    weights: np.ndarray,
    **_: Any,
) -> tuple[dict[str, Any], list[str]]:
    if contexts is None:
        return {"available": False}, []
    try:
        metrics = state_shift_diagnostics(contexts, weights=weights)
    except Exception:
        metrics = {"error": "shift diagnostics failed"}
    return metrics, []


register("overlap", _overlap)
register("ess", _ess)
register("weights", _weights)
register("shift", _shift)


__all__ = ["run_suite", "register", "REGISTRY"]
