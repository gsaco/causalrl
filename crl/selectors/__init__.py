"""Estimator selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators import (
    DoubleRLEstimator,
    DoublyRobustEstimator,
    DualDICEEstimator,
    FQEEstimator,
    HighConfidenceISEstimator,
    ISEstimator,
    MAGICEstimator,
    MarginalizedImportanceSamplingEstimator,
    MRDREstimator,
    OPEEstimator,
    PDISEstimator,
    WeightedDoublyRobustEstimator,
    WISEstimator,
)


@dataclass
class SelectionResult:
    """Result of estimator selection."""

    best: OPEEstimator
    scores: list[dict[str, Any]]


def select_estimator(
    dataset: Any,
    estimand: PolicyValueEstimand,
    candidates: Iterable[Any],
    *,
    criterion: str = "stability_weighted_mse_proxy",
    return_scores: bool = False,
) -> OPEEstimator | SelectionResult:
    """Select an estimator using diagnostics-based heuristics."""

    estimators = [_resolve_candidate(candidate, estimand) for candidate in candidates]
    score_fn = _criterion_fn(criterion)
    scores: list[dict[str, Any]] = []

    best_estimator: OPEEstimator | None = None
    best_score = float("-inf")
    for estimator in estimators:
        report = estimator.estimate(dataset)
        score = score_fn(report)
        scores.append(
            {
                "estimator": report.metadata.get("estimator", type(estimator).__name__),
                "score": score,
                "diagnostics": report.diagnostics,
                "warnings": report.warnings,
            }
        )
        if score > best_score:
            best_score = score
            best_estimator = estimator

    if best_estimator is None:
        raise ValueError("No estimators were provided for selection.")

    if return_scores:
        return SelectionResult(best=best_estimator, scores=scores)
    return best_estimator


def _resolve_candidate(candidate: Any, estimand: PolicyValueEstimand) -> OPEEstimator:
    if isinstance(candidate, OPEEstimator):
        return candidate
    if isinstance(candidate, str):
        return _estimator_from_name(candidate, estimand)
    if isinstance(candidate, type) and issubclass(candidate, OPEEstimator):
        return candidate(estimand)
    raise ValueError(f"Unsupported estimator candidate: {candidate}")


def _estimator_from_name(name: str, estimand: PolicyValueEstimand) -> OPEEstimator:
    registry: dict[str, type[OPEEstimator]] = {
        "is": ISEstimator,
        "wis": WISEstimator,
        "pdis": PDISEstimator,
        "dr": DoublyRobustEstimator,
        "wdr": WeightedDoublyRobustEstimator,
        "magic": MAGICEstimator,
        "mrdr": MRDREstimator,
        "mis": MarginalizedImportanceSamplingEstimator,
        "fqe": FQEEstimator,
        "dualdice": DualDICEEstimator,
        "double_rl": DoubleRLEstimator,
        "hcope": HighConfidenceISEstimator,
    }
    key = name.strip().lower()
    if key not in registry:
        raise ValueError(f"Unknown estimator name: {name}")
    return registry[key](estimand)


def _criterion_fn(name: str) -> Callable[[Any], float]:
    if name == "stability_weighted_mse_proxy":
        return _stability_weighted_mse_proxy
    raise ValueError(f"Unknown selection criterion: {name}")


def _stability_weighted_mse_proxy(report: Any) -> float:
    diagnostics = report.diagnostics or {}
    ess_ratio = diagnostics.get("ess", {}).get("ess_ratio", 0.0)
    max_weight = diagnostics.get("max_weight", 0.0)
    tail_fraction = diagnostics.get("weights", {}).get("tail_fraction", 0.0)
    model_mse = diagnostics.get("model", {}).get("q_model_mse", 0.0)

    score = ess_ratio
    if max_weight:
        score = score / (1.0 + max_weight)
    score -= tail_fraction
    score -= 0.1 * model_mse
    return float(score)


__all__ = ["SelectionResult", "select_estimator"]
