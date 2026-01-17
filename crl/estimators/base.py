"""Base classes for off-policy estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from crl.estimands.policy_value import PolicyValueEstimand


@dataclass
class DiagnosticsConfig:
    """Configuration for diagnostics thresholds.

    Estimand:
        Not applicable.
    Assumptions:
        None.
    Inputs:
        min_behavior_prob: Minimum behavior probability threshold.
        max_weight: Optional clipping threshold for importance weights.
        ess_threshold: Minimum ESS ratio before warnings.
        weight_tail_quantile: Quantile for tail summary.
        weight_tail_threshold: Threshold to flag heavy tails.
    Outputs:
        Configuration object.
    Failure modes:
        None.
    """

    min_behavior_prob: float = 1e-3
    max_weight: float | None = None
    ess_threshold: float = 0.1
    weight_tail_quantile: float = 0.99
    weight_tail_threshold: float = 10.0


@dataclass
class EstimatorReport:
    """Report returned by estimators.

    Estimand:
        Policy value for the estimator's target policy.
    Assumptions:
        Assumptions are recorded in the estimand and warnings highlight issues.
    Outputs:
        value: Estimated policy value.
        stderr: Estimated standard error, if available.
        ci: Optional confidence interval (low, high).
        diagnostics: Dictionary of diagnostic metrics.
        warnings: List of warning strings.
        metadata: Extra metadata (fit details, configs).
    Failure modes:
        Diagnostics may be None if disabled.
    """

    value: float
    stderr: float | None
    ci: tuple[float, float] | None
    diagnostics: dict[str, Any]
    warnings: list[str]
    metadata: dict[str, Any]


class OPEEstimator(ABC):
    """Base class for off-policy evaluation estimators.

    Estimand:
        PolicyValueEstimand.
    Assumptions:
        Each estimator declares required assumptions.
    Inputs:
        Dataset-specific objects such as TrajectoryDataset or LoggedBanditDataset.
    Outputs:
        EstimatorReport with value, diagnostics, and metadata.
    Failure modes:
        Raises ValueError if required assumptions are missing.
    """

    required_assumptions: list[str] = []

    def __init__(
        self,
        estimand: PolicyValueEstimand,
        run_diagnostics: bool = True,
        diagnostics_config: DiagnosticsConfig | None = None,
    ) -> None:
        self.estimand = estimand
        self.run_diagnostics = run_diagnostics
        self.diagnostics_config = diagnostics_config or DiagnosticsConfig()
        self.estimand.require(self.required_assumptions)

    @abstractmethod
    def estimate(self, data: Any) -> EstimatorReport:
        """Estimate policy value from data."""


def compute_ci(
    value: float, stderr: float | None, alpha: float = 0.05
) -> tuple[float, float] | None:
    """Compute a normal-approximation confidence interval."""

    if stderr is None:
        return None
    z = 1.96 if alpha == 0.05 else 1.96
    return (value - z * stderr, value + z * stderr)
