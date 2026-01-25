"""Base classes for off-policy estimators."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Literal

from crl.estimands.policy_value import PolicyValueEstimand
from crl.reporting import EstimateRow, ReportData, ReportMetadata
from crl.reporting.html import render_html
from crl.reporting.warnings import normalize_warnings
from crl.version import __version__
from crl.utils.validation import validate_dataset

REPORT_SCHEMA_VERSION = 2


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
class UncertaintySummary:
    """Structured summary of estimator uncertainty."""

    kind: Literal[
        "wald",
        "bootstrap",
        "empirical_bernstein",
        "hoeffding",
        "none",
    ]
    level: float
    interval: tuple[float, float] | None
    lower_bound: float | None
    upper_bound: float | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "level": self.level,
            "interval": self.interval,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "notes": list(self.notes),
        }


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
        assumptions_checked: Assumptions required by the estimator.
        assumptions_flagged: Assumptions flagged by diagnostics.
        warnings: List of warning strings.
        metadata: Extra metadata (fit details, configs).
    Failure modes:
        Diagnostics may be None if disabled.
    """

    value: float
    stderr: float | None
    ci: tuple[float, float] | None
    diagnostics: dict[str, Any]
    assumptions_checked: list[str] = field(default_factory=list)
    assumptions_flagged: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    lower_bound: float | None = None
    upper_bound: float | None = None
    uncertainty: UncertaintySummary | None = None

    def __post_init__(self) -> None:
        if self.uncertainty is None:
            if self.ci is not None:
                self.uncertainty = UncertaintySummary(
                    kind="wald",
                    level=0.95,
                    interval=self.ci,
                    lower_bound=self.ci[0],
                    upper_bound=self.ci[1],
                    notes=["Normal-approximation interval (level inferred as 0.95)."],
                )
            else:
                self.uncertainty = UncertaintySummary(
                    kind="none",
                    level=0.0,
                    interval=None,
                    lower_bound=self.lower_bound,
                    upper_bound=self.upper_bound,
                    notes=[],
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a pandas-friendly dict representation."""

        lower = self.lower_bound
        upper = self.upper_bound
        if self.uncertainty is not None:
            if self.uncertainty.lower_bound is not None:
                lower = self.uncertainty.lower_bound
            if self.uncertainty.upper_bound is not None:
                upper = self.uncertainty.upper_bound
        if lower is None and self.ci is not None:
            lower = self.ci[0]
        if upper is None and self.ci is not None:
            upper = self.ci[1]

        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "value": self.value,
            "stderr": self.stderr,
            "ci": self.ci,
            "uncertainty": self.uncertainty.to_dict()
            if self.uncertainty is not None
            else None,
            "diagnostics": self.diagnostics,
            "assumptions_checked": list(self.assumptions_checked),
            "assumptions_flagged": list(self.assumptions_flagged),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
            "lower_bound": lower,
            "upper_bound": upper,
        }

    def to_dataframe(self) -> Any:
        """Return a one-row pandas DataFrame if pandas is available."""

        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pandas is required for to_dataframe().") from exc
        return pd.DataFrame([self.to_dict()])

    def to_json(self) -> str:
        """Return a JSON string representation."""

        return json.dumps(
            self.to_dict(), default=_to_jsonable, indent=2, sort_keys=True
        )

    def save_json(self, path: str | Path) -> None:
        """Write report contents to a JSON file."""

        path_obj = Path(path)
        path_obj.write_text(self.to_json(), encoding="utf-8")

    def to_html(self) -> str:
        """Return a self-contained HTML report representation."""

        estimate = EstimateRow(
            estimator=self.metadata.get("estimator", "Estimator"),
            value=self.value,
            stderr=self.stderr,
            ci=self.ci,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            assumptions_checked=self.assumptions_checked,
            assumptions_flagged=self.assumptions_flagged,
            warnings=normalize_warnings(self.warnings),
            diagnostics=self.diagnostics,
            metadata=self.metadata,
        )
        report_data = ReportData(
            mode="estimator",
            metadata=ReportMetadata(
                package_version=__version__,
                estimand=self.metadata.get("estimand"),
                configs={"estimator": self.metadata.get("estimator")},
            ),
            estimates=[estimate],
            diagnostics=self.diagnostics,
            warnings=normalize_warnings(self.warnings),
        )
        title = f"Estimator Report: {estimate.estimator}"
        return render_html(report_data, title=title)

    def save_html(self, path: str | Path) -> None:
        """Write report contents to an HTML file."""

        path_obj = Path(path)
        path_obj.write_text(self.to_html(), encoding="utf-8")

    def __repr__(self) -> str:
        return (
            "EstimatorReport(value="
            f"{self.value:.6f}, stderr={self.stderr}, ci={self.ci}, "
            f"diagnostics_keys={list(self.diagnostics.keys())}, "
            f"warnings={len(self.warnings)}, metadata_keys={list(self.metadata.keys())})"
        )


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
    required_fields: list[str] = []
    diagnostics_keys: list[str] = []

    def __init__(
        self,
        estimand: PolicyValueEstimand,
        run_diagnostics: bool = True,
        diagnostics_config: DiagnosticsConfig | None = None,
        bootstrap: bool = False,
        bootstrap_config: Any | None = None,
    ) -> None:
        self.estimand = estimand
        self.run_diagnostics = run_diagnostics
        self.diagnostics_config = diagnostics_config or DiagnosticsConfig()
        self.bootstrap = bootstrap
        self.bootstrap_config = bootstrap_config
        self.estimand.require(self.required_assumptions)
        self._bootstrap_params: dict[str, Any] = {
            "estimand": estimand,
            "run_diagnostics": False,
            "diagnostics_config": self.diagnostics_config,
            "bootstrap": False,
            "bootstrap_config": None,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(run_diagnostics={self.run_diagnostics})"

    def _validate_dataset(self, data: Any) -> None:
        """Run dataset validation if available."""

        validate_dataset(data, self.required_fields)

    def _behavior_policy_source(self, data: Any) -> str | None:
        source = getattr(data, "behavior_policy_source", None)
        if source is not None:
            return str(source)
        metadata = getattr(data, "metadata", None)
        if isinstance(metadata, dict):
            for key in (
                "behavior_policy_source",
                "behavior_policy",
                "propensity_source",
            ):
                if key in metadata:
                    return str(metadata[key])
        return None

    def _flag_assumptions(
        self, diagnostics: dict[str, Any], warnings: list[str]
    ) -> list[str]:
        flagged: set[str] = set()
        if "overlap" in self.required_assumptions:
            overlap = diagnostics.get("overlap", {})
            if overlap.get("support_violations", 0) > 0:
                flagged.add("overlap")
            if overlap.get("fraction_behavior_below_threshold", 0.0) > 0.0:
                flagged.add("overlap")
            if any("overlap" in warning.lower() for warning in warnings):
                flagged.add("overlap")
        if "bounded_rewards" in self.required_assumptions:
            if any(
                "reward" in warning.lower() and "bound" in warning.lower()
                for warning in warnings
            ):
                flagged.add("bounded_rewards")
        return sorted(flagged)

    def _build_report(
        self,
        *,
        value: float,
        stderr: float | None,
        ci: tuple[float, float] | None,
        diagnostics: dict[str, Any],
        warnings: list[str],
        metadata: dict[str, Any],
        data: Any | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        uncertainty: UncertaintySummary | None = None,
    ) -> EstimatorReport:
        warnings_out = list(warnings)
        behavior_source = None
        if data is not None:
            behavior_source = self._behavior_policy_source(data)
        if behavior_source == "estimated":
            warnings_out.append(
                "Behavior propensities were estimated; additional modeling risk may apply."
            )
        if not self.bootstrap and ci is not None and data is not None:
            try:
                from crl.data.datasets import TrajectoryDataset

                if isinstance(data, TrajectoryDataset):
                    warnings_out.append(
                        "Normal-approximation CIs assume independent trajectories; "
                        "consider bootstrap intervals for dependent or heavy-tailed data."
                    )
            except Exception:
                pass
        assumptions_checked = list(self.required_assumptions)
        assumptions_flagged = self._flag_assumptions(diagnostics, warnings_out)
        metadata_out = dict(metadata)
        metadata_out.setdefault("required_fields", list(self.required_fields))
        metadata_out.setdefault("diagnostics_keys", list(self.diagnostics_keys))
        if behavior_source is not None:
            metadata_out.setdefault("behavior_policy_source", behavior_source)
        if uncertainty is None:
            if ci is not None:
                uncertainty = UncertaintySummary(
                    kind="wald",
                    level=0.95,
                    interval=ci,
                    lower_bound=ci[0],
                    upper_bound=ci[1],
                    notes=["Normal-approximation interval (level inferred as 0.95)."],
                )
            else:
                uncertainty = UncertaintySummary(
                    kind="none",
                    level=0.0,
                    interval=None,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    notes=[],
                )
        report = EstimatorReport(
            value=value,
            stderr=stderr,
            ci=ci,
            diagnostics=diagnostics,
            assumptions_checked=assumptions_checked,
            assumptions_flagged=assumptions_flagged,
            warnings=warnings_out,
            metadata=metadata_out,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            uncertainty=uncertainty,
        )
        return self._apply_bootstrap(report, data)

    def _bootstrap_factory(self) -> Callable[[], "OPEEstimator"]:
        params = dict(self._bootstrap_params)

        def _factory() -> OPEEstimator:
            return type(self)(**params)

        return _factory

    def _apply_bootstrap(
        self, report: EstimatorReport, data: Any | None
    ) -> EstimatorReport:
        if not self.bootstrap:
            return report
        if data is None:
            report.warnings.append(
                "Bootstrap CI requested but no dataset was provided."
            )
            return report
        from crl.data.datasets import LoggedBanditDataset, TrajectoryDataset
        from crl.estimators.bootstrap import BootstrapConfig, bootstrap_ci

        if not isinstance(data, (LoggedBanditDataset, TrajectoryDataset)):
            report.warnings.append(
                "Bootstrap CI requested but dataset type is unsupported."
            )
            return report

        config = (
            self.bootstrap_config
            if isinstance(self.bootstrap_config, BootstrapConfig)
            else BootstrapConfig()
        )
        stderr, ci = bootstrap_ci(self._bootstrap_factory(), data, config)
        report.stderr = stderr
        report.ci = ci
        report.uncertainty = UncertaintySummary(
            kind="bootstrap",
            level=1.0 - config.alpha,
            interval=ci,
            lower_bound=ci[0],
            upper_bound=ci[1],
            notes=[
                f"Bootstrap CI with method={config.method}, num_bootstrap={config.num_bootstrap}."
            ],
        )
        report.metadata = dict(report.metadata)
        report.metadata.setdefault(
            "bootstrap",
            {
                "num_bootstrap": config.num_bootstrap,
                "method": config.method,
                "alpha": config.alpha,
            },
        )
        return report

    @abstractmethod
    def estimate(self, data: Any) -> EstimatorReport:
        """Estimate policy value from data."""


def compute_ci(
    value: float, stderr: float | None, alpha: float = 0.05
) -> tuple[float, float] | None:
    """Compute a normal-approximation confidence interval."""

    if stderr is None:
        return None
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    return (value - z * stderr, value + z * stderr)


EstimationReport = EstimatorReport


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(val) for val in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return str(value)
