"""Reporting schema for CausalRL outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

REPORT_SCHEMA_VERSION = "1.0"


@dataclass
class ReportMetadata:
    """Metadata captured in report payloads."""

    run_name: str | None = None
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    package_version: str | None = None
    git_sha: str | None = None
    dataset_fingerprint: str | None = None
    dataset_summary: dict[str, Any] | None = None
    policy_name: str | None = None
    baseline_policy_name: str | None = None
    estimand: str | None = None
    seed: int | None = None
    configs: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "generated_at": self.generated_at,
            "package_version": self.package_version,
            "git_sha": self.git_sha,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_summary": self.dataset_summary,
            "policy_name": self.policy_name,
            "baseline_policy_name": self.baseline_policy_name,
            "estimand": self.estimand,
            "seed": self.seed,
            "configs": dict(self.configs),
            "environment": dict(self.environment),
        }


@dataclass
class EstimateRow:
    """Single row in the estimates table."""

    estimator: str
    value: float | None
    stderr: float | None = None
    ci: tuple[float, float] | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    assumptions_checked: list[str] = field(default_factory=list)
    assumptions_flagged: list[str] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimator": self.estimator,
            "value": self.value,
            "stderr": self.stderr,
            "ci": self.ci,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "assumptions_checked": list(self.assumptions_checked),
            "assumptions_flagged": list(self.assumptions_flagged),
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
            "metadata": dict(self.metadata),
        }


@dataclass
class ReportData:
    """Top-level report payload."""

    mode: str
    metadata: ReportMetadata
    estimates: list[EstimateRow] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    sensitivity: dict[str, Any] | None = None
    figures: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    tables: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "mode": self.mode,
            "metadata": _to_jsonable(self.metadata.to_dict()),
            "estimates": [estimate.to_dict() for estimate in self.estimates],
            "diagnostics": _to_jsonable(self.diagnostics),
            "sensitivity": _to_jsonable(self.sensitivity) if self.sensitivity else None,
            "figures": _to_jsonable(self.figures),
            "warnings": _to_jsonable(self.warnings),
            "tables": _to_jsonable(self.tables),
        }


def validate_minimal(payload: dict[str, Any]) -> list[str]:
    """Return a list of schema errors for missing required keys."""

    errors: list[str] = []
    for key in ("schema_version", "mode", "metadata", "estimates"):
        if key not in payload:
            errors.append(f"Missing required field: {key}")
    if payload.get("schema_version") != REPORT_SCHEMA_VERSION:
        errors.append(
            "Unsupported schema_version: "
            f"{payload.get('schema_version')} (expected {REPORT_SCHEMA_VERSION})"
        )
    return errors


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(val) for val in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "ReportMetadata",
    "EstimateRow",
    "ReportData",
    "validate_minimal",
]
