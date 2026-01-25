"""Policy sweep evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from crl.evaluation.decision import DecisionResult, DecisionSpec
from crl.evaluation.result import EvaluationResult
from crl.evaluation.spec import DiagnosticsSpec, EvaluationSpec, ReportSpec, SensitivitySpec
from crl.reporting import EstimateRow, ReportData, ReportMetadata
from crl.reporting.export import save_bundle as save_report_bundle
from crl.reporting.html import render_html


@dataclass
class EvaluationSuite:
    results: dict[str, EvaluationResult]
    decision: DecisionResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def recommendation_table(self) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pandas is required for recommendation_table().") from exc

        rows = []
        for name, result in self.results.items():
            value, lower, upper, estimator = _primary_estimate(result)
            rows.append(
                {
                    "policy": name,
                    "estimator": estimator,
                    "value": value,
                    "lower_bound": lower,
                    "upper_bound": upper,
                }
            )
        return pd.DataFrame(rows)

    def to_report_data(self) -> ReportData:
        estimates: list[EstimateRow] = []
        for name, result in self.results.items():
            value, lower, upper, _ = _primary_estimate(result)
            estimates.append(
                EstimateRow(
                    estimator=name,
                    value=value,
                    lower_bound=lower,
                    upper_bound=upper,
                    metadata={"policy": name},
                )
            )

        metadata = ReportMetadata(
            run_name=self.metadata.get("run_name", "policy_sweep"),
            package_version=self.metadata.get("package_version"),
            dataset_fingerprint=self.metadata.get("dataset_fingerprint"),
            dataset_summary=self.metadata.get("dataset_summary"),
            configs={"num_policies": len(self.results)},
        )

        diagnostics: dict[str, Any] = {}
        if self.decision is not None:
            diagnostics["decision"] = {
                "recommendation": self.decision.recommendation,
                "status": self.decision.status,
                "details": self.decision.details,
                "warnings": self.decision.warnings,
            }

        return ReportData(
            mode="suite",
            metadata=metadata,
            estimates=estimates,
            diagnostics=diagnostics,
        )

    def to_html(self) -> str:
        report_data = self.to_report_data()
        return render_html(report_data, title="Policy Sweep Report")

    def save_html(self, path: str) -> None:
        html = self.to_html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    def save_bundle(self, output_dir: str) -> None:
        report_data = self.to_report_data()
        html = render_html(report_data, title="Policy Sweep Report")
        save_report_bundle(
            output_dir,
            html=html,
            report_data=report_data.to_dict(),
            summary=self.recommendation_table(),
            figures={},
            metadata=self.metadata,
        )


def evaluate_many(
    *,
    dataset: Any,
    policies: Mapping[str, Any],
    estimators: Any = "auto",
    diagnostics: DiagnosticsSpec | None = None,
    sensitivity: SensitivitySpec | None = None,
    report: ReportSpec | None = None,
    decision: DecisionSpec | None = None,
    seed: int = 0,
) -> EvaluationSuite:
    from crl.ope import evaluate

    results: dict[str, EvaluationResult] = {}
    for name, policy in policies.items():
        spec = EvaluationSpec(
            dataset=dataset,
            policy=policy,
            estimators=estimators,
            diagnostics=diagnostics or DiagnosticsSpec(),
            sensitivity=sensitivity or SensitivitySpec(),
            report=report or ReportSpec(),
            seed=seed,
        )
        results[name] = evaluate(spec)

    decision_result: DecisionResult | None = None
    if decision is not None:
        decision_result = _compute_decision(results, decision)

    suite = EvaluationSuite(results=results, decision=decision_result)
    if results:
        first = next(iter(results.values()))
        suite.metadata.update(
            {
                "dataset_fingerprint": first.report.metadata.get("dataset_fingerprint"),
                "dataset_summary": first.report.metadata.get("dataset_summary"),
                "package_version": first.report.metadata.get("package_version"),
            }
        )
    return suite


def _primary_estimate(
    result: EvaluationResult,
) -> tuple[float | None, float | None, float | None, str]:
    if not result.report.reports:
        return None, None, None, "unknown"
    name, report = next(iter(result.report.reports.items()))
    lower = report.lower_bound
    upper = report.upper_bound
    if lower is None and report.ci is not None:
        lower = report.ci[0]
    if upper is None and report.ci is not None:
        upper = report.ci[1]
    return report.value, lower, upper, name


def _compute_decision(
    results: Mapping[str, EvaluationResult],
    spec: DecisionSpec,
) -> DecisionResult:
    if spec.baseline not in results:
        return DecisionResult(
            recommendation=spec.baseline,
            status="caution",
            warnings=["Baseline policy not found in results."],
        )

    baseline_value, _, _, _ = _primary_estimate(results[spec.baseline])
    if baseline_value is None:
        return DecisionResult(
            recommendation=spec.baseline,
            status="caution",
            warnings=["Baseline estimate unavailable."],
        )

    best_policy = spec.baseline
    best_lcb = baseline_value
    for name, result in results.items():
        value, lower, _, _ = _primary_estimate(result)
        if value is None:
            continue
        lcb = lower if lower is not None else value
        if lcb > best_lcb + spec.min_improvement:
            best_lcb = lcb
            best_policy = name

    status: str = "proceed" if best_policy != spec.baseline else "caution"
    return DecisionResult(
        recommendation=best_policy,
        status=status,
        details={"baseline": spec.baseline, "baseline_value": baseline_value, "best_lcb": best_lcb},
    )


__all__ = ["EvaluationSuite", "evaluate_many"]
