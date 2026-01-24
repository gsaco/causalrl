"""Evaluation result container for CRL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from crl.evaluation.spec import EvaluationSpec
from crl.ope import OpeReport
from crl.reporting import ReportData
from crl.reporting.export import save_bundle as save_report_bundle
from crl.reporting.html import render_html


@dataclass
class EvaluationResult:
    spec: EvaluationSpec
    report: OpeReport
    metadata: dict[str, Any] | None = None

    @property
    def reports(self) -> dict[str, Any]:
        return self.report.reports

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self.report.diagnostics

    @property
    def figures(self) -> dict[str, Any]:
        return self.report.figures

    def summary(self) -> Any:
        return self.report.summary_table()

    def to_report_data(self) -> ReportData:
        report_data = self.report.to_report_data()
        spec_dict = _spec_to_dict(self.spec)
        report_data.metadata.configs.setdefault("spec", spec_dict)
        if not self.spec.report.include_figures:
            report_data.figures = []
        return report_data

    def to_json(self) -> str:
        payload = self.to_report_data().to_dict()
        return json.dumps(payload, indent=2, sort_keys=True)

    def to_html(self) -> str:
        report_data = self.to_report_data()
        return render_html(
            report_data,
            title="Off-Policy Evaluation Report",
            theme=self.spec.report.theme,
        )

    def save_html(self, path: str) -> None:
        html = self.to_html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    def save_bundle(self, output_dir: str) -> None:
        report_data = self.to_report_data()
        html = render_html(
            report_data,
            title="Off-Policy Evaluation Report",
            theme=self.spec.report.theme,
        )
        save_report_bundle(
            output_dir,
            html=html,
            report_data=report_data.to_dict(),
            summary=self.summary(),
            figures=self.figures if self.spec.report.include_figures else {},
            metadata=self.report.metadata,
        )


def _spec_to_dict(spec: EvaluationSpec) -> dict[str, Any]:
    return {
        "estimand": spec.estimand,
        "estimators": spec.estimators,
        "inference": {
            "alpha": spec.inference.alpha,
            "method": spec.inference.method,
            "bootstrap_num": spec.inference.bootstrap_num,
            "bootstrap_kind": spec.inference.bootstrap_kind,
            "bootstrap_block_size": spec.inference.bootstrap_block_size,
            "seed": spec.inference.seed,
        },
        "diagnostics": {
            "enabled": spec.diagnostics.enabled,
            "suites": list(spec.diagnostics.suites),
            "fail_on": list(spec.diagnostics.fail_on),
            "min_ess": spec.diagnostics.min_ess,
            "max_weight": spec.diagnostics.max_weight,
        },
        "sensitivity": {
            "enabled": spec.sensitivity.enabled,
            "kind": spec.sensitivity.kind,
            "baseline_value": spec.sensitivity.baseline_value,
            "gammas": None
            if spec.sensitivity.gammas is None
            else spec.sensitivity.gammas.tolist(),
        },
        "report": {
            "html": spec.report.html,
            "include_figures": spec.report.include_figures,
            "theme": spec.report.theme,
        },
        "seed": spec.seed,
    }


__all__ = ["EvaluationResult"]
