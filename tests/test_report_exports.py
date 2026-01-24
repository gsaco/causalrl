from __future__ import annotations

import json

from crl.estimators.base import EstimatorReport


def test_report_json_and_html_exports(tmp_path):
    report = EstimatorReport(
        value=1.23,
        stderr=0.1,
        ci=(1.0, 1.4),
        diagnostics={"ess": 10.0},
        assumptions_checked=["overlap"],
        assumptions_flagged=[],
        warnings=[],
        metadata={"estimator": "IS"},
    )

    json_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"

    report.save_json(json_path)
    report.save_html(html_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["value"] == 1.23
    assert payload["uncertainty"]["kind"] == "wald"
    html = html_path.read_text(encoding="utf-8")
    assert "<html" in html
    assert "CausalRL Report" in html
