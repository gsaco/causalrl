from __future__ import annotations

import json
import re

from crl.reporting.html import render_html
from crl.reporting.schema import ReportData, ReportMetadata


def test_report_html_escapes_script_breakout():
    payload = ReportData(
        mode="ope",
        metadata=ReportMetadata(
            run_name='</script><script>window.evil=true</script>'
        ),
        estimates=[],
    )
    html = render_html(payload)

    match = re.search(
        r"<script type=\"application/json\" id=\"crl-report-data\">(.*?)</script>",
        html,
        re.DOTALL,
    )
    assert match is not None
    payload_text = match.group(1)

    assert "</script>" not in payload_text

    data = json.loads(payload_text)
    assert data["metadata"]["run_name"] == '</script><script>window.evil=true</script>'
