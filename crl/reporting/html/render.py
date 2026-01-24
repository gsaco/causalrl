"""HTML rendering for CRL reports."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from crl.reporting.schema import REPORT_SCHEMA_VERSION, ReportData


def render_html(
    report: ReportData | dict[str, Any],
    *,
    title: str | None = None,
    theme: str = "auto",
    assets_mode: str = "inline",
) -> str:
    """Render report payload to HTML."""

    payload = report.to_dict() if isinstance(report, ReportData) else report
    if payload.get("schema_version") != REPORT_SCHEMA_VERSION:
        # Leave rendering tolerant but annotate to metadata
        payload = dict(payload)
        payload.setdefault("metadata", {})
        payload["metadata"]["schema_warning"] = (
            f"Expected schema_version {REPORT_SCHEMA_VERSION}"
        )

    env = Environment(
        loader=PackageLoader("crl.reporting.html", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")

    css_text = ""
    js_text = ""
    if assets_mode == "inline":
        css_text = _read_asset("report.css")
        js_text = _read_asset("report.js")

    payload_json = json.dumps(payload, default=_to_jsonable)

    return template.render(
        title=title or "CausalRL Report",
        theme=theme,
        assets_mode=assets_mode,
        css_text=css_text,
        js_text=js_text,
        payload=payload,
        payload_json=payload_json,
    )


def _read_asset(name: str) -> str:
    with resources.files("crl.reporting.html.assets").joinpath(name).open(
        "r", encoding="utf-8"
    ) as f:
        return f.read()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(val) for val in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


__all__ = ["render_html"]
