"""Report export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_bundle(
    output_dir: str | Path,
    *,
    html: str,
    report_data: dict[str, Any],
    summary: Any | None = None,
    figures: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a report bundle to disk.

    Layout:
        report.html
        report.json
        summary.csv (if summary provided)
        figures/*.png (if figures provided)
        metadata.json
    """

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    (out_path / "report.html").write_text(html, encoding="utf-8")
    (out_path / "report.json").write_text(
        json.dumps(report_data, indent=2, sort_keys=True, default=_to_jsonable),
        encoding="utf-8",
    )

    if summary is not None:
        try:
            summary.to_csv(out_path / "summary.csv", index=False)
        except Exception:
            # Fallback: best-effort serialization
            (out_path / "summary.csv").write_text(str(summary), encoding="utf-8")

    if figures:
        figures_dir = out_path / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        from crl.viz import save_figure

        for name, fig in figures.items():
            try:
                save_figure(fig, figures_dir / name)
            except Exception:
                continue

    meta_payload = metadata or {}
    (out_path / "metadata.json").write_text(
        json.dumps(meta_payload, indent=2, sort_keys=True, default=_to_jsonable),
        encoding="utf-8",
    )
    return out_path


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(val) for val in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


__all__ = ["save_bundle"]
