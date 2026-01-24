"""Calibration diagnostics for behavior propensities."""

from __future__ import annotations

from typing import Any


def behavior_calibration_from_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Return calibration diagnostics stored in dataset metadata, if present."""

    if not metadata:
        return {"available": False}
    diagnostics = metadata.get("behavior_policy_diagnostics")
    if diagnostics is None:
        return {"available": False}
    return {
        "available": True,
        "diagnostics": diagnostics,
    }


__all__ = ["behavior_calibration_from_metadata"]
