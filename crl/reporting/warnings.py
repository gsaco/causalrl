"""Structured warnings for report payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WarningRecord:
    code: str
    severity: str
    message: str
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "context": dict(self.context or {}),
        }


def make_warning(
    code: str,
    message: str,
    *,
    severity: str = "warn",
    context: dict[str, Any] | None = None,
) -> WarningRecord:
    return WarningRecord(code=code, severity=severity, message=message, context=context)


def normalize_warnings(raw: list[Any] | None) -> list[dict[str, Any]]:
    """Normalize warnings into a list of dicts with code/severity/message."""

    if not raw:
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, WarningRecord):
            normalized.append(item.to_dict())
            continue
        if isinstance(item, dict):
            normalized.append(
                {
                    "code": item.get("code", "generic"),
                    "severity": item.get("severity", "warn"),
                    "message": item.get("message", str(item)),
                    "context": dict(item.get("context", {})),
                }
            )
            continue

        message = str(item)
        code, severity = _infer_code_and_severity(message)
        normalized.append(
            {
                "code": code,
                "severity": severity,
                "message": message,
                "context": {},
            }
        )
    return normalized


def _infer_code_and_severity(message: str) -> tuple[str, str]:
    msg = message.lower()
    if "overlap" in msg or "support" in msg:
        return "overlap_violation", "error"
    if "effective sample size" in msg or "ess" in msg:
        return "ess_too_low", "warn"
    if "heavy-tailed" in msg or "tail" in msg:
        return "extreme_tail", "warn"
    if "behavior" in msg and "below" in msg:
        return "behavior_prob_below_threshold", "warn"
    if "skipped estimator" in msg:
        return "estimator_skipped", "info"
    if "clipped" in msg:
        return "weights_clipped", "warn"
    return "generic", "warn"


__all__ = ["WarningRecord", "make_warning", "normalize_warnings"]
