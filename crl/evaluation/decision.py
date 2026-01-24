"""Decision support utilities for policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class DecisionSpec:
    baseline: str
    rule: Literal["lcb_vs_baseline", "robust_lcb_vs_baseline"] = "lcb_vs_baseline"
    min_improvement: float = 0.0


@dataclass
class DecisionResult:
    recommendation: str
    status: Literal["proceed", "caution", "do_not_deploy"]
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


__all__ = ["DecisionSpec", "DecisionResult"]
