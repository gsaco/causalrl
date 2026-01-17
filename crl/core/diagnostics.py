"""Diagnostics container for OPE reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Diagnostics:
    """Structured diagnostics with standard keys."""

    ess: dict[str, Any]
    overlap: dict[str, Any]
    weights: dict[str, Any]
    model: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""

        return {
            "ess": dict(self.ess),
            "overlap": dict(self.overlap),
            "weights": dict(self.weights),
            "model": dict(self.model),
        }

    def __repr__(self) -> str:
        return (
            "Diagnostics(ess_keys="
            f"{list(self.ess.keys())}, overlap_keys={list(self.overlap.keys())}, "
            f"weights_keys={list(self.weights.keys())}, model_keys={list(self.model.keys())})"
        )
