"""Assumption objects used to gate estimators and document identification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Assumption:
    """Represents a causal identification assumption.

    Estimand:
        Not applicable.
    Assumptions:
        None.
    Inputs:
        name: Stable identifier for the assumption.
        description: Short human-readable description.
    Failure modes:
        Mislabeling assumptions can lead to applying an estimator outside its
        identification scope.
    """

    name: str
    description: str

    def to_dict(self) -> dict[str, str]:
        """Return a dictionary representation."""

        return {"name": self.name, "description": self.description}

    def __repr__(self) -> str:
        return f"Assumption(name={self.name!r})"


class AssumptionSet:
    """Collection of assumptions used to validate estimator applicability.

    Estimand:
        Not applicable.
    Assumptions:
        None.
    Inputs:
        assumptions: Iterable of Assumption objects.
    Outputs:
        None. Use `has`/`require` to check membership.
    Failure modes:
        If assumptions are missing, estimators should refuse to run.
    """

    def __init__(self, assumptions: Iterable[Assumption]) -> None:
        self._assumptions = {assumption.name: assumption for assumption in assumptions}

    def has(self, name: str) -> bool:
        """Return True if the assumption name is present."""

        return name in self._assumptions

    def require(self, names: Iterable[str]) -> None:
        """Raise ValueError if any required assumption is missing."""

        missing = [name for name in names if name not in self._assumptions]
        if missing:
            raise ValueError(
                "Missing required assumptions: " + ", ".join(sorted(missing))
            )

    def names(self) -> list[str]:
        """Return assumption names in sorted order."""

        return sorted(self._assumptions.keys())

    def to_dict(self) -> dict[str, dict[str, str]]:
        """Return a dictionary keyed by assumption name."""

        return {
            name: assumption.to_dict() for name, assumption in self._assumptions.items()
        }

    def __repr__(self) -> str:
        return f"AssumptionSet(names={self.names()})"
