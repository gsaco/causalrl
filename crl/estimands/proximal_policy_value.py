"""Proximal policy value estimands."""

from __future__ import annotations

from dataclasses import dataclass

from crl.assumptions import AssumptionSet
from crl.core.policy import Policy


@dataclass(frozen=True)
class ProximalPolicyValueEstimand:
    """Policy value estimand identified under proximal assumptions."""

    policy: Policy
    discount: float
    horizon: int | None
    assumptions: AssumptionSet

    def require(self, names: list[str]) -> None:
        """Require that assumptions include the specified names."""

        self.assumptions.require(names)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": type(self.policy).__name__,
            "discount": self.discount,
            "horizon": self.horizon,
            "assumptions": self.assumptions.names(),
        }

    def __repr__(self) -> str:
        return (
            "ProximalPolicyValueEstimand(policy="
            f"{type(self.policy).__name__}, discount={self.discount}, horizon={self.horizon})"
        )


__all__ = ["ProximalPolicyValueEstimand"]
