"""Policy value estimands."""

from __future__ import annotations

from dataclasses import dataclass

from crl.assumptions import AssumptionSet
from crl.core.policy import Policy


@dataclass(frozen=True)
class PolicyValueEstimand:
    """Policy value estimand under intervention.

    Estimand:
        V^pi = E[sum_t gamma^t R_t | do(A_t ~ pi(\\cdot | S_t))].
    Assumptions:
        Sequential ignorability, positivity/overlap, and correct data contract.
    Inputs:
        policy: Target policy.
        discount: Discount factor.
        horizon: Optional horizon for finite episodes.
        assumptions: AssumptionSet describing identification conditions.
    Outputs:
        Estimand specification used by estimators.
    Failure modes:
        If required assumptions are missing, estimators should refuse to run.
    """

    policy: Policy
    discount: float
    horizon: int | None
    assumptions: AssumptionSet

    def require(self, names: list[str]) -> None:
        """Require that assumptions include the specified names."""

        self.assumptions.require(names)

    def to_dict(self) -> dict[str, object]:
        """Return a dictionary representation of the estimand."""

        return {
            "policy": type(self.policy).__name__,
            "discount": self.discount,
            "horizon": self.horizon,
            "assumptions": self.assumptions.names(),
        }

    def __repr__(self) -> str:
        return (
            "PolicyValueEstimand(policy="
            f"{type(self.policy).__name__}, discount={self.discount}, "
            f"horizon={self.horizon}, assumptions={self.assumptions.names()})"
        )


@dataclass(frozen=True)
class PolicyContrastEstimand:
    """Contrast between two policy values.

    Estimand:
        V^{pi_treatment} - V^{pi_control}.
    Assumptions:
        Same as PolicyValueEstimand for both policies.
    Inputs:
        treatment: Target policy value estimand.
        control: Control policy value estimand.
    Outputs:
        Contrast specification used by estimators or reports.
    Failure modes:
        If assumptions differ, the contrast may not be identified.
    """

    treatment: PolicyValueEstimand
    control: PolicyValueEstimand

    def to_dict(self) -> dict[str, object]:
        """Return a dictionary representation."""

        return {
            "treatment": self.treatment.to_dict(),
            "control": self.control.to_dict(),
        }

    def __repr__(self) -> str:
        return "PolicyContrastEstimand(treatment=..., control=...)"
