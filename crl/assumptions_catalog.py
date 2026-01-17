"""Default assumption objects for CRL estimands."""

from __future__ import annotations

from crl.assumptions import Assumption

SEQUENTIAL_IGNORABILITY = Assumption(
    name="sequential_ignorability",
    description="No unmeasured confounding given observed history.",
)
OVERLAP = Assumption(
    name="overlap",
    description="Behavior policy has support over target policy actions.",
)
MARKOV = Assumption(
    name="markov",
    description="State summarises history for transition and reward dynamics.",
)
CORRECT_MODEL = Assumption(
    name="correct_model",
    description="Function approximation is correctly specified.",
)
BOUNDED_REWARDS = Assumption(
    name="bounded_rewards",
    description="Rewards are bounded for concentration guarantees.",
)
BOUNDED_CONFOUNDING = Assumption(
    name="bounded_confounding",
    description="Unobserved confounding is bounded by a sensitivity parameter.",
)

__all__ = [
    "SEQUENTIAL_IGNORABILITY",
    "OVERLAP",
    "MARKOV",
    "CORRECT_MODEL",
    "BOUNDED_REWARDS",
    "BOUNDED_CONFOUNDING",
]
