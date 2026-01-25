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
BEHAVIOR_POLICY_KNOWN = Assumption(
    name="behavior_policy_known",
    description="Behavior policy propensities are known or correctly specified.",
)
PROPENSITY_MODEL_CORRECT = Assumption(
    name="propensity_model_correct",
    description="Estimated behavior propensities are correctly specified.",
)
Q_MODEL_REALIZABLE = Assumption(
    name="q_model_realizable",
    description="Value function lies in the chosen model class.",
)
BRIDGE_IDENTIFIABLE = Assumption(
    name="bridge_identifiable",
    description="Proximal bridge functions are identifiable and well-posed.",
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
    "BEHAVIOR_POLICY_KNOWN",
    "PROPENSITY_MODEL_CORRECT",
    "Q_MODEL_REALIZABLE",
    "BRIDGE_IDENTIFIABLE",
    "BOUNDED_REWARDS",
    "BOUNDED_CONFOUNDING",
]
