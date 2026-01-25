import numpy as np
import pytest

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, SEQUENTIAL_IGNORABILITY
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.dr import DoublyRobustEstimator
from crl.policies.tabular import TabularPolicy


def test_missing_required_assumption_raises():
    policy = TabularPolicy(np.array([[1.0, 0.0]]))
    estimand = PolicyValueEstimand(
        policy=policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, MARKOV]),
    )

    with pytest.raises(ValueError):
        DoublyRobustEstimator(estimand)
