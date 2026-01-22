import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.data.datasets import LoggedBanditDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.importance_sampling import ISEstimator
from crl.policies.tabular import TabularPolicy


def test_is_clipping_emits_warning_and_finite_value():
    contexts = np.zeros(5, dtype=int)
    actions = np.zeros(5, dtype=int)
    rewards = np.ones(5, dtype=float)
    behavior_probs = np.full(5, 0.01, dtype=float)
    dataset = LoggedBanditDataset(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        behavior_action_probs=behavior_probs,
        action_space_n=2,
    )

    policy = TabularPolicy(np.array([[1.0, 0.0]]))
    estimand = PolicyValueEstimand(
        policy=policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
    )

    estimator = ISEstimator(estimand, clip_rho=1.0)
    report = estimator.estimate(dataset)

    assert np.isfinite(report.value)
    assert any("Clipped" in warning for warning in report.warnings)
