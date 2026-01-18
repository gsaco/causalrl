import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.importance_sampling import ISEstimator, WISEstimator
from crl.selectors import select_estimator


def test_select_estimator_returns_instance():
    benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
    dataset = benchmark.sample(num_samples=200, seed=1)

    estimand = PolicyValueEstimand(
        policy=benchmark.target_policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
    )

    best = select_estimator(dataset, estimand, candidates=[ISEstimator, WISEstimator])
    assert isinstance(best, (ISEstimator, WISEstimator))
