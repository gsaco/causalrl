import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import BOUNDED_REWARDS, OVERLAP, SEQUENTIAL_IGNORABILITY, BEHAVIOR_POLICY_KNOWN
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.high_confidence import (
    HighConfidenceISConfig,
    HighConfidenceISEstimator,
)
from crl.estimators.importance_sampling import ISEstimator


def test_uncertainty_schema_kinds():
    bench = SyntheticBandit(SyntheticBanditConfig(seed=7))
    dataset = bench.sample(num_samples=200, seed=8)

    estimand = PolicyValueEstimand(
        policy=bench.target_policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN, BOUNDED_REWARDS]),
    )

    hcope_report = HighConfidenceISEstimator(
        estimand,
        config=HighConfidenceISConfig(delta=0.1, bound="empirical_bernstein"),
    ).estimate(dataset)
    assert hcope_report.uncertainty is not None
    assert hcope_report.uncertainty.kind == "empirical_bernstein"
    assert hcope_report.uncertainty.lower_bound is not None

    is_report = ISEstimator(estimand).estimate(dataset)
    assert is_report.uncertainty is not None
    assert is_report.uncertainty.kind == "wald"
    assert np.isfinite(is_report.uncertainty.level)
