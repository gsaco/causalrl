from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import BOUNDED_REWARDS, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.high_confidence import (
    HighConfidenceISConfig,
    HighConfidenceISEstimator,
)
from crl.estimators.importance_sampling import ISEstimator


def test_hcope_bound_is_lower_than_point_estimate():
    bench = SyntheticBandit(SyntheticBanditConfig(seed=15))
    dataset = bench.sample(num_samples=300, seed=16)

    estimand = PolicyValueEstimand(
        policy=bench.target_policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BOUNDED_REWARDS]),
    )

    hcope = HighConfidenceISEstimator(
        estimand, config=HighConfidenceISConfig(delta=0.1)
    ).estimate(dataset)
    is_report = ISEstimator(estimand).estimate(dataset)

    assert hcope.value <= is_report.value + 1e-6
    assert "clip" in hcope.metadata
    assert hcope.uncertainty is not None
    assert hcope.uncertainty.lower_bound is not None
    assert hcope.uncertainty.upper_bound is not None
