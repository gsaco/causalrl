import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY, BEHAVIOR_POLICY_KNOWN
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.dr import DoublyRobustEstimator, DRCrossFitConfig


def test_dr_estimator_on_synthetic_mdp():
    config = SyntheticMDPConfig(seed=0, reward_noise_std=0.0, horizon=4)
    benchmark = SyntheticMDP(config)
    benchmark.target_policy = benchmark.behavior_policy
    dataset = benchmark.sample(num_trajectories=500, seed=1)
    true_value = benchmark.true_policy_value(benchmark.target_policy)

    estimand = PolicyValueEstimand(
        policy=benchmark.target_policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN, MARKOV]),
    )

    estimator = DoublyRobustEstimator(
        estimand,
        config=DRCrossFitConfig(num_folds=2, num_iterations=10, seed=0),
    )
    report = estimator.estimate(dataset)

    assert np.isfinite(report.value)
    assert abs(report.value - true_value) < 0.2
