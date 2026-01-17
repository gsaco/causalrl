import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.fqe import FQEConfig, FQEEstimator


def test_fqe_estimator_on_synthetic_mdp():
    config = SyntheticMDPConfig(seed=2, reward_noise_std=0.0, horizon=4)
    benchmark = SyntheticMDP(config)
    benchmark.target_policy = benchmark.behavior_policy
    dataset = benchmark.sample(num_trajectories=400, seed=2)
    true_value = benchmark.true_policy_value(benchmark.target_policy)

    estimand = PolicyValueEstimand(
        policy=benchmark.target_policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV]),
    )

    estimator = FQEEstimator(
        estimand,
        config=FQEConfig(num_iterations=5, num_epochs=10, batch_size=128, seed=0),
    )
    report = estimator.estimate(dataset)

    assert np.isfinite(report.value)
    assert abs(report.value - true_value) < 0.5
