import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY, BEHAVIOR_POLICY_KNOWN
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.drl import DRLConfig, DRLEstimator


def test_drl_tiny_mdp_runs():
    config = SyntheticMDPConfig(
        seed=13,
        reward_noise_std=0.0,
        horizon=3,
        discount=0.9,
        num_states=3,
        num_actions=2,
    )
    bench = SyntheticMDP(config)
    bench.target_policy = bench.behavior_policy
    dataset = bench.sample(num_trajectories=300, seed=14)
    true_value = bench.true_policy_value(bench.target_policy)

    estimand = PolicyValueEstimand(
        policy=bench.target_policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN, MARKOV]),
    )

    report = DRLEstimator(
        estimand, config=DRLConfig(min_prob=1e-3, clip_ratio=5.0)
    ).estimate(dataset)
    assert np.isfinite(report.value)
    assert abs(report.value - true_value) < 5.0
