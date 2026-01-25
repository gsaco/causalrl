import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.gen_dice import GenDICEEstimator


def test_gendice_runs_on_tiny_mdp():
    config = SyntheticMDPConfig(
        seed=21,
        reward_noise_std=0.0,
        horizon=3,
        discount=0.9,
        num_states=4,
        num_actions=2,
    )
    bench = SyntheticMDP(config)
    bench.target_policy = bench.behavior_policy
    dataset = bench.sample(num_trajectories=300, seed=22)
    true_value = bench.true_policy_value(bench.target_policy)

    estimand = PolicyValueEstimand(
        policy=bench.target_policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, MARKOV]),
    )

    report = GenDICEEstimator(estimand).estimate(dataset)
    assert np.isfinite(report.value)
    assert abs(report.value - true_value) < 10.0
