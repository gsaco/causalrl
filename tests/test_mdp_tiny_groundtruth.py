import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY, BEHAVIOR_POLICY_KNOWN
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.dr import DoublyRobustEstimator
from crl.estimators.magic import MAGICEstimator
from crl.estimators.mrdr import MRDREstimator
from crl.estimators.wdr import WeightedDoublyRobustEstimator


def test_dr_family_tiny_mdp_groundtruth():
    config = SyntheticMDPConfig(
        seed=11,
        reward_noise_std=0.0,
        horizon=3,
        discount=0.8,
        num_states=3,
        num_actions=2,
    )
    bench = SyntheticMDP(config)
    bench.target_policy = bench.behavior_policy
    dataset = bench.sample(num_trajectories=250, seed=12)
    true_value = bench.true_policy_value(bench.target_policy)

    estimand = PolicyValueEstimand(
        policy=bench.target_policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN, MARKOV]),
    )

    estimators = [
        DoublyRobustEstimator(estimand),
        WeightedDoublyRobustEstimator(estimand),
        MAGICEstimator(estimand),
        MRDREstimator(estimand),
    ]

    for estimator in estimators:
        report = estimator.estimate(dataset)
        assert np.isfinite(report.value)
        assert abs(report.value - true_value) < 3.0
