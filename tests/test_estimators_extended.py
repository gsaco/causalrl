import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import BOUNDED_REWARDS, MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.bootstrap import BootstrapConfig
from crl.estimators.double_rl import DoubleRLEstimator
from crl.estimators.dual_dice import DualDICEEstimator
from crl.estimators.fqe import FQEConfig, FQEEstimator
from crl.estimators.high_confidence import HighConfidenceISEstimator
from crl.estimators.importance_sampling import ISEstimator
from crl.estimators.magic import MAGICEstimator
from crl.estimators.mis import MarginalizedImportanceSamplingEstimator
from crl.estimators.mrdr import MRDREstimator
from crl.estimators.wdr import WeightedDoublyRobustEstimator


def test_extended_mdp_estimators_run_and_are_reasonable():
    bench = SyntheticMDP(SyntheticMDPConfig(seed=1, horizon=4, num_states=5, num_actions=3))
    dataset = bench.sample(num_trajectories=300, seed=2)
    true_value = bench.true_policy_value(bench.target_policy)

    estimand = PolicyValueEstimand(
        policy=bench.target_policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV]),
    )

    estimators = [
        WeightedDoublyRobustEstimator(estimand),
        MRDREstimator(estimand),
        MarginalizedImportanceSamplingEstimator(estimand),
        MAGICEstimator(estimand),
        DualDICEEstimator(estimand),
    ]

    for estimator in estimators:
        report = estimator.estimate(dataset)
        assert np.isfinite(report.value)
        assert abs(report.value - true_value) < 5.0


def test_double_rl_and_hcope_bandit():
    bench = SyntheticBandit(SyntheticBanditConfig(seed=3))
    dataset = bench.sample(num_samples=1000, seed=4)
    true_value = bench.true_policy_value(bench.target_policy)

    estimand = PolicyValueEstimand(
        policy=bench.target_policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BOUNDED_REWARDS]),
    )

    double_rl_report = DoubleRLEstimator(estimand).estimate(dataset)
    assert np.isfinite(double_rl_report.value)
    assert abs(double_rl_report.value - true_value) < 2.0

    hcope_report = HighConfidenceISEstimator(estimand).estimate(dataset)
    is_report = ISEstimator(estimand).estimate(dataset)
    assert hcope_report.value <= is_report.value + 1e-6


def test_fqe_bootstrap_ci():
    bench = SyntheticMDP(SyntheticMDPConfig(seed=5, horizon=3, num_states=4, num_actions=2))
    dataset = bench.sample(num_trajectories=40, seed=6)

    estimand = PolicyValueEstimand(
        policy=bench.target_policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV]),
    )

    config = FQEConfig(
        num_epochs=2,
        num_iterations=2,
        bootstrap=True,
        bootstrap_config=BootstrapConfig(num_bootstrap=10, method="trajectory", seed=1),
    )
    report = FQEEstimator(estimand, config=config).estimate(dataset)
    assert report.ci is not None
