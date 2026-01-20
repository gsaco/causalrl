import numpy as np
import pytest

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import BOUNDED_CONFOUNDING, OVERLAP
from crl.data.datasets import LoggedBanditDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.policies.tabular import TabularPolicy
from crl.sensitivity.bandit import BanditPropensitySensitivity
from crl.sensitivity.namkoong2020 import confounded_ope_bounds


def test_bandit_sensitivity_curve_monotone():
    policy = TabularPolicy(np.array([[0.6, 0.4]]))
    contexts = np.zeros(100, dtype=int)
    actions = np.zeros(100, dtype=int)
    rewards = np.ones(100)
    behavior_action_probs = np.full(100, 0.5)

    dataset = LoggedBanditDataset(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        behavior_action_probs=behavior_action_probs,
        action_space_n=2,
    )

    estimand = PolicyValueEstimand(
        policy=policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([BOUNDED_CONFOUNDING, OVERLAP]),
    )

    sensitivity = BanditPropensitySensitivity(estimand)
    gammas = np.array([1.0, 1.5, 2.0])
    curve = sensitivity.curve(dataset, gammas)

    assert np.all(curve.lower[:-1] >= curve.lower[1:])
    assert np.all(curve.upper[:-1] <= curve.upper[1:])


def test_sequential_sensitivity_curve_monotone():
    from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig

    benchmark = SyntheticMDP(SyntheticMDPConfig(seed=0, horizon=3))
    dataset = benchmark.sample(num_trajectories=100, seed=1)
    gammas = np.array([1.0, 1.5, 2.0])
    curve = confounded_ope_bounds(dataset, benchmark.target_policy, gammas)

    assert np.all(curve.lower[:-1] >= curve.lower[1:])
    assert np.all(curve.upper[:-1] <= curve.upper[1:])


def test_bandit_sensitivity_requires_propensities():
    policy = TabularPolicy(np.array([[0.6, 0.4]]))
    contexts = np.zeros(10, dtype=int)
    actions = np.zeros(10, dtype=int)
    rewards = np.ones(10)

    dataset = LoggedBanditDataset(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        behavior_action_probs=None,
        action_space_n=2,
    )

    estimand = PolicyValueEstimand(
        policy=policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([BOUNDED_CONFOUNDING, OVERLAP]),
    )

    sensitivity = BanditPropensitySensitivity(estimand)
    with pytest.raises(ValueError, match="behavior_action_probs"):
        sensitivity.curve(dataset, np.array([1.0]))
