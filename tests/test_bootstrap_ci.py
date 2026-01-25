from __future__ import annotations

import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY, BEHAVIOR_POLICY_KNOWN
from crl.data.datasets import LoggedBanditDataset, TrajectoryDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.bootstrap import BootstrapConfig
from crl.estimators.importance_sampling import ISEstimator, PDISEstimator
from crl.policies.tabular import TabularPolicy


def test_bootstrap_ci_bandit():
    rng = np.random.default_rng(0)
    policy = TabularPolicy(np.array([[0.5, 0.5]]))
    contexts = np.zeros(50, dtype=int)
    actions = rng.integers(0, 2, size=50)
    rewards = rng.normal(size=50)
    behavior_action_probs = np.full(50, 0.5)

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
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN]),
    )

    estimator = ISEstimator(
        estimand,
        bootstrap=True,
        bootstrap_config=BootstrapConfig(num_bootstrap=30, seed=0),
    )
    report = estimator.estimate(dataset)

    assert report.ci is not None
    assert "bootstrap" in report.metadata


def test_bootstrap_ci_trajectory():
    rng = np.random.default_rng(1)
    num_traj = 20
    horizon = 4
    obs = np.zeros((num_traj, horizon), dtype=int)
    actions = rng.integers(0, 2, size=(num_traj, horizon))
    rewards = rng.normal(size=(num_traj, horizon))
    next_obs = np.zeros((num_traj, horizon), dtype=int)
    behavior_action_probs = np.full((num_traj, horizon), 0.5)
    mask = np.ones((num_traj, horizon), dtype=bool)

    dataset = TrajectoryDataset(
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        behavior_action_probs=behavior_action_probs,
        mask=mask,
        discount=0.9,
        action_space_n=2,
        state_space_n=1,
    )

    policy = TabularPolicy(np.array([[0.5, 0.5]]))
    estimand = PolicyValueEstimand(
        policy=policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN, MARKOV]),
    )

    estimator = PDISEstimator(
        estimand,
        bootstrap=True,
        bootstrap_config=BootstrapConfig(num_bootstrap=20, seed=1),
    )
    report = estimator.estimate(dataset)

    assert report.ci is not None
    assert "bootstrap" in report.metadata
