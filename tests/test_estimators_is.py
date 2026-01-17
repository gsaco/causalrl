import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.data.datasets import LoggedBanditDataset, TrajectoryDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.importance_sampling import ISEstimator, PDISEstimator, WISEstimator
from crl.estimators.utils import compute_trajectory_returns
from crl.policies.tabular import TabularPolicy


def test_is_wis_bandit_matches_on_policy_mean():
    rng = np.random.default_rng(0)
    policy = TabularPolicy(np.array([[0.5, 0.5]]))
    contexts = np.zeros(200, dtype=int)
    actions = rng.integers(0, 2, size=200)
    rewards = rng.normal(size=200)
    behavior_action_probs = np.full(200, 0.5)

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
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
    )

    expected = float(np.mean(rewards))
    is_report = ISEstimator(estimand).estimate(dataset)
    wis_report = WISEstimator(estimand).estimate(dataset)

    assert abs(is_report.value - expected) < 1e-6
    assert abs(wis_report.value - expected) < 1e-6


def test_is_pdis_trajectory_matches_on_policy_mean():
    rng = np.random.default_rng(1)
    num_traj = 50
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
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
    )

    returns = compute_trajectory_returns(rewards, mask, dataset.discount)
    expected = float(np.mean(returns))

    is_report = ISEstimator(estimand).estimate(dataset)
    pdis_report = PDISEstimator(estimand).estimate(dataset)

    assert abs(is_report.value - expected) < 1e-6
    assert abs(pdis_report.value - expected) < 1e-6
