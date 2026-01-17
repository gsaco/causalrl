import numpy as np
import pytest

from crl.data.datasets import LoggedBanditDataset, TrajectoryDataset


def test_logged_bandit_validation_passes():
    contexts = np.zeros((5, 2))
    actions = np.array([0, 1, 0, 1, 0])
    rewards = np.ones(5)
    behavior_action_probs = np.full(5, 0.5)
    dataset = LoggedBanditDataset(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        behavior_action_probs=behavior_action_probs,
        action_space_n=2,
    )
    assert dataset.num_samples == 5


def test_logged_bandit_validation_fails_on_action_range():
    contexts = np.zeros((3, 1))
    actions = np.array([0, 2, 1])
    rewards = np.ones(3)
    behavior_action_probs = np.full(3, 0.5)
    with pytest.raises(ValueError):
        LoggedBanditDataset(
            contexts=contexts,
            actions=actions,
            rewards=rewards,
            behavior_action_probs=behavior_action_probs,
            action_space_n=2,
        )


def test_trajectory_validation_passes():
    obs = np.zeros((2, 3))
    actions = np.zeros((2, 3), dtype=int)
    rewards = np.ones((2, 3))
    next_obs = np.zeros((2, 3))
    behavior_action_probs = np.full((2, 3), 0.5)
    mask = np.ones((2, 3), dtype=bool)
    dataset = TrajectoryDataset(
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        behavior_action_probs=behavior_action_probs,
        mask=mask,
        discount=0.9,
        action_space_n=2,
    )
    assert dataset.num_trajectories == 2


def test_trajectory_validation_fails_on_shape():
    obs = np.zeros((2, 3))
    actions = np.zeros((2, 3), dtype=int)
    rewards = np.ones((2, 3))
    next_obs = np.zeros((2, 2))
    behavior_action_probs = np.full((2, 3), 0.5)
    mask = np.ones((2, 3), dtype=bool)
    with pytest.raises(ValueError):
        TrajectoryDataset(
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            behavior_action_probs=behavior_action_probs,
            mask=mask,
            discount=0.9,
            action_space_n=2,
        )
