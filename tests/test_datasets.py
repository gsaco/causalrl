import numpy as np
import pytest

from crl.data.datasets import LoggedBanditDataset, TrajectoryDataset
from crl.data.transition import TransitionDataset


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


def test_logged_bandit_validation_fails_on_nan():
    contexts = np.zeros((2, 1))
    contexts[0, 0] = np.nan
    actions = np.array([0, 1])
    rewards = np.ones(2)
    behavior_action_probs = np.full(2, 0.5)
    with pytest.raises(ValueError):
        LoggedBanditDataset(
            contexts=contexts,
            actions=actions,
            rewards=rewards,
            behavior_action_probs=behavior_action_probs,
            action_space_n=2,
        )


def test_logged_bandit_describe():
    contexts = np.zeros((3, 2))
    actions = np.array([0, 1, 0])
    rewards = np.array([1.0, 0.5, 0.0])
    behavior_action_probs = np.full(3, 0.5)
    dataset = LoggedBanditDataset(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        behavior_action_probs=behavior_action_probs,
        action_space_n=2,
    )
    summary = dataset.describe()
    assert summary["num_samples"] == 3
    assert summary["context_dim"] == 2
    assert summary["behavior_action_probs_present"] is True


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


def test_trajectory_validation_fails_on_noncontiguous_mask():
    obs = np.zeros((1, 3))
    actions = np.zeros((1, 3), dtype=int)
    rewards = np.ones((1, 3))
    next_obs = np.zeros((1, 3))
    behavior_action_probs = np.full((1, 3), 0.5)
    mask = np.array([[True, False, True]])
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


def test_trajectory_validation_fails_on_empty_trajectory():
    obs = np.zeros((1, 2))
    actions = np.zeros((1, 2), dtype=int)
    rewards = np.ones((1, 2))
    next_obs = np.zeros((1, 2))
    behavior_action_probs = np.full((1, 2), 0.5)
    mask = np.zeros((1, 2), dtype=bool)
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


def test_trajectory_describe():
    obs = np.zeros((2, 3))
    actions = np.zeros((2, 3), dtype=int)
    rewards = np.ones((2, 3))
    next_obs = np.zeros((2, 3))
    behavior_action_probs = np.full((2, 3), 0.5)
    mask = np.array([[True, True, False], [True, True, True]])
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
    summary = dataset.describe()
    assert summary["num_trajectories"] == 2
    assert summary["horizon"] == 3
    assert summary["behavior_action_probs_present"] is True


def test_transition_validation_passes():
    states = np.zeros((4, 2))
    actions = np.array([0, 1, 0, 1])
    rewards = np.ones(4)
    next_states = np.zeros((4, 2))
    dones = np.array([False, False, True, True])
    dataset = TransitionDataset(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        behavior_action_probs=np.full(4, 0.5),
        discount=0.9,
        action_space_n=2,
        episode_ids=np.array([0, 0, 1, 1]),
        timesteps=np.array([0, 1, 0, 1]),
    )
    assert dataset.num_steps == 4


def test_transition_validation_fails_on_action_range():
    states = np.zeros((3, 1))
    actions = np.array([0, 2, 1])
    rewards = np.ones(3)
    next_states = np.zeros((3, 1))
    dones = np.array([False, False, True])
    with pytest.raises(ValueError):
        TransitionDataset(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            behavior_action_probs=np.full(3, 0.5),
            discount=0.9,
            action_space_n=2,
        )


def test_transition_to_trajectory():
    states = np.array([0, 1, 0, 1])
    actions = np.array([0, 1, 0, 1])
    rewards = np.ones(4)
    next_states = np.array([1, 0, 1, 0])
    dones = np.array([False, True, False, True])
    dataset = TransitionDataset(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
        behavior_action_probs=np.full(4, 0.5),
        discount=0.9,
        action_space_n=2,
        episode_ids=np.array([0, 0, 1, 1]),
        timesteps=np.array([0, 1, 0, 1]),
    )
    traj = dataset.to_trajectory()
    assert traj.num_trajectories == 2


def test_logged_bandit_from_numpy():
    contexts = np.zeros((3, 2))
    actions = np.array([0, 1, 0])
    rewards = np.array([1.0, 0.5, 0.25])
    behavior_action_probs = np.array([0.6, 0.4, 0.6])
    dataset = LoggedBanditDataset.from_numpy(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        behavior_action_probs=behavior_action_probs,
    )
    assert dataset.action_space_n == 2
    assert dataset.num_samples == 3


def test_logged_bandit_from_dataframe():
    import pandas as pd

    df = pd.DataFrame(
        {
            "x0": [0.1, 0.2, 0.3],
            "x1": [1.0, 0.0, 1.0],
            "action": [0, 1, 0],
            "reward": [1.0, 0.5, 0.25],
            "propensity": [0.6, 0.4, 0.6],
        }
    )
    dataset = LoggedBanditDataset.from_dataframe(
        df,
        context_columns=["x0", "x1"],
        behavior_prob_column="propensity",
    )
    assert dataset.contexts.shape == (3, 2)
    assert dataset.behavior_action_probs is not None


def test_trajectory_from_numpy():
    obs = np.zeros((2, 3, 1))
    next_obs = np.ones((2, 3, 1))
    actions = np.array([[0, 1, 0], [1, 0, 1]])
    rewards = np.ones((2, 3))
    dataset = TrajectoryDataset.from_numpy(
        observations=obs,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        discount=0.9,
        action_space_n=2,
    )
    assert dataset.num_trajectories == 2
    assert dataset.horizon == 3


def test_trajectory_from_dataframe():
    import pandas as pd

    df = pd.DataFrame(
        {
            "episode_id": [0, 0, 1],
            "timestep": [0, 1, 0],
            "obs": [0.0, 0.1, 0.2],
            "next_obs": [0.1, 0.2, 0.3],
            "action": [0, 1, 0],
            "reward": [1.0, 0.5, 0.25],
            "propensity": [0.6, 0.4, 0.6],
        }
    )
    dataset = TrajectoryDataset.from_dataframe(
        df,
        observation_columns=["obs"],
        next_observation_columns=["next_obs"],
        behavior_prob_column="propensity",
        discount=0.9,
        action_space_n=2,
    )
    assert dataset.num_trajectories == 2
    assert dataset.behavior_action_probs is not None
