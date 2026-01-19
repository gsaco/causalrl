"""Adapter for D4RL datasets."""

from __future__ import annotations

import numpy as np

from crl.data.trajectory import TrajectoryDataset
from crl.data.transition import TransitionDataset


def load_d4rl_dataset(
    env_name: str,
    *,
    discount: float = 0.99,
    include_timeouts: bool = True,
    return_type: str = "transition",
) -> TransitionDataset | TrajectoryDataset:
    """Load a D4RL dataset and map it to a TransitionDataset."""

    try:
        import d4rl  # noqa: F401
        import gym
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "d4rl and gym are required for D4RL dataset loading."
        ) from exc

    env = gym.make(env_name)
    data = env.get_dataset()

    observations = np.asarray(data["observations"])
    actions = np.asarray(data["actions"])
    rewards = np.asarray(data["rewards"]).reshape(-1)
    next_observations = np.asarray(data["next_observations"])
    terminals = np.asarray(data["terminals"]).astype(bool).reshape(-1)
    timeouts = np.asarray(data.get("timeouts", np.zeros_like(terminals))).astype(bool)

    dones = terminals | timeouts if include_timeouts else terminals
    episode_ids, timesteps = _infer_episode_ids(dones)

    action_space_n = None
    if hasattr(env.action_space, "n"):
        action_space_n = int(env.action_space.n)

    dataset = TransitionDataset(
        states=observations,
        actions=actions,
        rewards=rewards,
        next_states=next_observations,
        dones=dones,
        behavior_action_probs=None,
        discount=discount,
        action_space_n=action_space_n,
        episode_ids=episode_ids,
        timesteps=timesteps,
        metadata={
            "source": "d4rl",
            "env_name": env_name,
            "include_timeouts": include_timeouts,
        },
    )

    if return_type == "trajectory":
        return dataset.to_trajectory()
    if return_type == "transition":
        return dataset
    raise ValueError(f"Unknown return_type: {return_type}")


def _infer_episode_ids(dones: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dones = np.asarray(dones, dtype=bool)
    episode_ids = np.zeros_like(dones, dtype=int)
    timesteps = np.zeros_like(dones, dtype=int)
    current_ep = 0
    current_t = 0
    for idx, done in enumerate(dones):
        episode_ids[idx] = current_ep
        timesteps[idx] = current_t
        if done:
            current_ep += 1
            current_t = 0
        else:
            current_t += 1
    return episode_ids, timesteps


__all__ = ["load_d4rl_dataset"]
