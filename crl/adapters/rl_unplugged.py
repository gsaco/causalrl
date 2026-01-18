"""Adapter for RL Unplugged datasets via TFDS (RLDS format)."""

from __future__ import annotations

from typing import Any

import numpy as np

from crl.data.transition import TransitionDataset


def load_rl_unplugged_dataset(
    dataset_name: str,
    *,
    split: str = "train",
    discount: float = 0.99,
    return_type: str = "transition",
    max_episodes: int | None = None,
) -> TransitionDataset:
    """Load an RL Unplugged dataset and map it to a TransitionDataset."""

    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "tensorflow-datasets is required for RL Unplugged dataset loading."
        ) from exc

    ds = tfds.as_numpy(tfds.load(dataset_name, split=split))

    states: list[Any] = []
    actions: list[Any] = []
    rewards: list[float] = []
    next_states: list[Any] = []
    dones: list[bool] = []
    episode_ids: list[int] = []
    timesteps: list[int] = []

    episode_count = 0
    for episode in ds:
        if max_episodes is not None and episode_count >= max_episodes:
            break
        steps = episode.get("steps", None)
        if steps is None:
            raise ValueError(
                "Expected RLDS episodes with a 'steps' field for RL Unplugged."
            )
        obs = np.asarray(steps["observation"])
        act = np.asarray(steps["action"])
        rew = np.asarray(steps["reward"]).reshape(-1)
        is_last = steps.get("is_last", steps.get("is_terminal", None))
        if is_last is None:
            is_last = np.zeros_like(rew, dtype=bool)
            if rew.size:
                is_last[-1] = True
        else:
            is_last = np.asarray(is_last).astype(bool)

        for t in range(rew.size):
            states.append(obs[t])
            actions.append(act[t])
            rewards.append(float(rew[t]))
            next_states.append(obs[t + 1] if t + 1 < obs.shape[0] else obs[t])
            dones.append(bool(is_last[t]))
            episode_ids.append(episode_count)
            timesteps.append(t)

        episode_count += 1

    dataset = TransitionDataset(
        states=np.asarray(states),
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
        next_states=np.asarray(next_states),
        dones=np.asarray(dones, dtype=bool),
        behavior_action_probs=None,
        discount=discount,
        action_space_n=None,
        episode_ids=np.asarray(episode_ids, dtype=int),
        timesteps=np.asarray(timesteps, dtype=int),
        metadata={
            "source": "rl_unplugged",
            "dataset_name": dataset_name,
            "split": split,
        },
    )

    if return_type == "trajectory":
        return dataset.to_trajectory()
    if return_type == "transition":
        return dataset
    raise ValueError(f"Unknown return_type: {return_type}")


__all__ = ["load_rl_unplugged_dataset"]
