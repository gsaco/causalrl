"""Bootstrap utilities for estimator uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from crl.data.datasets import LoggedBanditDataset, TrajectoryDataset
from crl.estimators.base import OPEEstimator


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap confidence intervals."""

    num_bootstrap: int = 200
    method: str = "trajectory"
    block_size: int = 5
    alpha: float = 0.05
    seed: int = 0


def bootstrap_ci(
    estimator_factory: Callable[[], OPEEstimator],
    data: LoggedBanditDataset | TrajectoryDataset,
    config: BootstrapConfig,
) -> tuple[float, tuple[float, float]]:
    """Compute bootstrap standard error and confidence interval."""

    rng = np.random.default_rng(config.seed)
    estimates = np.zeros(config.num_bootstrap, dtype=float)
    for i in range(config.num_bootstrap):
        sample = _resample_dataset(data, rng, config)
        estimator = estimator_factory()
        estimates[i] = estimator.estimate(sample).value

    stderr = float(np.std(estimates, ddof=1))
    lower = float(np.quantile(estimates, config.alpha / 2))
    upper = float(np.quantile(estimates, 1 - config.alpha / 2))
    return stderr, (lower, upper)


def _resample_dataset(
    data: LoggedBanditDataset | TrajectoryDataset,
    rng: np.random.Generator,
    config: BootstrapConfig,
) -> LoggedBanditDataset | TrajectoryDataset:
    if isinstance(data, LoggedBanditDataset):
        idx = rng.integers(0, data.num_samples, size=data.num_samples)
        return LoggedBanditDataset(
            contexts=data.contexts[idx],
            actions=data.actions[idx],
            rewards=data.rewards[idx],
            behavior_action_probs=(
                data.behavior_action_probs[idx] if data.behavior_action_probs is not None else None
            ),
            action_space_n=data.action_space_n,
            metadata={"bootstrap": True},
        )

    if config.method == "trajectory":
        idx = rng.integers(0, data.num_trajectories, size=data.num_trajectories)
        return TrajectoryDataset(
            observations=data.observations[idx],
            actions=data.actions[idx],
            rewards=data.rewards[idx],
            next_observations=data.next_observations[idx],
            behavior_action_probs=(
                data.behavior_action_probs[idx] if data.behavior_action_probs is not None else None
            ),
            mask=data.mask[idx],
            discount=data.discount,
            action_space_n=data.action_space_n,
            state_space_n=data.state_space_n,
            metadata={"bootstrap": True},
        )

    if config.method == "iid":
        flat_obs = data.observations[data.mask]
        flat_actions = data.actions[data.mask]
        flat_rewards = data.rewards[data.mask]
        flat_next_obs = data.next_observations[data.mask]
        flat_probs = (
            data.behavior_action_probs[data.mask]
            if data.behavior_action_probs is not None
            else None
        )
        total_steps = data.num_trajectories * data.horizon
        idx = rng.integers(0, flat_obs.shape[0], size=total_steps)
        obs = flat_obs[idx].reshape(data.num_trajectories, data.horizon, *flat_obs.shape[1:])
        next_obs = flat_next_obs[idx].reshape(data.num_trajectories, data.horizon, *flat_next_obs.shape[1:])
        actions = flat_actions[idx].reshape(data.num_trajectories, data.horizon)
        rewards = flat_rewards[idx].reshape(data.num_trajectories, data.horizon)
        mask = np.ones_like(actions, dtype=bool)
        behavior_action_probs = (
            flat_probs[idx].reshape(data.num_trajectories, data.horizon) if flat_probs is not None else None
        )
        return TrajectoryDataset(
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            behavior_action_probs=behavior_action_probs,
            mask=mask,
            discount=data.discount,
            action_space_n=data.action_space_n,
            state_space_n=data.state_space_n,
            metadata={"bootstrap": True},
        )

    if config.method == "block":
        obs = np.zeros_like(data.observations)
        next_obs = np.zeros_like(data.next_observations)
        actions = np.zeros_like(data.actions)
        rewards = np.zeros_like(data.rewards)
        mask = np.ones_like(data.mask, dtype=bool)
        behavior_action_probs = (
            np.zeros_like(data.behavior_action_probs)
            if data.behavior_action_probs is not None
            else None
        )
        for i in range(data.num_trajectories):
            pos = 0
            while pos < data.horizon:
                traj_idx = rng.integers(0, data.num_trajectories)
                start = rng.integers(0, max(1, data.horizon - config.block_size + 1))
                end = min(start + config.block_size, data.horizon)
                length = end - start
                obs[i, pos : pos + length] = data.observations[traj_idx, start:end]
                next_obs[i, pos : pos + length] = data.next_observations[traj_idx, start:end]
                actions[i, pos : pos + length] = data.actions[traj_idx, start:end]
                rewards[i, pos : pos + length] = data.rewards[traj_idx, start:end]
                if behavior_action_probs is not None:
                    behavior_action_probs[i, pos : pos + length] = data.behavior_action_probs[
                        traj_idx, start:end
                    ]
                pos += length
        return TrajectoryDataset(
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            behavior_action_probs=behavior_action_probs,
            mask=mask,
            discount=data.discount,
            action_space_n=data.action_space_n,
            state_space_n=data.state_space_n,
            metadata={"bootstrap": True},
        )

    raise ValueError(f"Unknown bootstrap method: {config.method}")
