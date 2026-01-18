"""Behavior policy estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from crl.behavior.diagnostics import behavior_diagnostics
from crl.data.bandit import BanditDataset
from crl.data.trajectory import TrajectoryDataset
from crl.data.transition import TransitionDataset


@dataclass
class BehaviorPolicyFit:
    """Container for estimated behavior propensities and diagnostics."""

    propensities: np.ndarray
    model: Any
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def apply(
        self,
        dataset: BanditDataset | TrajectoryDataset | TransitionDataset,
    ) -> BanditDataset | TrajectoryDataset | TransitionDataset:
        """Return a dataset with estimated propensities attached."""

        meta = dict(getattr(dataset, "metadata", None) or {})
        meta.update(
            {
                "behavior_policy_source": "estimated",
                "behavior_policy_method": self.metadata.get("method"),
                "behavior_policy_diagnostics": self.diagnostics,
            }
        )

        if isinstance(dataset, BanditDataset):
            payload = dataset.to_dict()
            payload["behavior_action_probs"] = self.propensities
            payload["metadata"] = meta
            return dataset.__class__.from_dict(payload)

        if isinstance(dataset, TrajectoryDataset):
            payload = dataset.to_dict()
            payload["behavior_action_probs"] = self.propensities
            payload["metadata"] = meta
            return dataset.__class__.from_dict(payload)

        if isinstance(dataset, TransitionDataset):
            return TransitionDataset(
                states=dataset.states,
                actions=dataset.actions,
                rewards=dataset.rewards,
                next_states=dataset.next_states,
                dones=dataset.dones,
                behavior_action_probs=self.propensities,
                discount=dataset.discount,
                action_space_n=dataset.action_space_n,
                episode_ids=dataset.episode_ids,
                timesteps=dataset.timesteps,
                metadata=meta,
            )

        raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")


def fit_behavior_policy(
    dataset: BanditDataset | TrajectoryDataset | TransitionDataset,
    *,
    method: str = "logit",
    model: Any | None = None,
    model_kwargs: dict[str, Any] | None = None,
    clip_min: float = 1e-3,
    seed: int = 0,
    store_action_probs: bool = False,
) -> BehaviorPolicyFit:
    """Estimate behavior policy propensities from logged data."""

    model_kwargs = dict(model_kwargs or {})
    obs, actions, mask = _extract_features(dataset)
    num_actions = _infer_num_actions(dataset, actions)

    if model is None:
        model = _build_model(method, num_actions, seed, model_kwargs)

    model.fit(obs, actions)
    action_probs = model.predict_proba(obs)

    if action_probs.shape[1] != num_actions:
        raise ValueError("Estimated action_probs have unexpected action dimension.")

    propensities = action_probs[np.arange(actions.size), actions]
    propensities = np.clip(propensities, clip_min, 1.0)

    diagnostics = behavior_diagnostics(
        action_probs, actions, propensities, clip_min=clip_min
    )

    if isinstance(dataset, TrajectoryDataset) and mask is not None:
        full = np.ones_like(dataset.actions, dtype=float)
        full[mask] = propensities
        prop_out = full
    else:
        prop_out = propensities

    metadata: dict[str, Any] = {
        "method": method,
        "clip_min": clip_min,
        "num_actions": num_actions,
        "seed": seed,
        "model_type": type(model).__name__,
    }
    if store_action_probs:
        metadata["action_probs"] = action_probs

    return BehaviorPolicyFit(
        propensities=prop_out,
        model=model,
        diagnostics=diagnostics,
        metadata=metadata,
    )


def _extract_features(
    dataset: BanditDataset | TrajectoryDataset | TransitionDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if isinstance(dataset, BanditDataset):
        obs = np.asarray(dataset.contexts)
        actions = np.asarray(dataset.actions).reshape(-1)
        return _ensure_2d(obs), actions, None

    if isinstance(dataset, TrajectoryDataset):
        obs = dataset.observations[dataset.mask]
        actions = dataset.actions[dataset.mask].reshape(-1)
        return _ensure_2d(obs), actions, dataset.mask

    if isinstance(dataset, TransitionDataset):
        obs = np.asarray(dataset.states)
        actions = np.asarray(dataset.actions).reshape(-1)
        return _ensure_2d(obs), actions, None

    raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")


def _ensure_2d(observations: np.ndarray) -> np.ndarray:
    obs = np.asarray(observations)
    if obs.ndim == 1:
        return obs.reshape(-1, 1)
    if obs.ndim == 2:
        return obs
    return obs.reshape(obs.shape[0], -1)


def _infer_num_actions(
    dataset: BanditDataset | TrajectoryDataset | TransitionDataset,
    actions: np.ndarray,
) -> int:
    if isinstance(dataset, (BanditDataset, TrajectoryDataset)):
        return int(dataset.action_space_n)
    if dataset.action_space_n is None:
        raise ValueError("action_space_n is required for behavior estimation.")
    return int(dataset.action_space_n)


def _build_model(
    method: str, num_actions: int, seed: int, model_kwargs: dict[str, Any]
) -> Any:
    if method == "logit":
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "scikit-learn is required for behavior policy estimation."
            ) from exc
        default = {
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": seed,
        }
        default.update(model_kwargs)
        return LogisticRegression(**default)

    if method == "gboost":
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "scikit-learn is required for behavior policy estimation."
            ) from exc
        default = {"random_state": seed}
        default.update(model_kwargs)
        return GradientBoostingClassifier(**default)

    raise ValueError(f"Unknown behavior estimation method: {method}")
