"""Policy interfaces."""

from __future__ import annotations

import numpy as np


class Policy:
    """Abstract policy interface for discrete or continuous action spaces.

    Estimand:
        Not applicable.
    Assumptions:
        None.
    Inputs:
        observations: Array with shape (n, d) or (n,) representing states.
    Outputs:
        action_probs: Array with shape (n, a) for discrete policies.
        action_density: Array with shape (n,) for continuous policies.
    Failure modes:
        Implementations should raise ValueError if probabilities are invalid.
    """

    def action_probs(self, observations: np.ndarray) -> np.ndarray:
        """Return action probabilities for each observation (discrete)."""

        raise NotImplementedError("action_probs is not implemented for this policy.")

    def action_prob(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return probabilities for selected actions (discrete)."""

        probs = self.action_probs(observations)
        actions = np.asarray(actions).reshape(-1)
        return probs[np.arange(probs.shape[0]), actions]

    def action_density(
        self, observations: np.ndarray, actions: np.ndarray
    ) -> np.ndarray:
        """Return action densities for selected actions (continuous)."""

        raise NotImplementedError("action_density is not implemented for this policy.")

    def log_prob(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return log-probability or log-density for selected actions."""

        try:
            probs = self.action_prob(observations, actions)
        except NotImplementedError:
            probs = self.action_density(observations, actions)
        probs = np.asarray(probs, dtype=float)
        if np.any(probs <= 0.0):
            raise ValueError("Probabilities/densities must be positive for log_prob.")
        return np.log(probs)

    def sample_action(
        self, observations: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample actions for observations (optional)."""

        raise NotImplementedError("sample_action is not implemented for this policy.")

    @classmethod
    def from_sklearn(
        cls,
        model: object,
        action_space_n: int,
        *,
        deterministic: bool | None = None,
        name: str | None = None,
    ) -> "Policy":
        """Wrap a scikit-learn model as a policy.

        If deterministic is True, uses model.predict and treats outputs as actions.
        Otherwise uses model.predict_proba for action probabilities.
        """

        if deterministic is None:
            deterministic = not hasattr(model, "predict_proba")
        if deterministic:
            if not hasattr(model, "predict"):
                raise ValueError("model must implement predict for deterministic use.")

            def action_fn(obs: np.ndarray) -> np.ndarray:
                return np.asarray(model.predict(obs))

            from crl.policies.discrete import CallablePolicy

            return CallablePolicy(
                action_fn=action_fn,
                action_space_n=action_space_n,
                returns="actions",
                name=name or type(model).__name__,
            )

        if not hasattr(model, "predict_proba"):
            raise ValueError("model must implement predict_proba for stochastic use.")

        def prob_fn(obs: np.ndarray) -> np.ndarray:
            return np.asarray(model.predict_proba(obs))

        from crl.policies.discrete import StochasticPolicy

        return StochasticPolicy(
            prob_fn=prob_fn,
            action_space_n=action_space_n,
            name=name or type(model).__name__,
        )

    @classmethod
    def from_torch(
        cls,
        model: object,
        action_space_n: int,
        *,
        device: str = "cpu",
        name: str | None = None,
    ) -> "Policy":
        """Wrap a torch model that outputs action logits."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("torch is required for from_torch().") from exc

        torch_model = model
        torch_model.to(torch.device(device))
        torch_model.eval()

        def prob_fn(obs: np.ndarray) -> np.ndarray:
            obs_arr = np.asarray(obs, dtype=np.float32)
            if obs_arr.ndim == 1:
                obs_arr = obs_arr.reshape(-1, 1)
            with torch.no_grad():
                logits = torch_model(torch.from_numpy(obs_arr).to(device))
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs

        from crl.policies.discrete import StochasticPolicy

        return StochasticPolicy(
            prob_fn=prob_fn,
            action_space_n=action_space_n,
            name=name or type(model).__name__,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a dictionary representation."""

        return {"policy_type": type(self).__name__}

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
