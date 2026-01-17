"""Torch-backed MLP policy for continuous or discrete observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crl.policies.base import Policy

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - handled in runtime checks
    raise ImportError("TorchMLPPolicy requires torch to be installed.") from exc


@dataclass
class MLPConfig:
    """Configuration for MLP construction.

    Estimand:
        Not applicable.
    Assumptions:
        None.
    Inputs:
        input_dim: Observation dimension.
        action_dim: Number of actions.
        hidden_sizes: Hidden layer sizes.
        activation: Activation name (relu or tanh).
    Outputs:
        Configuration object.
    Failure modes:
        None.
    """

    input_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "relu"


class TorchMLPPolicy(Policy):
    """MLP policy implemented in PyTorch.

    Estimand:
        Not applicable.
    Assumptions:
        Observations are numeric features and actions are discrete.
    Inputs:
        model: torch.nn.Module mapping observations to logits.
        action_dim: Number of discrete actions.
        device: Torch device string.
    Outputs:
        action_probs: Array with shape (n, action_dim).
    Failure modes:
        Raises ValueError if logits have the wrong shape.
    """

    def __init__(self, model: nn.Module, action_dim: int, device: str = "cpu") -> None:
        self.model = model
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_config(cls, config: MLPConfig, device: str = "cpu") -> "TorchMLPPolicy":
        """Construct a policy from an MLPConfig."""

        layers: list[nn.Module] = []
        input_dim = config.input_dim
        for hidden in config.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU() if config.activation == "relu" else nn.Tanh())
            input_dim = hidden
        layers.append(nn.Linear(input_dim, config.action_dim))
        model = nn.Sequential(*layers)
        return cls(model=model, action_dim=config.action_dim, device=device)

    def action_probs(self, observations: np.ndarray) -> np.ndarray:
        """Return action probabilities for each observation."""

        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(obs).to(self.device))
            if logits.shape[-1] != self.action_dim:
                raise ValueError("Model output dimension does not match action_dim.")
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def sample_action(self, observations: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample actions for observations."""

        probs = self.action_probs(observations)
        return np.array([rng.choice(self.action_dim, p=p) for p in probs])
