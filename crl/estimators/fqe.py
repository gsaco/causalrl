"""Fitted Q Evaluation (FQE) estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from crl.data.datasets import TrajectoryDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.base import (
    DiagnosticsConfig,
    EstimatorReport,
    OPEEstimator,
    compute_ci,
)
from crl.estimators.bootstrap import BootstrapConfig
from crl.estimators.diagnostics import run_diagnostics
from crl.estimators.stats import mean_stderr
from crl.estimators.utils import compute_action_probs
from crl.policies.tabular import TabularPolicy
from crl.utils.seeding import set_seed

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("FQE estimator requires torch to be installed.") from exc


@dataclass
class FQEConfig:
    """Configuration for FQE training.

    Estimand:
        Not applicable.
    Assumptions:
        None.
    Inputs:
        hidden_sizes: Hidden layer sizes for the Q network.
        learning_rate: Optimizer learning rate.
        batch_size: Mini-batch size.
        num_epochs: Epochs per iteration.
        num_iterations: Number of fitted Q iterations.
        weight_decay: L2 penalty.
        seed: RNG seed for torch and numpy.
    Outputs:
        Configuration object.
    Failure modes:
        None.
    """

    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 1e-3
    batch_size: int = 128
    num_epochs: int = 20
    num_iterations: int = 10
    weight_decay: float = 0.0
    seed: int = 0
    bootstrap: bool = False
    bootstrap_config: BootstrapConfig | None = None
    uncertainty: str | None = None
    n_bootstrap: int = 200


class TorchQNetwork(nn.Module):
    """Simple MLP for Q-function approximation."""

    def __init__(self, input_dim: int, hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.ReLU())
            dim = hidden
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FQEEstimator(OPEEstimator):
    """Fitted Q Evaluation estimator for finite-horizon MDPs.

    Estimand:
        PolicyValueEstimand for the target policy.
    Assumptions:
        Sequential ignorability, overlap, and correct function approximation.
    Inputs:
        TrajectoryDataset (n, t).
    Outputs:
        EstimatorReport with value and diagnostics.
    Failure modes:
        Extrapolation error for out-of-distribution actions.
    """

    required_assumptions = ["sequential_ignorability", "overlap", "markov"]
    required_fields: list[str] = []
    diagnostics_keys = ["overlap", "ess", "weights", "max_weight", "model"]

    def __init__(
        self,
        estimand: PolicyValueEstimand,
        run_diagnostics: bool = True,
        diagnostics_config: DiagnosticsConfig | None = None,
        config: FQEConfig | None = None,
        device: str = "cpu",
        bootstrap: bool = False,
        bootstrap_config: Any | None = None,
    ) -> None:
        super().__init__(
            estimand,
            run_diagnostics,
            diagnostics_config,
            bootstrap,
            bootstrap_config,
        )
        self.config = config or FQEConfig()
        self.device = torch.device(device)
        if self.config.bootstrap or self.config.uncertainty == "bootstrap":
            self.bootstrap = True
            if self.config.bootstrap_config is not None:
                self.bootstrap_config = self.config.bootstrap_config
            else:
                self.bootstrap_config = BootstrapConfig(
                    num_bootstrap=self.config.n_bootstrap
                )
        self._bootstrap_params.update(
            {
                "config": self.config,
                "device": str(self.device),
                "bootstrap": False,
                "bootstrap_config": None,
            }
        )

    def estimate(self, data: TrajectoryDataset) -> EstimatorReport:
        """Estimate policy value via FQE."""

        self._validate_dataset(data)
        if self._can_use_tabular(data):
            values = self._tabular_values(data)
            value = float(np.mean(values))
            stderr = mean_stderr(values)
            q_network = None
            model_metrics: dict[str, Any] = {}
        else:
            q_network, train_mse = self._fit_q_function(data)
            initial_states = data.observations[:, 0]
            values = self._compute_state_values(
                q_network, initial_states, data.state_space_n
            )
            value = float(np.mean(values))
            stderr = mean_stderr(values)
            model_metrics = {"q_model_mse": train_mse}

        ci = compute_ci(value, stderr)

        diagnostics: dict[str, Any] = {}
        warnings: list[str] = []
        if self.run_diagnostics and data.behavior_action_probs is not None:
            target_probs = compute_action_probs(
                self.estimand.policy, data.observations, data.actions
            )
            ratios = np.where(data.mask, target_probs / data.behavior_action_probs, 1.0)
            weights = np.prod(ratios, axis=1)
            diagnostics, warnings = run_diagnostics(
                weights,
                target_probs,
                data.behavior_action_probs,
                data.mask,
                self.diagnostics_config,
            )
            diagnostics["model"] = model_metrics if q_network is not None else {}
        elif self.run_diagnostics:
            warnings.append(
                "behavior_action_probs missing; skipping weight diagnostics."
            )

        return self._build_report(
            value=value,
            stderr=stderr,
            ci=ci,
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={
                "estimator": "FQE",
                "config": self.config.__dict__,
                "tabular": q_network is None,
            },
            data=data,
        )

    def _bootstrap_factory(self):
        config = FQEConfig(
            hidden_sizes=self.config.hidden_sizes,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            num_epochs=self.config.num_epochs,
            num_iterations=self.config.num_iterations,
            weight_decay=self.config.weight_decay,
            seed=self.config.seed,
            bootstrap=False,
            bootstrap_config=None,
            uncertainty=None,
            n_bootstrap=self.config.n_bootstrap,
        )

        def _factory():
            return FQEEstimator(
                estimand=self.estimand,
                run_diagnostics=False,
                diagnostics_config=self.diagnostics_config,
                config=config,
                device=str(self.device),
            )

        return _factory

    def _can_use_tabular(self, data: TrajectoryDataset) -> bool:
        if data.state_space_n is None:
            return False
        if not np.issubdtype(data.observations.dtype, np.integer):
            return False
        if not np.issubdtype(data.next_observations.dtype, np.integer):
            return False
        if not hasattr(self.estimand.policy, "table"):
            return False
        return True

    def _tabular_values(self, data: TrajectoryDataset) -> np.ndarray:
        policy = cast(TabularPolicy, self.estimand.policy)
        states = data.state_space_n or 0
        actions = data.action_space_n
        counts = np.zeros((states, actions), dtype=float)
        reward_sums = np.zeros((states, actions), dtype=float)
        transition_counts = np.zeros((states, actions, states), dtype=float)

        obs = data.observations[data.mask]
        next_obs = data.next_observations[data.mask]
        acts = data.actions[data.mask]
        rews = data.rewards[data.mask]

        for s, a, r, s_next in zip(obs, acts, rews, next_obs, strict=True):
            s_idx = int(s)
            a_idx = int(a)
            s_next_idx = int(s_next)
            counts[s_idx, a_idx] += 1.0
            reward_sums[s_idx, a_idx] += float(r)
            transition_counts[s_idx, a_idx, s_next_idx] += 1.0

        counts = np.maximum(counts, 1.0)
        reward_means = reward_sums / counts
        transition_probs = transition_counts / counts[:, :, None]

        q = np.zeros((states, actions), dtype=float)
        for _ in range(self.config.num_iterations):
            v = (policy.table * q).sum(axis=1)
            q = reward_means + data.discount * np.einsum(
                "sak,k->sa", transition_probs, v
            )

        initial_states = data.observations[:, 0].reshape(-1).astype(int)
        v0 = (policy.table * q).sum(axis=1)
        return v0[initial_states]

    def _fit_q_function(self, data: TrajectoryDataset) -> tuple[TorchQNetwork, float]:
        set_seed(self.config.seed)

        obs, next_obs, actions, rewards, next_policy_probs = self._flatten_transitions(
            data
        )
        obs_features = self._features(obs, data.state_space_n)
        next_features = self._features(next_obs, data.state_space_n)

        action_one_hot = self._one_hot(actions, data.action_space_n)
        obs_action = np.concatenate([obs_features, action_one_hot], axis=1)

        input_dim = obs_action.shape[1]
        q_network = TorchQNetwork(input_dim, self.config.hidden_sizes).to(self.device)
        optimizer = torch.optim.Adam(
            q_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.MSELoss()

        dataset = TensorDataset(
            torch.tensor(obs_action, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_features, dtype=torch.float32),
            torch.tensor(next_policy_probs, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        for _ in range(self.config.num_iterations):
            for _ in range(self.config.num_epochs):
                for (
                    batch_obs_action,
                    batch_rewards,
                    batch_next,
                    batch_policy_probs,
                ) in loader:
                    batch_obs_action = batch_obs_action.to(self.device)
                    batch_rewards = batch_rewards.to(self.device)
                    batch_next = batch_next.to(self.device)
                    batch_policy_probs = batch_policy_probs.to(self.device)

                    with torch.no_grad():
                        next_values = self._torch_policy_value(
                            q_network,
                            batch_next,
                            batch_policy_probs,
                            data.action_space_n,
                        )
                        targets = batch_rewards + data.discount * next_values

                    predictions = q_network(batch_obs_action)
                    loss = loss_fn(predictions, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        with torch.no_grad():
            obs_action_tensor = torch.tensor(obs_action, dtype=torch.float32).to(
                self.device
            )
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            next_tensor = torch.tensor(next_features, dtype=torch.float32).to(
                self.device
            )
            policy_tensor = torch.tensor(next_policy_probs, dtype=torch.float32).to(
                self.device
            )
            next_values = self._torch_policy_value(
                q_network, next_tensor, policy_tensor, data.action_space_n
            )
            targets = rewards_tensor + data.discount * next_values
            preds = q_network(obs_action_tensor)
            train_mse = float(torch.mean((preds - targets) ** 2).cpu().item())

        return q_network, train_mse

    def _flatten_transitions(
        self, data: TrajectoryDataset
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mask = data.mask
        obs = data.observations[mask]
        next_obs = data.next_observations[mask]
        actions = data.actions[mask]
        rewards = data.rewards[mask]
        next_policy_probs = self.estimand.policy.action_probs(next_obs)
        return obs, next_obs, actions, rewards, next_policy_probs

    def _features(
        self, observations: np.ndarray, state_space_n: int | None
    ) -> np.ndarray:
        obs = np.asarray(observations)
        if obs.ndim == 1:
            if state_space_n is not None:
                return self._one_hot(obs.astype(int), state_space_n)
            return obs.reshape(-1, 1).astype(float)
        if obs.ndim == 2:
            if obs.shape[1] == 1 and state_space_n is not None:
                return self._one_hot(obs.reshape(-1).astype(int), state_space_n)
            return obs.astype(float)
        raise ValueError("Observations must be 1D or 2D after flattening.")

    @staticmethod
    def _one_hot(values: np.ndarray, num_classes: int) -> np.ndarray:
        values = values.astype(int)
        one_hot = np.zeros((values.shape[0], num_classes), dtype=float)
        one_hot[np.arange(values.shape[0]), values] = 1.0
        return one_hot

    def _torch_policy_value(
        self,
        q_network: TorchQNetwork,
        obs_features: torch.Tensor,
        policy_probs: torch.Tensor,
        action_space_n: int,
    ) -> torch.Tensor:
        obs_np = obs_features.cpu().numpy()
        obs_rep = np.repeat(obs_np, action_space_n, axis=0)
        actions = np.tile(np.arange(action_space_n), obs_np.shape[0])
        action_one_hot = self._one_hot(actions, action_space_n)
        obs_action = np.concatenate([obs_rep, action_one_hot], axis=1)
        obs_action_tensor = torch.tensor(obs_action, dtype=torch.float32).to(
            self.device
        )
        q_values = q_network(obs_action_tensor).view(obs_np.shape[0], action_space_n)
        return torch.sum(policy_probs * q_values, dim=1)

    def _compute_state_values(
        self,
        q_network: TorchQNetwork,
        observations: np.ndarray,
        state_space_n: int | None,
    ) -> np.ndarray:
        obs_features = self._features(observations, state_space_n)
        policy_probs = self.estimand.policy.action_probs(observations)
        action_space_n = policy_probs.shape[1]
        obs_tensor = torch.tensor(obs_features, dtype=torch.float32).to(self.device)
        policy_tensor = torch.tensor(policy_probs, dtype=torch.float32).to(self.device)
        values = self._torch_policy_value(
            q_network, obs_tensor, policy_tensor, action_space_n
        )
        return values.detach().cpu().numpy()
