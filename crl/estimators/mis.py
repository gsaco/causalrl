"""Marginalized importance sampling estimator."""

from __future__ import annotations

from typing import Any

import numpy as np

from crl.data.datasets import TrajectoryDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.base import (
    DiagnosticsConfig,
    EstimatorReport,
    OPEEstimator,
    compute_ci,
)
from crl.estimators.diagnostics import run_diagnostics
from crl.estimators.stats import mean_stderr
from crl.estimators.utils import compute_action_probs


class MarginalizedImportanceSamplingEstimator(OPEEstimator):
    """MIS estimator (Xie et al., 2019) for discrete state-action spaces."""

    required_assumptions = ["sequential_ignorability", "overlap", "markov"]
    required_fields = ["state_space_n"]
    diagnostics_keys = ["overlap", "ess", "weights", "max_weight", "model"]

    def __init__(
        self,
        estimand: PolicyValueEstimand,
        run_diagnostics: bool = True,
        diagnostics_config: DiagnosticsConfig | None = None,
        min_prob: float = 1e-6,
    ) -> None:
        super().__init__(estimand, run_diagnostics, diagnostics_config)
        self.min_prob = min_prob

    def estimate(self, data: TrajectoryDataset) -> EstimatorReport:
        self._validate_dataset(data)
        if data.state_space_n is None:
            raise ValueError("MIS requires discrete state_space_n.")

        obs = data.observations
        actions = data.actions
        rewards = data.rewards
        mask = data.mask
        num_states = data.state_space_n
        num_actions = data.action_space_n

        target_probs = compute_action_probs(self.estimand.policy, obs, actions)
        ratios = np.zeros_like(rewards, dtype=float)
        behavior_probs = np.zeros_like(rewards, dtype=float)

        for t in range(data.horizon):
            mask_t = mask[:, t]
            if not np.any(mask_t):
                continue
            states_t = obs[mask_t, t].astype(int)
            actions_t = actions[mask_t, t].astype(int)

            counts_s = np.bincount(states_t, minlength=num_states).astype(float)
            counts_sa = np.zeros((num_states, num_actions), dtype=float)
            for s, a in zip(states_t, actions_t, strict=True):
                counts_sa[s, a] += 1.0

            denom = np.maximum(counts_s[:, None], 1.0)
            behavior_prob_t = counts_sa / denom
            behavior_prob_t = np.clip(behavior_prob_t, self.min_prob, 1.0)

            behavior_probs[mask_t, t] = behavior_prob_t[states_t, actions_t]
            ratios[mask_t, t] = target_probs[mask_t, t] / behavior_probs[mask_t, t]

        discounts = data.discount ** np.arange(data.horizon)
        values = np.sum(ratios * rewards * discounts, axis=1)
        value = float(np.mean(values))
        stderr = mean_stderr(values)

        diagnostics: dict[str, Any] = {}
        warnings: list[str] = []
        if self.run_diagnostics:
            weights = np.prod(np.where(mask, ratios, 1.0), axis=1)
            diagnostics, warnings = run_diagnostics(
                weights, target_probs, behavior_probs, mask, self.diagnostics_config
            )

        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={"estimator": "MIS", "min_prob": self.min_prob},
            data=data,
        )
