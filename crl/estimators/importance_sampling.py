"""Importance sampling estimators for OPE."""

from __future__ import annotations

import numpy as np

from crl.data.datasets import LoggedBanditDataset, TrajectoryDataset
from crl.estimators.base import EstimatorReport, OPEEstimator, compute_ci
from crl.estimators.diagnostics import run_diagnostics
from crl.estimators.stats import mean_stderr, weighted_mean_and_stderr
from crl.estimators.utils import (
    compute_action_probs,
    compute_stepwise_returns,
    compute_trajectory_returns,
)


class ISEstimator(OPEEstimator):
    """Trajectory-level importance sampling estimator.

    Estimand:
        PolicyValueEstimand for the target policy.
    Assumptions:
        Sequential ignorability and overlap/positivity.
    Inputs:
        LoggedBanditDataset (n,) or TrajectoryDataset (n, t).
    Outputs:
        EstimatorReport with value and diagnostics.
    Failure modes:
        High variance under weak overlap.
    """

    required_assumptions = ["sequential_ignorability", "overlap"]
    required_fields = ["behavior_action_probs"]
    diagnostics_keys = ["overlap", "ess", "weights", "max_weight", "model"]

    def estimate(
        self, data: LoggedBanditDataset | TrajectoryDataset
    ) -> EstimatorReport:
        """Estimate policy value via IS."""

        self._validate_dataset(data)
        if isinstance(data, LoggedBanditDataset):
            return self._estimate_bandit(data)
        return self._estimate_trajectory(data)

    def _estimate_bandit(self, data: LoggedBanditDataset) -> EstimatorReport:
        if data.behavior_action_probs is None:
            raise ValueError(
                "behavior_action_probs are required for IS on bandit data."
            )
        target_probs = self.estimand.policy.action_prob(data.contexts, data.actions)
        ratios = target_probs / data.behavior_action_probs
        weights = ratios.copy()
        if self.diagnostics_config.max_weight is not None:
            weights = np.minimum(weights, self.diagnostics_config.max_weight)
        values = weights * data.rewards
        value = float(np.mean(values))
        stderr = mean_stderr(values)
        diagnostics, warnings = (
            run_diagnostics(
                ratios,
                target_probs,
                data.behavior_action_probs,
                None,
                self.diagnostics_config,
            )
            if self.run_diagnostics
            else ({}, [])
        )
        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={"estimator": "IS", "num_samples": data.num_samples},
            data=data,
        )

    def _estimate_trajectory(self, data: TrajectoryDataset) -> EstimatorReport:
        if data.behavior_action_probs is None:
            raise ValueError(
                "behavior_action_probs are required for IS on trajectories."
            )
        target_probs = compute_action_probs(
            self.estimand.policy, data.observations, data.actions
        )
        ratios = np.where(data.mask, target_probs / data.behavior_action_probs, 1.0)
        weights = np.prod(ratios, axis=1)
        weights_for_est = weights.copy()
        if self.diagnostics_config.max_weight is not None:
            weights_for_est = np.minimum(
                weights_for_est, self.diagnostics_config.max_weight
            )
        returns = compute_trajectory_returns(data.rewards, data.mask, data.discount)
        values = weights_for_est * returns
        value = float(np.mean(values))
        stderr = mean_stderr(values)
        diagnostics, warnings = (
            run_diagnostics(
                weights,
                target_probs,
                data.behavior_action_probs,
                data.mask,
                self.diagnostics_config,
            )
            if self.run_diagnostics
            else ({}, [])
        )
        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={"estimator": "IS", "num_trajectories": data.num_trajectories},
            data=data,
        )


class WISEstimator(OPEEstimator):
    """Weighted importance sampling estimator.

    Estimand:
        PolicyValueEstimand for the target policy.
    Assumptions:
        Sequential ignorability and overlap/positivity.
    Inputs:
        LoggedBanditDataset (n,) or TrajectoryDataset (n, t).
    Outputs:
        EstimatorReport with value and diagnostics.
    Failure modes:
        Bias from normalization in small samples.
    """

    required_assumptions = ["sequential_ignorability", "overlap"]
    required_fields = ["behavior_action_probs"]
    diagnostics_keys = ["overlap", "ess", "weights", "max_weight", "model"]

    def estimate(
        self, data: LoggedBanditDataset | TrajectoryDataset
    ) -> EstimatorReport:
        """Estimate policy value via WIS."""

        self._validate_dataset(data)
        if isinstance(data, LoggedBanditDataset):
            return self._estimate_bandit(data)
        return self._estimate_trajectory(data)

    def _estimate_bandit(self, data: LoggedBanditDataset) -> EstimatorReport:
        if data.behavior_action_probs is None:
            raise ValueError(
                "behavior_action_probs are required for WIS on bandit data."
            )
        target_probs = self.estimand.policy.action_prob(data.contexts, data.actions)
        ratios = target_probs / data.behavior_action_probs
        weights = ratios.copy()
        if self.diagnostics_config.max_weight is not None:
            weights = np.minimum(weights, self.diagnostics_config.max_weight)
        value, stderr = weighted_mean_and_stderr(data.rewards, weights)
        diagnostics, warnings = (
            run_diagnostics(
                ratios,
                target_probs,
                data.behavior_action_probs,
                None,
                self.diagnostics_config,
            )
            if self.run_diagnostics
            else ({}, [])
        )
        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={"estimator": "WIS", "num_samples": data.num_samples},
            data=data,
        )

    def _estimate_trajectory(self, data: TrajectoryDataset) -> EstimatorReport:
        if data.behavior_action_probs is None:
            raise ValueError(
                "behavior_action_probs are required for WIS on trajectories."
            )
        target_probs = compute_action_probs(
            self.estimand.policy, data.observations, data.actions
        )
        ratios = np.where(data.mask, target_probs / data.behavior_action_probs, 1.0)
        weights = np.prod(ratios, axis=1)
        weights_for_est = weights.copy()
        if self.diagnostics_config.max_weight is not None:
            weights_for_est = np.minimum(
                weights_for_est, self.diagnostics_config.max_weight
            )
        returns = compute_trajectory_returns(data.rewards, data.mask, data.discount)
        value, stderr = weighted_mean_and_stderr(returns, weights_for_est)
        diagnostics, warnings = (
            run_diagnostics(
                weights,
                target_probs,
                data.behavior_action_probs,
                data.mask,
                self.diagnostics_config,
            )
            if self.run_diagnostics
            else ({}, [])
        )
        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={"estimator": "WIS", "num_trajectories": data.num_trajectories},
            data=data,
        )


class PDISEstimator(OPEEstimator):
    """Per-decision importance sampling estimator.

    Estimand:
        PolicyValueEstimand for the target policy.
    Assumptions:
        Sequential ignorability and overlap/positivity.
    Inputs:
        TrajectoryDataset (n, t).
    Outputs:
        EstimatorReport with value and diagnostics.
    Failure modes:
        Variance grows with horizon under weak overlap.
    """

    required_assumptions = ["sequential_ignorability", "overlap"]
    required_fields = ["behavior_action_probs"]
    diagnostics_keys = ["overlap", "ess", "weights", "max_weight", "model"]

    def estimate(
        self, data: LoggedBanditDataset | TrajectoryDataset
    ) -> EstimatorReport:
        """Estimate policy value via PDIS."""

        self._validate_dataset(data)
        if isinstance(data, LoggedBanditDataset):
            return ISEstimator(
                self.estimand, self.run_diagnostics, self.diagnostics_config
            )._estimate_bandit(data)
        return self._estimate_trajectory(data)

    def _estimate_trajectory(self, data: TrajectoryDataset) -> EstimatorReport:
        if data.behavior_action_probs is None:
            raise ValueError(
                "behavior_action_probs are required for PDIS on trajectories."
            )
        target_probs = compute_action_probs(
            self.estimand.policy, data.observations, data.actions
        )
        ratios = np.where(data.mask, target_probs / data.behavior_action_probs, 1.0)
        cumulative = np.cumprod(ratios, axis=1)
        if self.diagnostics_config.max_weight is not None:
            cumulative = np.minimum(cumulative, self.diagnostics_config.max_weight)
        step_returns = compute_stepwise_returns(data.rewards, data.mask, data.discount)
        values = np.sum(cumulative * step_returns, axis=1)
        value = float(np.mean(values))
        stderr = mean_stderr(values)
        diagnostics, warnings = (
            run_diagnostics(
                cumulative[:, -1],
                target_probs,
                data.behavior_action_probs,
                data.mask,
                self.diagnostics_config,
            )
            if self.run_diagnostics
            else ({}, [])
        )
        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={"estimator": "PDIS", "num_trajectories": data.num_trajectories},
            data=data,
        )
