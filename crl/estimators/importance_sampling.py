"""Importance sampling estimators for OPE."""

from __future__ import annotations

from typing import Any

import numpy as np

from crl.data.datasets import LoggedBanditDataset, TrajectoryDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.base import (
    DiagnosticsConfig,
    EstimatorReport,
    OPEEstimator,
    compute_ci,
)
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
        Sequential ignorability, overlap/positivity, and known behavior propensities.
    Inputs:
        LoggedBanditDataset (n,) or TrajectoryDataset (n, t).
    Outputs:
        EstimatorReport with value and diagnostics.
    Failure modes:
        High variance under weak overlap.
    """

    required_assumptions = [
        "sequential_ignorability",
        "overlap",
        "behavior_policy_known",
    ]
    required_fields = ["behavior_action_probs"]
    diagnostics_keys = ["overlap", "ess", "weights", "max_weight", "model"]

    def __init__(
        self,
        estimand: PolicyValueEstimand,
        run_diagnostics: bool = True,
        diagnostics_config: DiagnosticsConfig | None = None,
        *,
        clip_rho: float | None = None,
        use_log_weights: bool = True,
        normalize: bool = False,
        min_prob: float = 1e-12,
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
        self.clip_rho = clip_rho
        self.use_log_weights = use_log_weights
        self.normalize = normalize
        self.min_prob = min_prob
        self._bootstrap_params.update(
            {
                "clip_rho": self.clip_rho,
                "use_log_weights": self.use_log_weights,
                "normalize": self.normalize,
                "min_prob": self.min_prob,
                "bootstrap": False,
                "bootstrap_config": None,
            }
        )

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
        behavior_probs = np.clip(data.behavior_action_probs, self.min_prob, 1.0)
        ratios = target_probs / behavior_probs
        weights = ratios.copy()
        clip = self.clip_rho or self.diagnostics_config.max_weight
        warnings: list[str] = []
        if clip is not None:
            if np.any(ratios > clip):
                warnings.append(
                    f"Clipped importance ratios at {clip:.3f} for stability."
                )
            weights = np.minimum(weights, clip)
        if self.normalize:
            value, stderr = weighted_mean_and_stderr(data.rewards, weights)
        else:
            values = weights * data.rewards
            value = float(np.mean(values))
            stderr = mean_stderr(values)
        diagnostics, diag_warnings = (
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
        warnings.extend(diag_warnings)
        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={
                "estimator": "IS",
                "num_samples": data.num_samples,
                "clip_rho": self.clip_rho,
                "use_log_weights": self.use_log_weights,
                "normalize": self.normalize,
            },
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
        behavior_probs = np.clip(data.behavior_action_probs, self.min_prob, 1.0)
        ratios = np.where(data.mask, target_probs / behavior_probs, 1.0)
        if self.use_log_weights:
            log_ratios = np.where(
                data.mask, np.log(np.clip(ratios, self.min_prob, None)), 0.0
            )
            weights = np.exp(np.sum(log_ratios, axis=1))
        else:
            weights = np.prod(ratios, axis=1)
        weights_for_est = weights.copy()
        clip = self.clip_rho or self.diagnostics_config.max_weight
        warnings: list[str] = []
        if clip is not None:
            if np.any(weights > clip):
                warnings.append(
                    f"Clipped trajectory weights at {clip:.3f} for stability."
                )
            weights_for_est = np.minimum(weights_for_est, clip)
        returns = compute_trajectory_returns(data.rewards, data.mask, data.discount)
        if self.normalize:
            value, stderr = weighted_mean_and_stderr(returns, weights_for_est)
        else:
            values = weights_for_est * returns
            value = float(np.mean(values))
            stderr = mean_stderr(values)
        diagnostics, diag_warnings = (
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
        warnings.extend(diag_warnings)
        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={
                "estimator": "IS",
                "num_trajectories": data.num_trajectories,
                "clip_rho": self.clip_rho,
                "use_log_weights": self.use_log_weights,
                "normalize": self.normalize,
            },
            data=data,
        )


class WISEstimator(OPEEstimator):
    """Weighted importance sampling estimator.

    Estimand:
        PolicyValueEstimand for the target policy.
    Assumptions:
        Sequential ignorability, overlap/positivity, and known behavior propensities.
    Inputs:
        LoggedBanditDataset (n,) or TrajectoryDataset (n, t).
    Outputs:
        EstimatorReport with value and diagnostics.
    Failure modes:
        Bias from normalization in small samples.
    """

    required_assumptions = [
        "sequential_ignorability",
        "overlap",
        "behavior_policy_known",
    ]
    required_fields = ["behavior_action_probs"]
    diagnostics_keys = ["overlap", "ess", "weights", "max_weight", "model"]

    def __init__(
        self,
        estimand: PolicyValueEstimand,
        run_diagnostics: bool = True,
        diagnostics_config: DiagnosticsConfig | None = None,
        *,
        clip_rho: float | None = None,
        use_log_weights: bool = True,
        min_prob: float = 1e-12,
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
        self.clip_rho = clip_rho
        self.use_log_weights = use_log_weights
        self.min_prob = min_prob
        self._bootstrap_params.update(
            {
                "clip_rho": self.clip_rho,
                "use_log_weights": self.use_log_weights,
                "min_prob": self.min_prob,
                "bootstrap": False,
                "bootstrap_config": None,
            }
        )

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
        behavior_probs = np.clip(data.behavior_action_probs, self.min_prob, 1.0)
        ratios = target_probs / behavior_probs
        weights = ratios.copy()
        clip = self.clip_rho or self.diagnostics_config.max_weight
        warnings: list[str] = []
        if clip is not None:
            if np.any(ratios > clip):
                warnings.append(
                    f"Clipped importance ratios at {clip:.3f} for stability."
                )
            weights = np.minimum(weights, clip)
        value, stderr = weighted_mean_and_stderr(data.rewards, weights)
        diagnostics, diag_warnings = (
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
        warnings.extend(diag_warnings)
        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={
                "estimator": "WIS",
                "num_samples": data.num_samples,
                "clip_rho": self.clip_rho,
                "use_log_weights": self.use_log_weights,
            },
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
        behavior_probs = np.clip(data.behavior_action_probs, self.min_prob, 1.0)
        ratios = np.where(data.mask, target_probs / behavior_probs, 1.0)
        if self.use_log_weights:
            log_ratios = np.where(
                data.mask, np.log(np.clip(ratios, self.min_prob, None)), 0.0
            )
            weights = np.exp(np.sum(log_ratios, axis=1))
        else:
            weights = np.prod(ratios, axis=1)
        weights_for_est = weights.copy()
        clip = self.clip_rho or self.diagnostics_config.max_weight
        warnings: list[str] = []
        if clip is not None:
            if np.any(weights > clip):
                warnings.append(
                    f"Clipped trajectory weights at {clip:.3f} for stability."
                )
            weights_for_est = np.minimum(weights_for_est, clip)
        returns = compute_trajectory_returns(data.rewards, data.mask, data.discount)
        value, stderr = weighted_mean_and_stderr(returns, weights_for_est)
        diagnostics, diag_warnings = (
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
        warnings.extend(diag_warnings)
        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={
                "estimator": "WIS",
                "num_trajectories": data.num_trajectories,
                "clip_rho": self.clip_rho,
                "use_log_weights": self.use_log_weights,
            },
            data=data,
        )


class PDISEstimator(OPEEstimator):
    """Per-decision importance sampling estimator.

    Estimand:
        PolicyValueEstimand for the target policy.
    Assumptions:
        Sequential ignorability, overlap/positivity, and known behavior propensities.
    Inputs:
        TrajectoryDataset (n, t).
    Outputs:
        EstimatorReport with value and diagnostics.
    Failure modes:
        Variance grows with horizon under weak overlap.
    """

    required_assumptions = [
        "sequential_ignorability",
        "overlap",
        "behavior_policy_known",
    ]
    required_fields = ["behavior_action_probs"]
    diagnostics_keys = ["overlap", "ess", "weights", "max_weight", "model"]

    def __init__(
        self,
        estimand: PolicyValueEstimand,
        run_diagnostics: bool = True,
        diagnostics_config: DiagnosticsConfig | None = None,
        *,
        clip_rho: float | None = None,
        use_log_weights: bool = True,
        min_prob: float = 1e-12,
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
        self.clip_rho = clip_rho
        self.use_log_weights = use_log_weights
        self.min_prob = min_prob
        self._bootstrap_params.update(
            {
                "clip_rho": self.clip_rho,
                "use_log_weights": self.use_log_weights,
                "min_prob": self.min_prob,
                "bootstrap": False,
                "bootstrap_config": None,
            }
        )

    def estimate(
        self, data: LoggedBanditDataset | TrajectoryDataset
    ) -> EstimatorReport:
        """Estimate policy value via PDIS."""

        self._validate_dataset(data)
        if isinstance(data, LoggedBanditDataset):
            return ISEstimator(
                self.estimand,
                self.run_diagnostics,
                self.diagnostics_config,
                clip_rho=self.clip_rho,
                use_log_weights=self.use_log_weights,
                min_prob=self.min_prob,
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
        behavior_probs = np.clip(data.behavior_action_probs, self.min_prob, 1.0)
        ratios = np.where(data.mask, target_probs / behavior_probs, 1.0)
        if self.use_log_weights:
            log_ratios = np.where(
                data.mask, np.log(np.clip(ratios, self.min_prob, None)), 0.0
            )
            cumulative = np.exp(np.cumsum(log_ratios, axis=1))
        else:
            cumulative = np.cumprod(ratios, axis=1)
        clip = self.clip_rho or self.diagnostics_config.max_weight
        warnings: list[str] = []
        if clip is not None:
            if np.any(cumulative > clip):
                warnings.append(
                    f"Clipped per-decision weights at {clip:.3f} for stability."
                )
            cumulative = np.minimum(cumulative, clip)
        step_returns = compute_stepwise_returns(data.rewards, data.mask, data.discount)
        values = np.sum(cumulative * step_returns, axis=1)
        value = float(np.mean(values))
        stderr = mean_stderr(values)
        diagnostics, diag_warnings = (
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
        warnings.extend(diag_warnings)
        return self._build_report(
            value=value,
            stderr=stderr,
            ci=compute_ci(value, stderr),
            diagnostics=diagnostics,
            warnings=warnings,
            metadata={
                "estimator": "PDIS",
                "num_trajectories": data.num_trajectories,
                "clip_rho": self.clip_rho,
                "use_log_weights": self.use_log_weights,
            },
            data=data,
        )
