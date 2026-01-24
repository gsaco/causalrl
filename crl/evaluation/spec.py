"""Typed evaluation specifications for CRL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from crl.assumptions import AssumptionSet
from crl.core.policy import Policy
from crl.data.datasets import LoggedBanditDataset, TrajectoryDataset
from crl.data.transition import TransitionDataset

Dataset = LoggedBanditDataset | TrajectoryDataset | TransitionDataset

EstimatorName = Literal[
    "is",
    "wis",
    "pdis",
    "dr",
    "wdr",
    "magic",
    "mrdr",
    "mis",
    "fqe",
    "dual_dice",
    "gen_dice",
    "double_rl",
    "drl",
    "hcope",
]


@dataclass(frozen=True)
class InferenceSpec:
    alpha: float = 0.05
    method: Literal["asymptotic", "bootstrap", "none"] = "asymptotic"
    bootstrap_num: int = 200
    bootstrap_kind: Literal["trajectory", "iid", "block"] = "trajectory"
    bootstrap_block_size: int = 5
    seed: int = 0


@dataclass(frozen=True)
class DiagnosticsSpec:
    enabled: bool = True
    suites: Sequence[
        Literal[
            "overlap",
            "ess",
            "weights",
            "shift",
            "model_fit",
            "calibration",
        ]
    ] = ("overlap", "ess", "weights", "shift")
    fail_on: Sequence[
        Literal[
            "overlap_violation",
            "ess_too_low",
            "extreme_tail",
            "severe_shift",
        ]
    ] = ()
    min_ess: float | None = None
    max_weight: float | None = None


@dataclass(frozen=True)
class SensitivitySpec:
    enabled: bool = False
    kind: Literal["bandit_gamma", "sequential_gamma"] = "bandit_gamma"
    gammas: np.ndarray | None = None
    baseline_value: float | None = None


@dataclass(frozen=True)
class ReportSpec:
    html: bool = True
    include_figures: bool = True
    theme: Literal["light", "dark", "auto"] = "auto"


@dataclass(frozen=True)
class EvaluationSpec:
    policy: Policy
    dataset: Dataset
    estimand: Literal["policy_value", "policy_contrast"] = "policy_value"
    baseline_policy: Policy | None = None
    assumptions: AssumptionSet | None = None
    estimators: Sequence[EstimatorName] | Literal["auto"] = "auto"
    inference: InferenceSpec = InferenceSpec()
    diagnostics: DiagnosticsSpec = DiagnosticsSpec()
    sensitivity: SensitivitySpec = SensitivitySpec()
    report: ReportSpec = ReportSpec()
    seed: int = 0


__all__ = [
    "Dataset",
    "EstimatorName",
    "InferenceSpec",
    "DiagnosticsSpec",
    "SensitivitySpec",
    "ReportSpec",
    "EvaluationSpec",
]
