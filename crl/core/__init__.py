"""Core interfaces for CRL."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crl.core.datasets import BanditDataset, TrajectoryDataset, TransitionDataset
from crl.core.diagnostics import Diagnostics
from crl.core.policy import Policy

if TYPE_CHECKING:
    from crl.core.reports import EstimationReport
    from crl.estimands.policy_value import PolicyContrastEstimand, PolicyValueEstimand
    from crl.estimators.base import EstimatorReport, OPEEstimator

__all__ = [
    "Diagnostics",
    "BanditDataset",
    "TrajectoryDataset",
    "TransitionDataset",
    "Policy",
    "EstimationReport",
    "EstimatorReport",
    "Estimator",
    "OPEEstimator",
    "PolicyValueEstimand",
    "PolicyContrastEstimand",
]


def __getattr__(name: str):
    if name == "EstimationReport":
        from crl.core.reports import EstimationReport

        return EstimationReport
    if name in {"PolicyValueEstimand", "PolicyContrastEstimand"}:
        from crl.estimands.policy_value import (
            PolicyContrastEstimand,
            PolicyValueEstimand,
        )

        return (
            PolicyValueEstimand
            if name == "PolicyValueEstimand"
            else PolicyContrastEstimand
        )
    if name in {"EstimatorReport", "Estimator", "OPEEstimator"}:
        from crl.estimators.base import EstimatorReport, OPEEstimator

        if name == "EstimatorReport":
            return EstimatorReport
        return OPEEstimator
    raise AttributeError(f"module 'crl.core' has no attribute {name!r}")
