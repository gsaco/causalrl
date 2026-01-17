"""Core interfaces for CRL."""

from crl.core.diagnostics import Diagnostics
from crl.core.datasets import BanditDataset, TrajectoryDataset
from crl.core.policy import Policy
from crl.core.reports import EstimationReport
from crl.estimands.policy_value import PolicyContrastEstimand, PolicyValueEstimand
from crl.estimators.base import EstimatorReport, OPEEstimator

Estimator = OPEEstimator

__all__ = [
    "Diagnostics",
    "BanditDataset",
    "TrajectoryDataset",
    "Policy",
    "EstimationReport",
    "EstimatorReport",
    "Estimator",
    "PolicyValueEstimand",
    "PolicyContrastEstimand",
]
