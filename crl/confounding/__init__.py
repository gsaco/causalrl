"""Confounding-robust OPE modules."""

from crl.confounding.proximal import (
    ProximalBanditDataset,
    ProximalBanditEstimator,
    ProximalConfig,
    ProximalOPEEstimator,
)

__all__ = [
    "ProximalBanditDataset",
    "ProximalBanditEstimator",
    "ProximalConfig",
    "ProximalOPEEstimator",
]
