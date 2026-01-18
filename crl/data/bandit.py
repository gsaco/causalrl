"""Bandit dataset contract."""

from __future__ import annotations

from crl.data.datasets import LoggedBanditDataset as BanditDataset

LoggedBanditDataset = BanditDataset

__all__ = ["BanditDataset", "LoggedBanditDataset"]
