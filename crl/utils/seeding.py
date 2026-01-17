"""Seeding utilities for deterministic behavior."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch (if available)."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
