"""Plotting utilities for diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def plot_weight_histogram(weights: np.ndarray, bins: int = 50) -> Any:
    """Plot histogram of importance weights using matplotlib.

    Estimand:
        Not applicable.
    Assumptions:
        matplotlib is installed.
    Inputs:
        weights: Array of importance weights.
        bins: Number of histogram bins.
    Outputs:
        Matplotlib figure.
    Failure modes:
        Raises ImportError if matplotlib is unavailable.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is required for plotting diagnostics.") from exc

    w = np.asarray(weights, dtype=float)
    fig, ax = plt.subplots()
    ax.hist(w, bins=bins)
    ax.set_title("Importance Weight Histogram")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    return fig


def plot_ratio_histogram(ratios: np.ndarray, bins: int = 50) -> Any:
    """Plot histogram of target/behavior ratios.

    Estimand:
        Not applicable.
    Assumptions:
        matplotlib is installed.
    Inputs:
        ratios: Array of target/behavior ratios.
        bins: Number of histogram bins.
    Outputs:
        Matplotlib figure.
    Failure modes:
        Raises ImportError if matplotlib is unavailable.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is required for plotting diagnostics.") from exc

    r = np.asarray(ratios, dtype=float)
    fig, ax = plt.subplots()
    ax.hist(r, bins=bins)
    ax.set_title("Target/Behavior Ratio Histogram")
    ax.set_xlabel("Ratio")
    ax.set_ylabel("Count")
    return fig
