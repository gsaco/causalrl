"""Plotting utilities for diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def plot_weight_histogram(
    weights: np.ndarray,
    bins: int = 50,
    *,
    xlabel: str = r"$\hat{w}$",
    ylabel: str = "Count",
    title: str | None = None,
    column: str = "double",
    aspect: float = 0.55,
    clip_quantile: float | None = None,
    log_y: bool = False,
    ax: Any | None = None,
) -> Any:
    """
    Journal-ready histogram for importance weights.
    Returns the Matplotlib figure (so callers can save/export consistently).
    """
    from crl.viz import apply_axes_style, journal_style, new_figure, paper_figspec

    w = _as_finite_1d(weights)
    if clip_quantile is not None and w.size > 0:
        hi = float(np.quantile(w, clip_quantile))
        w = np.clip(w, None, hi)

    with journal_style():
        if ax is None:
            spec = paper_figspec(column=column, aspect=aspect)
            fig, ax = new_figure(spec)
        else:
            fig = ax.figure

        ax.hist(
            w,
            bins=bins,
            color="0.40",
            edgecolor="0.10",
            linewidth=1.0,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        if log_y:
            ax.set_yscale("log")

        apply_axes_style(ax)
        return fig


def plot_ratio_histogram(
    ratios: np.ndarray,
    bins: int = 50,
    *,
    xlabel: str = r"$\hat{\nu}$",
    ylabel: str = "Count",
    title: str | None = None,
    column: str = "double",
    aspect: float = 0.55,
    clip_quantile: float | None = None,
    log_y: bool = False,
    ax: Any | None = None,
) -> Any:
    """Journal-ready histogram for target/behavior ratios."""
    return plot_weight_histogram(
        ratios,
        bins=bins,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        column=column,
        aspect=aspect,
        clip_quantile=clip_quantile,
        log_y=log_y,
        ax=ax,
    )
