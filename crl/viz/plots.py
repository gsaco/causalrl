"""Publication-ready plotting helpers."""

from __future__ import annotations

import textwrap
from typing import Any

import numpy as np

from crl.diagnostics.ess import effective_sample_size
from crl.diagnostics.overlap import compute_overlap_metrics
from crl.viz.style import (
    FigureSpec,
    apply_axes_style,
    journal_style,
    new_figure,
    paper_figspec,
)


def _as_finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[np.isfinite(x)]


def _resolve_axes(
    ax: Any | None,
    *,
    column: str,
    aspect: float,
    height_override: float | None = None,
) -> tuple[Any, Any]:
    if ax is None:
        spec = paper_figspec(column=column, aspect=aspect)
        if height_override is not None:
            spec = FigureSpec(width_in=spec.width_in, height_in=height_override)
        fig, ax = new_figure(spec)
    else:
        fig = ax.figure
    return fig, ax


def plot_importance_weights(
    weights: np.ndarray,
    *,
    bins: int = 40,
    logy: bool = True,
    xlabel: str = r"$\hat{w}$",
    ylabel: str = "Count",
    title: str | None = None,
    column: str = "single",
    aspect: float = 0.62,
    ax: Any | None = None,
) -> Any:
    """Plot importance weight distribution."""
    w = _as_finite_1d(weights)

    with journal_style():
        fig, ax = _resolve_axes(ax, column=column, aspect=aspect)
        ax.hist(
            w,
            bins=bins,
            color="0.40",
            edgecolor="0.10",
            linewidth=1.0,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if not logy else f"{ylabel} (log)")
        if title:
            ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        ax.margins(x=0.02)
        apply_axes_style(ax)
        return fig


def plot_effective_sample_size(
    weights: np.ndarray,
    *,
    by_time: bool = False,
    title: str | None = None,
    column: str = "single",
    aspect: float = 0.62,
    ax: Any | None = None,
) -> Any:
    """Plot effective sample size diagnostics."""
    from matplotlib.ticker import MaxNLocator

    w = np.asarray(weights, dtype=float)

    with journal_style():
        fig, ax = _resolve_axes(ax, column=column, aspect=aspect)
        if by_time:
            if w.ndim != 2:
                raise ValueError("by_time=True requires weights with shape (n, t).")
            ess_series = np.array(
                [effective_sample_size(w[:, t]) for t in range(w.shape[1])]
            )
            ax.plot(
                np.arange(w.shape[1]),
                ess_series,
                marker="o",
                color="0.25",
                markerfacecolor="0.25",
                markeredgecolor="0.10",
                linewidth=1.6,
            )
            ax.set_xlabel("Time step")
            ax.set_ylabel("ESS")
            if title:
                ax.set_title(title)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
            ax.set_ylim(bottom=0)
        else:
            ess_value = effective_sample_size(w.reshape(-1))
            ax.bar(["ESS"], [ess_value], color="0.35", edgecolor="0.10", linewidth=1.0)
            ax.set_ylabel("ESS")
            if title:
                ax.set_title(title)
            ax.set_ylim(bottom=0)
        ax.margins(y=0.1)
        apply_axes_style(ax)
        return fig


def plot_overlap_diagnostics(
    target_action_probs: np.ndarray,
    behavior_action_probs: np.ndarray,
    mask: np.ndarray | None = None,
    threshold: float = 1e-3,
    *,
    bins: int = 40,
    xlabel: str = r"$\hat{\nu}$",
    ylabel: str = "Count",
    title: str | None = None,
    column: str = "double",
    aspect: float = 0.55,
    ax: Any | None = None,
) -> Any:
    """Plot overlap diagnostics from target/behavior propensities."""
    metrics = compute_overlap_metrics(
        target_action_probs, behavior_action_probs, mask=mask, threshold=threshold
    )
    ratios = np.asarray(target_action_probs) / np.asarray(behavior_action_probs)
    ratios = _as_finite_1d(ratios)

    with journal_style():
        fig, ax = _resolve_axes(ax, column=column, aspect=aspect)
        ax.hist(
            ratios,
            bins=bins,
            color="0.40",
            edgecolor="0.10",
            linewidth=1.0,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.text(
            0.98,
            0.95,
            f"support violations: {metrics['support_violations']}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "0.85",
                "alpha": 0.9,
            },
        )
        ax.margins(x=0.02)
        apply_axes_style(ax)
        return fig


def plot_estimator_comparison(
    df: Any,
    truth: float | None = None,
    *,
    xlabel: str = "Estimated value",
    title: str | None = None,
    column: str = "double",
    aspect: float = 0.62,
    ax: Any | None = None,
) -> Any:
    """Plot estimator comparison with confidence intervals."""
    if hasattr(df, "to_dict"):
        records = df.to_dict(orient="records")
    else:
        records = list(df)

    labels = [str(r.get("estimator", f"est_{i}")) for i, r in enumerate(records)]
    values = np.array([r.get("value") for r in records], dtype=float)
    ci = [r.get("ci") for r in records]
    lower = np.array([c[0] if c else np.nan for c in ci], dtype=float)
    upper = np.array([c[1] if c else np.nan for c in ci], dtype=float)
    err_low = values - lower
    err_high = upper - values

    wrapped_labels = [
        textwrap.fill(label, width=26, break_long_words=False, break_on_hyphens=False)
        for label in labels
    ]
    line_count = sum(label.count("\n") + 1 for label in wrapped_labels)
    fig_height = max(2.6, 0.45 * line_count + 0.8)

    with journal_style():
        fig, ax = _resolve_axes(
            ax,
            column=column,
            aspect=aspect,
            height_override=fig_height,
        )
        y = np.arange(len(labels))
        ax.scatter(
            values,
            y,
            s=36,
            color="0.20",
            edgecolor="0.10",
            linewidth=0.8,
            zorder=3,
        )
        mask = np.isfinite(err_low) & np.isfinite(err_high)
        if np.any(mask):
            ax.errorbar(
                values[mask],
                y[mask],
                xerr=[err_low[mask], err_high[mask]],
                fmt="none",
                ecolor="0.35",
                elinewidth=1.4,
                capsize=3,
                zorder=2,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(wrapped_labels)
        ax.set_xlabel(xlabel)
        if title:
            ax.set_title(title)
        ax.margins(x=0.05)
        if truth is not None:
            ax.axvline(
                truth, color="0.15", linestyle="--", linewidth=1.2, label="True value"
            )
            ax.legend(loc="upper right", frameon=False)
        apply_axes_style(ax)
        return fig


def plot_bias_variance_tradeoff(
    results: Any,
    *,
    title: str | None = None,
    column: str = "single",
    aspect: float = 0.62,
    ax: Any | None = None,
) -> Any:
    """Plot bias-variance tradeoff for estimators with known truth."""
    if hasattr(results, "to_dict"):
        records = results.to_dict(orient="records")
    else:
        records = list(results)

    labels = [str(r.get("estimator", f"est_{i}")) for i, r in enumerate(records)]
    bias = np.array([r.get("bias", 0.0) for r in records], dtype=float)
    variance = np.array([r.get("variance", 0.0) for r in records], dtype=float)
    bias_sq = bias**2

    with journal_style():
        fig, ax = _resolve_axes(ax, column=column, aspect=aspect)
        ax.scatter(
            variance,
            bias_sq,
            color="0.25",
            edgecolor="0.10",
            linewidth=0.8,
            s=45,
            zorder=3,
        )
        offsets = np.linspace(-8, 8, max(len(labels), 2))
        for i, label in enumerate(labels):
            offset = offsets[i % len(offsets)]
            ax.annotate(
                label,
                (variance[i], bias_sq[i]),
                fontsize=9,
                xytext=(6, offset),
                textcoords="offset points",
                ha="left",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": "0.85",
                    "alpha": 0.9,
                },
            )
        ax.set_xlabel("Variance")
        ax.set_ylabel("Bias$^2$")
        if title:
            ax.set_title(title)
        ax.margins(x=0.05, y=0.1)
        apply_axes_style(ax)
        return fig


def plot_sensitivity_curve(
    bounds_df: Any,
    *,
    xlabel: str = r"Sensitivity parameter $\gamma$",
    ylabel: str = "Policy value",
    title: str | None = None,
    column: str = "double",
    aspect: float = 0.62,
    ax: Any | None = None,
) -> Any:
    """Plot sensitivity curve with lower/upper bounds."""
    if hasattr(bounds_df, "to_dict"):
        records = bounds_df.to_dict(orient="records")
    else:
        records = list(bounds_df)

    gamma = np.array([r.get("gamma") for r in records], dtype=float)
    lower = np.array([r.get("lower") for r in records], dtype=float)
    upper = np.array([r.get("upper") for r in records], dtype=float)

    with journal_style():
        fig, ax = _resolve_axes(ax, column=column, aspect=aspect)
        ax.plot(gamma, lower, color="0.20", label="Lower", linewidth=1.8)
        ax.plot(
            gamma,
            upper,
            color="0.50",
            label="Upper",
            linewidth=1.8,
            linestyle="--",
        )
        ax.fill_between(gamma, lower, upper, color="0.85", alpha=1.0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        legend = ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="0.85",
        )
        legend.get_frame().set_linewidth(0.6)
        ax.margins(x=0.02)
        apply_axes_style(ax)
        return fig
