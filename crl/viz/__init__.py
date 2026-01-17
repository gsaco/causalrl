"""Visualization helpers (matplotlib is optional)."""

from crl.viz.plots import (
    plot_bias_variance_tradeoff,
    plot_effective_sample_size,
    plot_estimator_comparison,
    plot_importance_weights,
    plot_overlap_diagnostics,
    plot_sensitivity_curve,
)
from crl.viz.style import (
    FigureSpec,
    apply_axes_style,
    configure_notebook_display,
    journal_style,
    new_figure,
    paper_context,
    paper_figspec,
    save_figure,
    set_style,
)

__all__ = [
    "FigureSpec",
    "apply_axes_style",
    "configure_notebook_display",
    "journal_style",
    "new_figure",
    "paper_context",
    "paper_figspec",
    "save_figure",
    "set_style",
    "plot_importance_weights",
    "plot_effective_sample_size",
    "plot_overlap_diagnostics",
    "plot_estimator_comparison",
    "plot_bias_variance_tradeoff",
    "plot_sensitivity_curve",
]
