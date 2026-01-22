# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Diagnostics: Overlap and Effective Sample Size
#
# We compute overlap diagnostics and visualize effective sample size to assess
# weight stability before trusting OPE estimates.

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np

from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.core import Diagnostics
from crl.diagnostics import (
    compute_overlap_metrics,
    effective_sample_size,
    ess_ratio,
    state_shift_diagnostics,
    weight_tail_stats,
    weight_time_diagnostics,
)
from crl.estimators.utils import compute_action_probs
from crl.viz import (
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
from crl.viz.plots import (
    plot_bias_variance_tradeoff,
    plot_effective_sample_size,
    plot_overlap_diagnostics,
)

# %%
np.random.seed(0)
configure_notebook_display()

benchmark = SyntheticMDP(SyntheticMDPConfig(seed=3, horizon=6))
dataset = benchmark.sample(num_trajectories=400, seed=4)

# %%
target_probs = compute_action_probs(
    benchmark.target_policy, dataset.observations, dataset.actions
)
ratios = np.where(dataset.mask, target_probs / dataset.behavior_action_probs, 1.0)
weights = np.prod(ratios, axis=1)

fig_overlap = plot_overlap_diagnostics(
    target_probs, dataset.behavior_action_probs, mask=dataset.mask
)
fig_overlap

# %%
fig_ess = plot_effective_sample_size(ratios, by_time=True)
fig_ess

# %% [markdown]
# ## Numeric diagnostics
#
# The low-level diagnostics helpers return structured summaries you can log or
# store alongside estimates.

# %%
overlap_metrics = compute_overlap_metrics(
    target_probs, dataset.behavior_action_probs, mask=dataset.mask
)
ess_value = effective_sample_size(weights)
ess_ratio_value = ess_ratio(weights)
tail_stats = weight_tail_stats(weights)
time_stats = weight_time_diagnostics(np.cumprod(ratios, axis=1), dataset.mask)
shift_stats = state_shift_diagnostics(
    dataset.observations[dataset.mask], weights=ratios[dataset.mask]
)

diagnostics_obj = Diagnostics(
    ess={"ess": ess_value, "ess_ratio": ess_ratio_value},
    overlap=overlap_metrics,
    weights=tail_stats,
    model={},
)
diagnostics_obj.to_dict(), shift_stats

# %% [markdown]
# ## Bias-variance tradeoff plot
#
# This plot is useful for comparing estimator families when repeated runs or
# benchmark suites provide bias and variance estimates.

# %%
trade_rows = [
    {"estimator": "IS", "bias": 0.08, "variance": 0.30},
    {"estimator": "WIS", "bias": 0.04, "variance": 0.18},
    {"estimator": "DR", "bias": 0.02, "variance": 0.22},
]
fig_trade = plot_bias_variance_tradeoff(trade_rows)
fig_trade

# %% [markdown]
# ## Styling utilities
#
# The style helpers allow you to build consistent, paper-ready figures. Here we
# create a custom ESS plot using the raw diagnostics output.

# %%
set_style()
spec = paper_figspec(column="single", aspect=0.55)
with journal_style():
    fig_custom, ax = new_figure(spec)
    ax.plot(
        np.arange(len(time_stats["ess"])),
        time_stats["ess"],
        marker="o",
        color="0.25",
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("ESS")
    apply_axes_style(ax)

spec_alt = FigureSpec(width_in=3.6, height_in=2.2)
with paper_context():
    fig_alt, ax_alt = new_figure(spec_alt)
    ax_alt.bar(["tail_fraction"], [tail_stats["tail_fraction"]], color="0.45")
    ax_alt.set_ylabel("Fraction")
    apply_axes_style(ax_alt)

fig_custom, fig_alt

# %%
output_dir = Path("docs/assets/figures")
output_dir.mkdir(parents=True, exist_ok=True)
save_figure(fig_overlap, output_dir / "diagnostics_overlap")
save_figure(fig_ess, output_dir / "diagnostics_ess")
save_figure(fig_trade, output_dir / "diagnostics_bias_variance_tradeoff")
save_figure(fig_custom, output_dir / "diagnostics_custom_ess")
save_figure(fig_alt, output_dir / "diagnostics_tail_fraction")
