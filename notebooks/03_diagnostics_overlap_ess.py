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
from crl.estimators.utils import compute_action_probs
from crl.viz import configure_notebook_display, save_figure
from crl.viz.plots import plot_effective_sample_size, plot_overlap_diagnostics

# %%
np.random.seed(0)
configure_notebook_display()

benchmark = SyntheticMDP(SyntheticMDPConfig(seed=3, horizon=6))
dataset = benchmark.sample(num_trajectories=400, seed=4)

# %%
target_probs = compute_action_probs(
    benchmark.target_policy, dataset.observations, dataset.actions
)
ratios = np.where(
    dataset.mask, target_probs / dataset.behavior_action_probs, 1.0
)
weights = np.prod(ratios, axis=1)

fig_overlap = plot_overlap_diagnostics(
    target_probs, dataset.behavior_action_probs, mask=dataset.mask
)
fig_overlap

# %%
fig_ess = plot_effective_sample_size(ratios, by_time=True)
fig_ess

# %%
output_dir = Path("docs/assets/figures")
output_dir.mkdir(parents=True, exist_ok=True)
save_figure(fig_overlap, output_dir / "diagnostics_overlap")
save_figure(fig_ess, output_dir / "diagnostics_ess")
