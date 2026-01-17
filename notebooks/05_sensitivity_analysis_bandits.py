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
# # Sensitivity Analysis for Bandit OPE
#
# When ignorability may fail, sensitivity analysis reports partial-identification
# bounds across a range of confounding strengths.

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np

from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.sensitivity.bandits import sensitivity_bounds
from crl.viz import configure_notebook_display, save_figure
from crl.viz.plots import plot_sensitivity_curve

# %%
np.random.seed(0)
configure_notebook_display()

benchmark = SyntheticBandit(SyntheticBanditConfig(seed=10))
dataset = benchmark.sample(num_samples=1_500, seed=11)

# %%
gammas = np.linspace(1.0, 3.0, 15)
bounds = sensitivity_bounds(dataset, benchmark.target_policy, gammas)

fig = plot_sensitivity_curve(
    [
        {"gamma": g, "lower": lo, "upper": up}
        for g, lo, up in zip(bounds.gammas, bounds.lower, bounds.upper)
    ]
)
fig

# %%
output_dir = Path("docs/assets/figures")
output_dir.mkdir(parents=True, exist_ok=True)
save_figure(fig, output_dir / "sensitivity_bandits_curve")
