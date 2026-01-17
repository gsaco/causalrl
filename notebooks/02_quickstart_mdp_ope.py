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
# # Quickstart: MDP Off-Policy Evaluation
#
# We evaluate a target policy on a finite-horizon synthetic MDP and compare
# multiple estimators. Figures are rendered inline and exported for docs.

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np

from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.ope import evaluate
from crl.viz import configure_notebook_display, save_figure

# %%
np.random.seed(0)
configure_notebook_display()

benchmark = SyntheticMDP(SyntheticMDPConfig(seed=0, horizon=5))
dataset = benchmark.sample(num_trajectories=300, seed=1)
true_value = benchmark.true_policy_value(benchmark.target_policy)

report = evaluate(
    dataset=dataset,
    policy=benchmark.target_policy,
    estimators=["is", "wis", "pdis", "dr", "wdr", "mrdr", "fqe"],
)

summary = report.summary_table()
summary

# %%
fig = report.plot_estimator_comparison(truth=true_value)
fig

# %%
output_dir = Path("docs/assets/figures")
output_dir.mkdir(parents=True, exist_ok=True)
save_figure(fig, output_dir / "quickstart_mdp_estimator_comparison")
