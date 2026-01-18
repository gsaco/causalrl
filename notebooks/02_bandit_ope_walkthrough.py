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
# # 02 — Bandit OPE Walkthrough
#
# We compare IS, WIS, and Double RL on a synthetic contextual bandit. This
# notebook emphasizes diagnostics: overlap, ESS, and weight tails.

# %% [markdown]
# ## Setup
#
# ```
# pip install "causalrl[plots]"
# ```

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np

from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.ope import evaluate
from crl.utils.seeding import set_seed
from crl.viz import configure_notebook_display, save_figure

# %%
set_seed(0)
np.random.seed(0)
configure_notebook_display()

# %% [markdown]
# ## Run estimators

# %%
benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = benchmark.sample(num_samples=1_000, seed=1)
true_value = benchmark.true_policy_value(benchmark.target_policy)

report = evaluate(
    dataset=dataset,
    policy=benchmark.target_policy,
    estimators=["is", "wis", "double_rl"],
)
summary = report.summary_table()
summary

# %% [markdown]
# ## Diagnostics and plots
#
# We'll plot estimator comparisons and inspect weight distributions.

# %%
fig = report.plot_estimator_comparison(truth=true_value)
fig

# %%
weights = (
    benchmark.target_policy.action_prob(dataset.contexts, dataset.actions)
    / dataset.behavior_action_probs
)
fig_w = report.plot_importance_weights(weights, logy=True)
fig_w

# %% [markdown]
# ## Save figures
#
# These files are used in the docs site.

# %%
output_dir = Path("docs/assets/figures")
output_dir.mkdir(parents=True, exist_ok=True)
save_figure(fig, output_dir / "bandit_walkthrough_estimator_comparison")
save_figure(fig_w, output_dir / "bandit_walkthrough_weights")

# %% [markdown]
# ## Takeaways
#
# - IS is unbiased but can be high variance.
# - WIS normalizes weights to reduce variance, at the cost of bias.
# - Diagnostics (ESS, overlap, tails) tell you when to trust estimates.
