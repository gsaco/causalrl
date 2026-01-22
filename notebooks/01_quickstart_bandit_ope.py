# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quickstart: Bandit Off-Policy Evaluation
#
# This notebook runs a full bandit OPE workflow with an academic plotting style
# and a self-contained report. We use the synthetic benchmark so the true policy
# value is known for comparison.

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np

from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.ope import evaluate
from crl.viz import configure_notebook_display, save_figure

# %%
np.random.seed(0)
configure_notebook_display()

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

# %%
print(
    summary[["estimator", "value", "lower_bound", "upper_bound"]]
    .round(3)
    .to_string(index=False)
)

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

# %%
output_dir = Path("docs/assets/figures")
output_dir.mkdir(parents=True, exist_ok=True)
save_figure(fig, output_dir / "quickstart_bandit_estimator_comparison")
save_figure(fig_w, output_dir / "quickstart_bandit_weights")

# %% [markdown]
# ## Takeaways
#
# - IS is unbiased but can be noisy; WIS trades bias for stability.
# - Double RL can outperform if the outcome model fits well.
# - Always sanity-check weight diagnostics before trusting estimates.
