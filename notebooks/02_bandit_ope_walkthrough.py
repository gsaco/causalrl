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
# # 02 — Bandit OPE Walkthrough
#
# We compare IS, WIS, and Double RL on a synthetic contextual bandit. This
# notebook emphasizes diagnostics: overlap, ESS, and weight tails.

# %% [markdown]
# ## Setup
#
# ```
# pip install ".[plots]"
# ```

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import OVERLAP, SEQUENTIAL_IGNORABILITY, BEHAVIOR_POLICY_KNOWN
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.behavior import BehaviorPolicyFit, behavior_diagnostics, fit_behavior_policy
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.base import DiagnosticsConfig, EstimatorReport, OPEEstimator
from crl.estimators.double_rl import DoubleRLConfig, DoubleRLEstimator
from crl.estimators.importance_sampling import ISEstimator
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

estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=1.0,
    horizon=1,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN]),
)

report = evaluate(
    dataset=dataset,
    policy=benchmark.target_policy,
    estimators=["is", "wis", "double_rl"],
)
summary = report.summary_table()
summary

# %% [markdown]
# ## Estimate behavior propensities
#
# When propensities are missing, you can estimate a behavior policy from logged
# data. The helper returns both propensities and diagnostics.

# %%
behavior_fit = fit_behavior_policy(
    dataset, method="logit", seed=0, store_action_probs=True
)
behavior_fit.diagnostics, isinstance(behavior_fit, BehaviorPolicyFit)

# %%
extra_diag = behavior_diagnostics(
    behavior_fit.metadata["action_probs"], dataset.actions, behavior_fit.propensities
)
extra_diag["propensity"]

# %%
dataset_est = behavior_fit.apply(dataset)
dataset_est.describe()

# %%
print(
    summary[["estimator", "value", "lower_bound", "upper_bound"]]
    .round(3)
    .to_string(index=False)
)
best = summary.sort_values("stderr").iloc[0]
print(f"Lowest stderr: {best['estimator']} (stderr={best['stderr']:.4f})")

# %% [markdown]
# ## Custom estimator configuration
#
# You can tune estimator hyperparameters and diagnostics behavior explicitly.

# %%
double_rl_config = DoubleRLConfig(num_folds=3, ridge=1e-2, seed=0)
double_rl_report = DoubleRLEstimator(estimand, config=double_rl_config).estimate(
    dataset
)

clipped_cfg = DiagnosticsConfig(max_weight=10.0)
is_clipped = ISEstimator(estimand, diagnostics_config=clipped_cfg).estimate(dataset)

(
    double_rl_report.to_dataframe(),
    is_clipped.to_dataframe(),
    isinstance(ISEstimator(estimand), OPEEstimator),
    isinstance(is_clipped, EstimatorReport),
)

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
