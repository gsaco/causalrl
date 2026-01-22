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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 — Estimands and Assumptions
#
# CausalRL requires you to be explicit about **what** you estimate (the estimand)
# and **which assumptions** identify that estimand. This notebook demonstrates:
#
# - How to define a `PolicyValueEstimand`.
# - How assumptions are enforced by estimators.
# - How diagnostics flag overlap violations.

# %% [markdown]
# ## Setup
#
# Suggested environment:
#
# ```
# pip install "causalrl[plots]"
# ```

# %%
from __future__ import annotations

import numpy as np

from crl.assumptions import Assumption, AssumptionSet
from crl.assumptions_catalog import (
    CORRECT_MODEL,
    MARKOV,
    OVERLAP,
    SEQUENTIAL_IGNORABILITY,
)
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.diagnostics.plots import plot_weight_histogram
from crl.estimands.policy_value import PolicyContrastEstimand, PolicyValueEstimand
from crl.estimators.importance_sampling import ISEstimator
from crl.policies import UniformPolicy
from crl.utils.seeding import set_seed
from crl.viz import configure_notebook_display

# %%
set_seed(0)
np.random.seed(0)
configure_notebook_display()

# %% [markdown]
# ## Custom assumptions
#
# Assumptions are explicit, typed objects. You can add your own and mix them
# with catalog defaults.

# %%
benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = benchmark.sample(num_samples=500, seed=1)

stationarity = Assumption(
    name="stationarity",
    description="The environment dynamics do not drift during collection.",
)

assumptions = AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, stationarity])
assumptions.to_dict()

# %% [markdown]
# The catalog entries are reusable primitives for building checklists.

# %%
AssumptionSet([MARKOV]).names()

# %% [markdown]
# ## Define an estimand
#
# The policy value estimand encodes the target policy, horizon, discount, and
# identification assumptions. Estimators check these assumptions before
# running.

# %%
estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=1.0,
    horizon=1,
    assumptions=assumptions,
)

estimand

# %% [markdown]
# ## Policy contrast estimand
#
# A contrast estimand expresses the difference between two policy values.

# %%
control_policy = UniformPolicy(action_space_n=dataset.action_space_n)
control_estimand = PolicyValueEstimand(
    policy=control_policy,
    discount=1.0,
    horizon=1,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
)

contrast = PolicyContrastEstimand(treatment=estimand, control=control_estimand)
contrast.to_dict()

# %% [markdown]
# Model-based estimators typically add a correct-model assumption.

# %%
AssumptionSet([CORRECT_MODEL]).names()

# %% [markdown]
# ## Overlap violations
#
# If the logging policy never takes some actions that the target policy would
# take, importance-weighted estimators become unstable. We'll create a toy
# overlap violation and inspect the diagnostics.

# %%
dataset_bad = benchmark.sample(num_samples=500, seed=2)
dataset_bad.behavior_action_probs = np.clip(
    dataset_bad.behavior_action_probs, 0.0, 0.02
)

report = ISEstimator(estimand).estimate(dataset_bad)
report.diagnostics

# %% [markdown]
# ## Results snapshot
#
# The overlap stress test produces large weights. We summarize the estimate and
# visualize the weight distribution.

# %%
summary = report.to_dataframe()
print(
    summary[["value", "lower_bound", "upper_bound"]]
    .round(3)
    .to_string(index=False)
)
summary

# %%
weights = (
    benchmark.target_policy.action_prob(dataset_bad.contexts, dataset_bad.actions)
    / dataset_bad.behavior_action_probs
)
fig_weights = plot_weight_histogram(
    weights,
    log_y=True,
    title="Weight histogram under overlap stress",
    clip_quantile=0.995,
)
fig_weights

# %% [markdown]
# ## Takeaways
#
# - Estimands make assumptions explicit and enforceable.
# - Diagnostics are tied to assumptions (e.g., overlap).
# - When overlap is poor, estimators warn and flag the assumption.
