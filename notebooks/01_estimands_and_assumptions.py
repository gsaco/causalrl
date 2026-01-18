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

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.importance_sampling import ISEstimator
from crl.utils.seeding import set_seed

# %%
set_seed(0)
np.random.seed(0)

# %% [markdown]
# ## Define an estimand
#
# The policy value estimand encodes the target policy, horizon, discount, and
# identification assumptions. Estimators check these assumptions before
# running.

# %%
benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = benchmark.sample(num_samples=500, seed=1)

estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=1.0,
    horizon=1,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
)

estimand

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
# ## Takeaways
#
# - Estimands make assumptions explicit and enforceable.
# - Diagnostics are tied to assumptions (e.g., overlap).
# - When overlap is poor, estimators warn and flag the assumption.
