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
# # 05 — Advanced DR Family (WDR, MAGIC, MRDR)
#
# This notebook compares DR-family estimators that blend model-based and
# importance-weighted signals to reduce variance.

# %% [markdown]
# ## Setup
#
# ```
# pip install "causalrl[plots]"
# ```

# %%
from __future__ import annotations

import numpy as np

from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.ope import evaluate
from crl.utils.seeding import set_seed

# %%
set_seed(0)
np.random.seed(0)

# %% [markdown]
# ## Run estimators

# %%
benchmark = SyntheticMDP(SyntheticMDPConfig(seed=0, horizon=5))
dataset = benchmark.sample(num_trajectories=200, seed=1)
true_value = benchmark.true_policy_value(benchmark.target_policy)

report = evaluate(
    dataset=dataset,
    policy=benchmark.target_policy,
    estimators=["dr", "wdr", "magic", "mrdr"],
)
report.summary_table()

# %% [markdown]
# ## Diagnostics
#
# Advanced DR estimators report weight normalization behavior and per-step ESS
# to help you understand variance tradeoffs.

# %%
report.diagnostics

# %% [markdown]
# ## Takeaways
#
# - WDR normalizes per-step weights to reduce variance.
# - MAGIC blends truncated estimators based on variance estimates.
# - MRDR trains the model component to minimize DR variance.
