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
# # 08 — Estimator Selection and Debugging
#
# We show a diagnostics-driven estimator selection workflow and a debugging
# checklist when overlap or model fit is poor.

# %% [markdown]
# ## Setup
#
# ```
# pip install "causalrl[plots]"
# ```

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.selectors import SelectionResult, select_estimator
from crl.utils.seeding import set_seed

# %%
set_seed(0)
np.random.seed(0)

# %% [markdown]
# ## Run selection
#
# We use a heuristic score that favors stable importance weights and reasonable
# model fit.

# %%
benchmark = SyntheticMDP(SyntheticMDPConfig(seed=0, horizon=5))
dataset = benchmark.sample(num_trajectories=200, seed=1)

estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=dataset.discount,
    horizon=dataset.horizon,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV]),
)

selection = select_estimator(
    dataset,
    estimand,
    candidates=["is", "wis", "pdis", "dr", "wdr", "mrdr", "fqe"],
    return_scores=True,
)
selection.best, isinstance(selection, SelectionResult)

# %%
score_df = pd.DataFrame(selection.scores).sort_values("score", ascending=False)
score_df

# %%
print(f"Recommended estimator: {selection.best}")

# %%
fig, ax = plt.subplots(figsize=(5.2, 2.8))
ax.bar(score_df["estimator"], score_df["score"], color="0.35")
ax.set_ylabel("Selection score")
ax.set_title("Estimator selection scores")
ax.set_ylim(bottom=0.0)
ax.tick_params(axis="x", rotation=25)
fig.tight_layout()
fig

# %% [markdown]
# ## Debug playbook
#
# - **Overlap bad** → inspect ESS and weight tails, consider WIS/DR, or collect
#   more coverage.
# - **Model fit bad** → check Q-model MSE, increase model capacity, or switch to
#   IS-based estimators.
# - **Propensities unknown** → estimate behavior policy or use model-based OPE.

# %% [markdown]
# ## Takeaways
#
# - Estimator selection is heuristic, but diagnostics make it principled.
# - Always triangulate with multiple estimators and failure-mode checks.
