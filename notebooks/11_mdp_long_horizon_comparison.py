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
# # Long-Horizon MDP OPE Comparison
#
# This notebook highlights how different estimators behave as horizon grows:
#
# - IS/PDIS can explode with weak overlap.
# - MIS and density-ratio methods stabilize long-horizon estimates.
# - Model-based estimators (FQE, DRL) can be competitive when realizability holds.

# %%
import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import (
    MARKOV,
    OVERLAP,
    Q_MODEL_REALIZABLE,
    SEQUENTIAL_IGNORABILITY,
)
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.utils import compute_action_probs
from crl.ope import evaluate
from crl.viz import configure_notebook_display, save_figure

# %% [markdown]
# ## 1) Generate a long-horizon synthetic MDP
#
# Longer horizons magnify variance in importance-weighted estimators, which is
# why this setting is useful for comparing MIS/DICE-style methods.

# %% [markdown]
# ## 2) Define the estimand

# %%
configure_notebook_display()
bench = SyntheticMDP(SyntheticMDPConfig(seed=3, horizon=10, discount=0.95))
dataset = bench.sample(num_trajectories=500, seed=4)
dataset.describe()

# %%
estimand = PolicyValueEstimand(
    policy=bench.target_policy,
    discount=dataset.discount,
    horizon=dataset.horizon,
    assumptions=AssumptionSet(
        [SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV, Q_MODEL_REALIZABLE]
    ),
)

# %% [markdown]
# ## 3) Run a diverse estimator suite

# %%
report = evaluate(
    dataset=dataset,
    policy=bench.target_policy,
    estimand=estimand,
    estimators=["is", "pdis", "mis", "wdr", "fqe", "dualdice", "gendice", "drl"],
)
summary = report.to_dataframe()
summary

# %%
print(
    summary[["estimator", "value", "lower_bound", "upper_bound"]]
    .round(3)
    .to_string(index=False)
)

# %% [markdown]
# ## Interpretation notes
#
# - Density-ratio methods can be unstable when overlap is weak or sample sizes
#   are small; tune regularization and inspect diagnostics.
# - Compare IS/PDIS to MIS/DICE to see how variance grows with horizon.

# %% [markdown]
# ## 4) Compare against ground truth

# %%

true_value = bench.true_policy_value(bench.target_policy)
print(f"True policy value: {true_value:.3f}")
fig_comp = report.plot_estimator_comparison(truth=true_value)
fig_comp

# %% [markdown]
# ## 5) Weight diagnostics (trajectory IS)

# %%

target_probs = compute_action_probs(bench.target_policy, dataset.observations, dataset.actions)
ratios = np.where(dataset.mask, target_probs / dataset.behavior_action_probs, 1.0)
weights = np.prod(ratios, axis=1)
fig_weights = report.plot_importance_weights(weights)
fig_weights

# %% [markdown]
# ## 6) ESS by time step

# %%

fig_ess = report.plot_effective_sample_size(np.cumprod(ratios, axis=1), by_time=True)
fig_ess

# %% [markdown]
# ## 7) Export figures for the docs

# %%
save_figure(fig_comp, "docs/assets/figures/mdp_long_horizon_estimator_comparison")
save_figure(fig_weights, "docs/assets/figures/mdp_long_horizon_weights")
save_figure(fig_ess, "docs/assets/figures/mdp_long_horizon_ess")
