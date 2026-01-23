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
# # Bandit OPE End-to-End
#
# This notebook demonstrates a research-grade workflow:
#
# 1. Define an explicit estimand and assumptions.
# 2. Run multiple estimators with diagnostics.
# 3. Compare against ground truth from a synthetic benchmark.
# 4. Quantify sensitivity to unobserved confounding.

# %%
import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import BOUNDED_CONFOUNDING, OVERLAP, SEQUENTIAL_IGNORABILITY, BEHAVIOR_POLICY_KNOWN
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimands.sensitivity_policy_value import SensitivityPolicyValueEstimand
from crl.ope import evaluate
from crl.viz import configure_notebook_display, save_figure

# %% [markdown]
# ## 1) Generate a synthetic logged bandit dataset

# %%
bench = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = bench.sample(num_samples=500, seed=1)
dataset.describe()
configure_notebook_display()

# %% [markdown]
# ## 2) Define the estimand (with assumptions)

# %%

estimand = PolicyValueEstimand(
    policy=bench.target_policy,
    discount=1.0,
    horizon=1,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN]),
)

# %% [markdown]
# ## 3) Run estimators + diagnostics

# %%

sensitivity = SensitivityPolicyValueEstimand(
    policy=bench.target_policy,
    discount=1.0,
    horizon=1,
    gammas=np.linspace(1.0, 2.0, 6),
    assumptions=AssumptionSet([BOUNDED_CONFOUNDING]),
)

report = evaluate(
    dataset=dataset,
    policy=bench.target_policy,
    estimand=estimand,
    estimators=["is", "wis", "double_rl"],
    sensitivity=sensitivity,
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
# ## 4) Compare to ground truth

# %%

true_value = bench.true_policy_value(bench.target_policy)
print(f"True policy value: {true_value:.3f}")
fig_comp = report.plot_estimator_comparison(truth=true_value)
fig_comp

# %% [markdown]
# ## 5) Weight diagnostics

# %%

target_probs = bench.target_policy.action_prob(dataset.contexts, dataset.actions)
weights = target_probs / dataset.behavior_action_probs
fig_weights = report.plot_importance_weights(weights)
fig_weights

# %% [markdown]
# ## 6) Sensitivity bounds

# %%

report.figures.get("sensitivity_bounds")

# %% [markdown]
# ## 7) Export figures for the docs

# %%
save_figure(fig_comp, "docs/assets/figures/bandit_end_to_end_estimator_comparison")
save_figure(fig_weights, "docs/assets/figures/bandit_end_to_end_weights")
