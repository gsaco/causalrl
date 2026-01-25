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
# # 05 — Advanced DR Family (WDR, MAGIC, MRDR)
#
# This notebook compares DR-family estimators that blend model-based and
# importance-weighted signals to reduce variance.

# %% [markdown]
# ## Setup
#
# ```
# pip install ".[plots]"
# ```

# %%
from __future__ import annotations

import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY, BEHAVIOR_POLICY_KNOWN
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.dr import DRCrossFitConfig, DoublyRobustEstimator
from crl.estimators.magic import MAGICConfig, MAGICEstimator
from crl.estimators.mrdr import MRDRConfig, MRDREstimator
from crl.estimators.wdr import WDRConfig, WeightedDoublyRobustEstimator
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

# %%
summary = report.summary_table()
print(
    summary[["estimator", "value", "lower_bound", "upper_bound"]]
    .round(3)
    .to_string(index=False)
)

# %%
fig = report.plot_estimator_comparison(truth=true_value)
fig

# %% [markdown]
# ## Custom DR-family configuration
#
# Each estimator exposes a config object for cross-fitting, ridge strengths, and
# mixing parameters. We instantiate them directly to show the controls.

# %%
estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=dataset.discount,
    horizon=dataset.horizon,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN, MARKOV]),
)

dr_config = DRCrossFitConfig(num_folds=3, num_iterations=3, ridge=5e-3, seed=0)
wdr_config = WDRConfig(num_folds=3, num_iterations=3, ridge=5e-3, seed=0)
magic_config = MAGICConfig(num_iterations=4, ridge=1e-3, mixing_horizons=(0, 2, 4))
mrdr_config = MRDRConfig(num_folds=3, num_iterations=3, ridge=5e-3, seed=0)

custom_estimators = [
    DoublyRobustEstimator(estimand, config=dr_config),
    WeightedDoublyRobustEstimator(estimand, config=wdr_config),
    MAGICEstimator(estimand, config=magic_config),
    MRDREstimator(estimand, config=mrdr_config),
]

rows = []
for estimator in custom_estimators:
    est_report = estimator.estimate(dataset)
    rows.append(
        {
            "estimator": est_report.metadata["estimator"],
            "value": est_report.value,
            "stderr": est_report.stderr,
            "ci": est_report.ci,
        }
    )

rows

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
