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
# # 04 — Confidence Intervals and HCOPE
#
# We demonstrate bootstrap confidence intervals and a high-confidence lower
# bound (HCOPE) for bandit OPE.

# %% [markdown]
# ## Setup
#
# ```
# pip install "causalrl[plots]"
# ```

# %%
from __future__ import annotations

import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import BOUNDED_REWARDS, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.bootstrap import BootstrapConfig, bootstrap_ci
from crl.estimators.high_confidence import HighConfidenceISEstimator
from crl.estimators.importance_sampling import ISEstimator
from crl.utils.seeding import set_seed

# %%
set_seed(0)
np.random.seed(0)

# %% [markdown]
# ## Bootstrap CI for IS

# %%
benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = benchmark.sample(num_samples=1_000, seed=1)

estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=1.0,
    horizon=1,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BOUNDED_REWARDS]),
)

is_estimator = ISEstimator(estimand)
bootstrap_cfg = BootstrapConfig(num_bootstrap=200, method="trajectory", alpha=0.05, seed=0)
stderr, ci = bootstrap_ci(lambda: ISEstimator(estimand), dataset, bootstrap_cfg)
stderr, ci

# %% [markdown]
# ## High-confidence lower bound (HCOPE)
#
# HCOPE produces a lower bound that holds with probability `1 - delta` under
# bounded rewards.

# %%
hcope_report = HighConfidenceISEstimator(estimand).estimate(dataset)
hcope_report.value, hcope_report.ci

# %% [markdown]
# ## Takeaways
#
# - Bootstrap CIs provide a general uncertainty estimate.
# - HCOPE yields a conservative lower bound with explicit guarantees.
