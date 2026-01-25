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
# # 04 — Confidence Intervals and HCOPE
#
# We demonstrate bootstrap confidence intervals and a high-confidence lower
# bound (HCOPE) for bandit OPE.

# %% [markdown]
# ## Setup
#
# ```
# pip install ".[plots]"
# ```

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import BOUNDED_REWARDS, OVERLAP, SEQUENTIAL_IGNORABILITY, BEHAVIOR_POLICY_KNOWN
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.bootstrap import BootstrapConfig, bootstrap_ci
from crl.estimators.high_confidence import (
    HighConfidenceConfig,
    HighConfidenceISEstimator,
)
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
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN, BOUNDED_REWARDS]),
)

is_estimator = ISEstimator(estimand)
bootstrap_cfg = BootstrapConfig(
    num_bootstrap=200, method="trajectory", alpha=0.05, seed=0
)
stderr, ci = bootstrap_ci(lambda: ISEstimator(estimand), dataset, bootstrap_cfg)
stderr, ci

# %%
is_report = is_estimator.estimate(dataset)
print(
    "IS estimate: "
    f"{is_report.value:.3f} | bootstrap CI=({ci[0]:.3f}, {ci[1]:.3f})"
)

# %% [markdown]
# ## High-confidence lower bound (HCOPE)
#
# HCOPE produces a lower bound that holds with probability `1 - delta` under
# bounded rewards.

# %%
hcope_report = HighConfidenceISEstimator(estimand).estimate(dataset)
hcope_report.value, hcope_report.ci

# %%
print(
    "HCOPE lower bound: "
    f"{hcope_report.value:.3f} | implied upper={hcope_report.ci[1]:.3f}"
)

# %% [markdown]
# ## Explicit HCOPE configuration
#
# When you know a reward bound, pass it explicitly for tighter, reliable bounds.

# %%
hcope_config = HighConfidenceConfig(delta=0.1, reward_bound=2.0)
hcope_report_cfg = HighConfidenceISEstimator(estimand, config=hcope_config).estimate(
    dataset
)
hcope_report_cfg.to_dataframe()

# %% [markdown]
# ## Visual comparison
#
# Compare a two-sided bootstrap CI with a one-sided high-confidence lower bound.

# %%
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.errorbar(
    [0],
    [is_report.value],
    yerr=[[is_report.value - ci[0]], [ci[1] - is_report.value]],
    fmt="o",
    capsize=4,
    color="tab:blue",
    label="IS (bootstrap CI)",
)
ax.errorbar(
    [1],
    [hcope_report.value],
    yerr=[[0.0], [hcope_report.ci[1] - hcope_report.value]],
    fmt="o",
    capsize=4,
    color="tab:orange",
    label="HCOPE (lower bound)",
)
ax.set_xticks([0, 1], ["IS", "HCOPE"])
ax.set_ylabel("Estimated policy value")
ax.set_title("CI vs. high-confidence bound")
ax.legend(frameon=False)
fig.tight_layout()
fig

# %% [markdown]
# ## Takeaways
#
# - Bootstrap CIs provide a general uncertainty estimate.
# - HCOPE yields a conservative lower bound with explicit guarantees.
