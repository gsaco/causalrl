# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bandit Off-Policy Evaluation (OPE)
#
# This quickstart notebook walks through bandit OPE with the synthetic benchmark.
# We compare IS and WIS estimates, inspect diagnostics, and keep runtimes small
# with deterministic seeds.

# %%
from __future__ import annotations

from pprint import pprint

import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.importance_sampling import ISEstimator, WISEstimator

# %%
np.random.seed(0)

benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = benchmark.sample(num_samples=1_000, seed=1)
true_value = benchmark.true_policy_value(benchmark.target_policy)

estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=1.0,
    horizon=1,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
)

estimators = [ISEstimator(estimand), WISEstimator(estimand)]

# %%
rows = []
for estimator in estimators:
    report = estimator.estimate(dataset)
    rows.append(
        {
            "estimator": report.metadata["estimator"],
            "estimate": report.value,
            "stderr": report.stderr,
            "true_value": true_value,
            "ess_ratio": report.diagnostics["ess"]["ess_ratio"],
            "max_weight": report.diagnostics["weights"]["max"],
            "warnings": report.warnings,
        }
    )

pprint(rows)
