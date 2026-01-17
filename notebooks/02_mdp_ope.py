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
# # MDP Off-Policy Evaluation (OPE)
#
# This notebook evaluates a target policy in a small synthetic MDP. We compare
# IS, WIS, PDIS, DR, and FQE estimators against the ground-truth value.

# %%
from __future__ import annotations

from pprint import pprint

import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.dr import DoublyRobustEstimator
from crl.estimators.fqe import FQEEstimator
from crl.estimators.importance_sampling import ISEstimator, PDISEstimator, WISEstimator

# %%
np.random.seed(0)

benchmark = SyntheticMDP(SyntheticMDPConfig(seed=0, horizon=5))
dataset = benchmark.sample(num_trajectories=200, seed=1)
true_value = benchmark.true_policy_value(benchmark.target_policy)

estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=dataset.discount,
    horizon=dataset.horizon,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV]),
)

estimators = [
    ISEstimator(estimand),
    WISEstimator(estimand),
    PDISEstimator(estimand),
    DoublyRobustEstimator(estimand),
    FQEEstimator(estimand),
]

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
            "warnings": report.warnings,
        }
    )

pprint(rows)
