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
# # Sensitivity Analysis for Bounded Confounding
#
# This notebook demonstrates bandit propensity sensitivity analysis. The
# sensitivity curve reports lower/upper bounds on the policy value as we relax
# the unconfoundedness assumption with a multiplicative gamma parameter.

# %%
from __future__ import annotations

from pprint import pprint

import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import BOUNDED_CONFOUNDING, OVERLAP
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.sensitivity.bandit import BanditPropensitySensitivity

# %%
np.random.seed(0)

benchmark = SyntheticBandit(SyntheticBanditConfig(seed=21))
dataset = benchmark.sample(num_samples=1_000, seed=22)

estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=1.0,
    horizon=1,
    assumptions=AssumptionSet([OVERLAP, BOUNDED_CONFOUNDING]),
)

sensitivity = BanditPropensitySensitivity(estimand)

# %%
# gamma = 1.0 corresponds to no unobserved confounding.

gammas = np.linspace(1.0, 3.0, 5)
curve = sensitivity.curve(dataset, gammas)

summary = [
    {
        "gamma": float(gamma),
        "lower": float(lower),
        "upper": float(upper),
    }
    for gamma, lower, upper in zip(curve.gammas, curve.lower, curve.upper, strict=False)
]

pprint(summary)
