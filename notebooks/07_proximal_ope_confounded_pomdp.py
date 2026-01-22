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
# # Advanced: Proximal OPE under Confounding
#
# This notebook demonstrates a confounded bandit where standard OPE is biased
# and a proximal estimator leverages proxy variables for robustness.

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.confounded_bandit import ConfoundedBandit, ConfoundedBanditConfig
from crl.confounding.proximal import (
    ProximalBanditDataset,
    ProximalBanditEstimator,
    ProximalConfig,
)
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.importance_sampling import ISEstimator
from crl.viz import configure_notebook_display, save_figure
from crl.viz.plots import plot_estimator_comparison

# %%
np.random.seed(0)
configure_notebook_display()

benchmark = ConfoundedBandit(ConfoundedBanditConfig(seed=7))
prox_data = benchmark.sample(num_samples=2_000, seed=8)
logged_data = prox_data.to_logged_dataset()
true_value = benchmark.true_policy_value(benchmark.target_policy)

estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=1.0,
    horizon=1,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
)

is_report = ISEstimator(estimand).estimate(logged_data)
prox_config = ProximalConfig(ridge=5e-3)
prox_report = ProximalBanditEstimator(
    benchmark.target_policy, config=prox_config
).estimate(prox_data)

isinstance(prox_data, ProximalBanditDataset), prox_data.to_dict().keys()

rows = [
    {"estimator": "IS", "value": is_report.value, "ci": is_report.ci},
    {"estimator": "Proximal", "value": prox_report, "ci": None},
]

results = pd.DataFrame(rows)
results["abs_error"] = (results["value"] - true_value).abs()
print(results.round(3).to_string(index=False))
results

# %%
fig = plot_estimator_comparison(rows, truth=true_value)
fig

# %%
output_dir = Path("docs/assets/figures")
output_dir.mkdir(parents=True, exist_ok=True)
save_figure(fig, output_dir / "proximal_confounded_bandit")
