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
# # 03 — MDP OPE Walkthrough
#
# We compare trajectory-based estimators on a synthetic MDP and interpret
# horizon effects and diagnostics.

# %% [markdown]
# ## Setup
#
# ```
# pip install ".[plots]"
# ```

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import (
    BEHAVIOR_POLICY_KNOWN,
    MARKOV,
    OVERLAP,
    Q_MODEL_REALIZABLE,
    SEQUENTIAL_IGNORABILITY,
)
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.dual_dice import DualDICEConfig, DualDICEEstimator
from crl.estimators.utils import compute_action_probs
from crl.ope import evaluate
from crl.utils.seeding import set_seed
from crl.viz import configure_notebook_display, save_figure

# %%
set_seed(0)
np.random.seed(0)
configure_notebook_display()

# %% [markdown]
# ## Run estimators

# %%
benchmark = SyntheticMDP(SyntheticMDPConfig(seed=0, horizon=5))
dataset = benchmark.sample(num_trajectories=200, seed=1)
true_value = benchmark.true_policy_value(benchmark.target_policy)

estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=dataset.discount,
    horizon=dataset.horizon,
    assumptions=AssumptionSet(
        [
            SEQUENTIAL_IGNORABILITY,
            OVERLAP,
            BEHAVIOR_POLICY_KNOWN,
            MARKOV,
            Q_MODEL_REALIZABLE,
        ]
    ),
)

report = evaluate(
    dataset=dataset,
    policy=benchmark.target_policy,
    estimand=estimand,
    estimators=["is", "wis", "pdis", "dr", "mis", "dualdice", "fqe"],
)
summary = report.summary_table()
summary

# %%
print(
    summary[["estimator", "value", "lower_bound", "upper_bound"]]
    .round(3)
    .to_string(index=False)
)

# %% [markdown]
# ## DualDICE configuration example
#
# DualDICE is a behavior-agnostic estimator for discrete MDPs. Here we use a
# custom ridge setting to illustrate configuration.

# %%
estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=dataset.discount,
    horizon=dataset.horizon,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV]),
)

dualdice_config = DualDICEConfig(ridge=5e-3, normalize=False)
dualdice_report = DualDICEEstimator(estimand, config=dualdice_config).estimate(dataset)
dualdice_report.to_dataframe()

# %% [markdown]
# ## Visual comparison

# %%
fig = report.plot_estimator_comparison(truth=true_value)
fig

# %%
target_probs = compute_action_probs(
    benchmark.target_policy, dataset.observations, dataset.actions
)
ratios = np.where(dataset.mask, target_probs / dataset.behavior_action_probs, 1.0)
fig_ess = report.plot_effective_sample_size(np.cumprod(ratios, axis=1), by_time=True)
fig_ess

# %% [markdown]
# ## Save figures for docs

# %%
output_dir = Path("docs/assets/figures")
output_dir.mkdir(parents=True, exist_ok=True)
save_figure(fig, output_dir / "mdp_walkthrough_estimator_comparison")

# %% [markdown]
# ## Takeaways
#
# - Horizon length amplifies importance-weight variance.
# - DR and FQE can reduce variance but introduce model bias.
# - Always inspect diagnostics alongside point estimates.
