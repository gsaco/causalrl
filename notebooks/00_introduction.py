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
# # 00 - Introduction: Estimand-First OPE
#
# This notebook is a fast, narrative tour of CausalRL. The key idea is
# estimand-first OPE: you declare the estimand and its assumptions, then every
# estimator and report is obligated to surface diagnostics that tell you whether
# the assumptions look plausible in the data.
#
# We run a small synthetic bandit experiment (so we know ground truth), inspect
# the report schema, and export a self-contained HTML report. Along the way we
# touch the public APIs for policies, data contracts, benchmarks, and
# experiment runners.

# %% [markdown]
# ## Setup
#
# Suggested environment:
#
# ```
# pip install "causalrl[plots]"
# ```
#
# (You can add `[docs]` or `[notebooks]` if you want full notebook tooling.)

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import crl
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.benchmarks.harness import (
    run_all_benchmarks,
    run_bandit_benchmark,
    run_mdp_benchmark,
)
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.data import (
    BanditDataset,
    LoggedBanditDataset,
    TrajectoryDataset,
    TransitionDataset,
)
from crl.experiments import run_benchmark_suite, run_benchmarks_to_table
from crl.ope import OpeReport, evaluate
from crl.policies import (
    BehaviorPolicy,
    MLPConfig,
    Policy,
    StochasticPolicy,
    TabularPolicy,
    TorchMLPPolicy,
    UniformPolicy,
)
from crl.utils.seeding import set_seed
from crl.utils.validation import (
    require_finite,
    require_in_unit_interval,
    require_ndarray,
    require_same_length,
    require_shape,
)
from crl.viz import configure_notebook_display

# %%
set_seed(0)
np.random.seed(0)
configure_notebook_display()

print("crl", crl.__version__)

# %% [markdown]
# ## Synthetic data for the tour
#
# We use the built-in synthetic bandit benchmark as a common source of logged
# data for policy and estimator demos.

# %%
benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = benchmark.sample(num_samples=1_000, seed=1)
true_value = benchmark.true_policy_value(benchmark.target_policy)

pd.DataFrame([dataset.describe()])

# %% [markdown]
# ## Policy interfaces
#
# Policies expose action probabilities, log-probabilities, and sampling. Below
# we create uniform, stochastic, tabular, torch-backed, and behavior policy
# wrappers and display a snapshot of their action probabilities.

# %%
rng = np.random.default_rng(0)
sample_contexts = dataset.contexts[:6]

uniform_policy = UniformPolicy(action_space_n=dataset.action_space_n)


def softmax_probs(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs)
    obs_2d = obs if obs.ndim == 2 else obs.reshape(-1, 1)
    logits = np.zeros((obs_2d.shape[0], dataset.action_space_n), dtype=float)
    logits[:, 0] = obs_2d[:, 0]
    if dataset.action_space_n > 1:
        logits[:, 1] = -obs_2d[:, 0]
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


stochastic_policy = StochasticPolicy(
    prob_fn=softmax_probs, action_space_n=dataset.action_space_n, name="softmax_demo"
)

# Tabular policies require discrete state indices.
# We use a tiny two-state example to illustrate the interface.
tabular_policy = TabularPolicy(np.array([[0.7, 0.3], [0.2, 0.8]]))
tabular_states = np.array([0, 1, 0, 1, 1])

mlp_policy = TorchMLPPolicy.from_config(
    MLPConfig(
        input_dim=sample_contexts.shape[1] if sample_contexts.ndim > 1 else 1,
        action_dim=dataset.action_space_n,
        hidden_sizes=(16, 16),
        activation="tanh",
    )
)

behavior_policy = BehaviorPolicy(
    policy=uniform_policy, source="known", diagnostics={"note": "demo"}
)

sample_actions = uniform_policy.sample_action(sample_contexts[:4], rng)
log_probs = uniform_policy.log_prob(sample_contexts[:4], sample_actions)

policy_rows = [
    {
        "policy": "uniform",
        "sample_probs": uniform_policy.action_probs(sample_contexts[:1])[0],
    },
    {
        "policy": "stochastic",
        "sample_probs": stochastic_policy.action_probs(sample_contexts[:1])[0],
    },
    {
        "policy": "tabular",
        "sample_probs": tabular_policy.action_probs(tabular_states[:1])[0],
    },
    {
        "policy": "torch_mlp",
        "sample_probs": mlp_policy.action_probs(sample_contexts[:1])[0],
    },
]

pd.DataFrame(policy_rows), behavior_policy.to_dict(), log_probs[:3]

# %%
isinstance(uniform_policy, Policy)

# %% [markdown]
# ## Data contracts and validation helpers
#
# CausalRL ships data classes for logged bandits, trajectories, and transitions.
# We also provide validation helpers for shapes and probability bounds.

# %%
manual_contexts = np.random.normal(size=(6, 2))
manual_actions = np.array([0, 1, 0, 1, 0, 1])
manual_rewards = np.random.normal(size=6)
manual_behavior_probs = np.full(6, 0.5)

require_ndarray("contexts", manual_contexts)
require_shape("actions", manual_actions, 1)
require_same_length(
    ["contexts", "actions", "rewards"],
    [manual_contexts, manual_actions, manual_rewards],
)
require_finite("rewards", manual_rewards)
require_in_unit_interval("behavior_action_probs", manual_behavior_probs)

manual_dataset = LoggedBanditDataset(
    contexts=manual_contexts,
    actions=manual_actions,
    rewards=manual_rewards,
    behavior_action_probs=manual_behavior_probs,
    action_space_n=2,
    metadata={"source": "manual_demo"},
)

roundtrip = LoggedBanditDataset.from_dict(manual_dataset.to_dict())

pd.DataFrame([roundtrip.describe()]), isinstance(roundtrip, BanditDataset)

# %%
mdp_benchmark = SyntheticMDP(SyntheticMDPConfig(seed=5, horizon=4))
traj_dataset = mdp_benchmark.sample(num_trajectories=5, seed=6)

mask_flat = traj_dataset.mask.reshape(-1)
obs_flat = traj_dataset.observations.reshape(
    -1, *traj_dataset.observations.shape[2:]
)[mask_flat]
next_obs_flat = traj_dataset.next_observations.reshape(
    -1, *traj_dataset.next_observations.shape[2:]
)[mask_flat]
actions_flat = traj_dataset.actions.reshape(-1)[mask_flat]
rewards_flat = traj_dataset.rewards.reshape(-1)[mask_flat]
dones_flat = traj_dataset.dones.reshape(-1)[mask_flat]
behavior_probs_flat = (
    traj_dataset.behavior_action_probs.reshape(-1)[mask_flat]
    if traj_dataset.behavior_action_probs is not None
    else None
)

episode_ids = np.repeat(
    np.arange(traj_dataset.num_trajectories), traj_dataset.horizon
)[mask_flat]
timesteps = np.tile(np.arange(traj_dataset.horizon), traj_dataset.num_trajectories)[
    mask_flat
]

transition_dataset = TransitionDataset(
    states=obs_flat,
    actions=actions_flat,
    rewards=rewards_flat,
    next_states=next_obs_flat,
    dones=dones_flat,
    behavior_action_probs=behavior_probs_flat,
    discount=traj_dataset.discount,
    action_space_n=traj_dataset.action_space_n,
    episode_ids=episode_ids,
    timesteps=timesteps,
    metadata={"source": "flattened_from_trajectory"},
)

trajectory_roundtrip = transition_dataset.to_trajectory()

(
    pd.DataFrame([transition_dataset.describe()]),
    pd.DataFrame([trajectory_roundtrip.describe()]),
    isinstance(trajectory_roundtrip, TrajectoryDataset),
)

# %% [markdown]
# ## Quick OPE run
#
# We evaluate a target policy and compare estimators in a single call. The
# resulting report is a structured object that can serialize to a DataFrame or
# HTML.

# %%
report = evaluate(dataset=dataset, policy=benchmark.target_policy)
report.summary_table(), isinstance(report, OpeReport)

# %%
from crl.core import EstimationReport

first_report = next(iter(report.reports.values()))
isinstance(first_report, EstimationReport)

# %% [markdown]
# ## Visualization helpers
#
# OpeReport includes convenience plotting methods for estimator comparisons and
# weight diagnostics.

# %%
fig_comparison = report.plot_estimator_comparison(truth=true_value)
fig_comparison

# %%
weights = (
    benchmark.target_policy.action_prob(dataset.contexts, dataset.actions)
    / dataset.behavior_action_probs
)
fig_weights = report.plot_importance_weights(weights, logy=True)
fig_weights

# %%
fig_ess = report.plot_effective_sample_size(weights, by_time=False)
fig_ess

# %% [markdown]
# ## Report schema and HTML export
#
# Every estimator returns the same schema: point estimate, uncertainty
# (stderr/CI), diagnostics, and assumption flags. This makes it easy to compare
# methods side-by-side and to automate downstream checks.

# %%
output_dir = Path("docs/assets/reports")
output_dir.mkdir(parents=True, exist_ok=True)
report_path = output_dir / "intro_bandit_report.html"
report.save_html(str(report_path))
report_path

# %% [markdown]
# ## Benchmarks and experiment runners
#
# The benchmark harness runs synthetic bandit and MDP suites with known ground
# truth. Experiment helpers write CSV/JSONL outputs and aggregate tables.

# %%
bandit_results = run_bandit_benchmark(num_samples=300, seed=0)
mdp_results = run_mdp_benchmark(num_trajectories=80, seed=0)
all_results = run_all_benchmarks(num_samples=300, num_trajectories=80, seed=0)

pd.DataFrame(all_results)

# %%
bench_out = Path("docs/assets/benchmarks/intro")
bench_out.mkdir(parents=True, exist_ok=True)
bench_records = run_benchmarks_to_table(
    output_dir=bench_out, num_samples=200, num_trajectories=60, seed=0
)

pd.DataFrame(bench_records).head()

# %%
# A tiny custom suite to exercise run_benchmark_suite without long runtimes.
custom_suite_dir = bench_out / "suite_configs"
custom_suite_dir.mkdir(parents=True, exist_ok=True)
custom_suite = {
    "suite": "intro_demo",
    "benchmarks": [
        {
            "name": "bandit_tiny",
            "type": "bandit",
            "num_samples": 200,
            "estimators": ["is", "wis"],
            "behavior_known": True,
        }
    ],
}
(custom_suite_dir / "intro_demo.yaml").write_text(
    yaml.safe_dump(custom_suite), encoding="utf-8"
)

suite_df = run_benchmark_suite(
    suite="intro_demo",
    output_dir=bench_out / "suite_run",
    seeds=[0],
    config_dir=custom_suite_dir,
)

suite_df.head()

# %% [markdown]
# ## Takeaways
#
# - Policies, data contracts, and estimands form the backbone of OPE workflows.
# - Reports standardize metrics, diagnostics, and figures across estimators.
# - Benchmarks and experiment runners provide reproducible comparisons.
# - HTML export creates shareable artifacts for reviews.
