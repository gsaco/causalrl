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
# # Diagnostics and Overlap Checks
#
# Diagnostics help flag weak overlap and heavy-tailed weights before trusting
# off-policy estimates. This notebook computes overlap metrics, ESS, and
# weight-tail summaries from a synthetic bandit dataset.

# %%
from __future__ import annotations

from pprint import pprint

import numpy as np

from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.diagnostics.ess import effective_sample_size, ess_ratio
from crl.diagnostics.overlap import compute_overlap_metrics
from crl.diagnostics.plots import plot_ratio_histogram, plot_weight_histogram
from crl.diagnostics.weights import weight_tail_stats

# %%
np.random.seed(0)

benchmark = SyntheticBandit(SyntheticBanditConfig(seed=7))
dataset = benchmark.sample(num_samples=1_000, seed=11)

# %%
# Importance weights for the target policy on logged actions.

target_probs = benchmark.target_policy.action_prob(dataset.contexts, dataset.actions)
behavior_probs = dataset.behavior_action_probs
ratios = target_probs / behavior_probs

metrics = {
    "overlap": compute_overlap_metrics(target_probs, behavior_probs, threshold=1e-3),
    "ess": effective_sample_size(ratios),
    "ess_ratio": ess_ratio(ratios),
    "weight_tail": weight_tail_stats(ratios, quantile=0.99, threshold=10.0),
}

pprint(metrics)

# %%
# Optional plots (requires matplotlib).
plot_weight_histogram(ratios, bins=40)
plot_ratio_histogram(ratios, bins=40)
