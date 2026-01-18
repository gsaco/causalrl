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
# # 06 — Sensitivity to Unobserved Confounding
#
# We compute sequential sensitivity bounds for OPE under a Gamma-bounded
# confounding model. The bounds widen as Gamma increases.

# %% [markdown]
# ## Setup
#
# ```
# pip install "causalrl[plots]"
# ```

# %%
from __future__ import annotations

import numpy as np

from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.sensitivity.namkoong2020 import confounded_ope_bounds
from crl.utils.seeding import set_seed

# %%
set_seed(0)
np.random.seed(0)

# %% [markdown]
# ## Compute sensitivity bounds

# %%
benchmark = SyntheticMDP(SyntheticMDPConfig(seed=0, horizon=5))
dataset = benchmark.sample(num_trajectories=200, seed=1)

gammas = np.array([1.0, 1.25, 1.5, 2.0])
curve = confounded_ope_bounds(dataset, benchmark.target_policy, gammas)
curve.to_dict()

# %% [markdown]
# ## Takeaways
#
# - Gamma controls the strength of unobserved confounding.
# - Larger Gamma yields wider (more conservative) bounds.
# - Sensitivity analysis complements standard OPE estimates.
