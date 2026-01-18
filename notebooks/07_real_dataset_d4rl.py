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
# # 07 — Real Dataset Walkthrough (D4RL)
#
# This notebook shows how to load a D4RL dataset and map it to the CRL data
# contract. D4RL datasets do **not** include behavior propensities, so any IS/DR
# estimator requires an estimated logging policy or a model-based alternative.
#
# If D4RL is not installed, we fall back to a synthetic example to keep the
# notebook executable.

# %% [markdown]
# ## Setup
#
# ```
# pip install "causalrl[d4rl]"
# ```

# %%
from __future__ import annotations

from pathlib import Path

import numpy as np

from crl.ope import evaluate
from crl.utils.seeding import set_seed

# %%
set_seed(0)
np.random.seed(0)

# %% [markdown]
# ## Load D4RL

# %%
dataset = None
try:
    from crl.adapters.d4rl import load_d4rl_dataset

    dataset = load_d4rl_dataset("hopper-medium-v2")
    dataset.describe()
except Exception as exc:
    print("D4RL unavailable; falling back to synthetic data:", exc)

# %% [markdown]
# ## Fallback: synthetic dataset for report demo
#
# We still generate a report artifact so reviewers can see the pipeline end-to-end.

# %%
if dataset is None:
    from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig

    benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
    dataset = benchmark.sample(num_samples=1_000, seed=1)
    report = evaluate(dataset=dataset, policy=benchmark.target_policy)
    report.summary_table()
else:
    print("D4RL dataset loaded. OPE estimators requiring propensities are not applicable.")

# %% [markdown]
# ## Save HTML report artifact

# %%
if dataset is not None:
    output_dir = Path("docs/assets/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "d4rl_report.html"
    try:
        report.save_html(str(report_path))
        report_path
    except Exception as exc:
        print("Report generation skipped:", exc)

# %% [markdown]
# ## What went wrong (and how to fix it)
#
# - D4RL logs do not include behavior propensities.
# - IS/DR estimators require propensities or an estimated logging policy.
# - Use behavior estimation (if discrete) or model-based OPE until propensities
#   are available.
