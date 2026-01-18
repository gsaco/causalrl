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
# # 00 — Introduction: Estimand-First OPE
#
# This notebook is a fast, narrative tour of CausalRL. The key idea is
# **estimand-first OPE**: you declare the estimand and its assumptions, then
# every estimator and report is obligated to surface diagnostics that tell you
# whether the assumptions look plausible in the data.
#
# We'll run a small synthetic bandit experiment (so we know ground truth),
# inspect the report schema, and export a self-contained HTML report.

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

import crl
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.ope import evaluate
from crl.utils.seeding import set_seed
from crl.viz import configure_notebook_display

# %%
set_seed(0)
np.random.seed(0)
configure_notebook_display()

print("crl", crl.__version__)

# %% [markdown]
# ## Quick OPE Run
#
# We sample a logged bandit dataset, evaluate a target policy, and compare
# estimators in a single call. The report is a structured object that can
# serialize to a DataFrame or HTML.

# %%
benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = benchmark.sample(num_samples=1_000, seed=1)
true_value = benchmark.true_policy_value(benchmark.target_policy)

report = evaluate(dataset=dataset, policy=benchmark.target_policy)
report.summary_table()

# %% [markdown]
# ## Report Schema and HTML Export
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
# ## Takeaways
#
# - The report schema is standardized across estimators.
# - Diagnostics make assumption violations visible early.
# - HTML export creates shareable artifacts for reviews.
