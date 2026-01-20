# CausalRL

<p align="center">
  <img src="docs/assets/branding/causalrl-banner.svg" alt="CausalRL banner" width="100%" />
</p>

<p align="center">
  <strong>Estimand-first causal reinforcement learning & off-policy evaluation</strong><br />
  Assumptions in the open. Diagnostics by default. Reproducible benchmarks.
</p>

<p align="center">
  <a href="https://github.com/gsaco/causalrl/actions/workflows/ci.yml"><img src="https://github.com/gsaco/causalrl/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://gsaco.github.io/causalrl/"><img src="https://github.com/gsaco/causalrl/actions/workflows/pages.yml/badge.svg?branch=v4" alt="Docs" /></a>
  <a href="https://codecov.io/gh/gsaco/causalrl"><img src="https://codecov.io/gh/gsaco/causalrl/branch/main/graph/badge.svg" alt="Coverage" /></a>
  <a href="https://pypi.org/project/causalrl/"><img src="https://img.shields.io/badge/PyPI-coming%20soon-lightgrey" alt="PyPI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/gsaco/causalrl" alt="License" /></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python" /></a>
</p>

<p align="center">
  <a href="https://gsaco.github.io/causalrl/">Docs</a> |
  <a href="https://gsaco.github.io/causalrl/tutorials/">Tutorials</a> |
  <a href="examples/README.md">Examples</a> |
  <a href="docs/project_status.md">Project status</a> |
  <a href="CITATION.cff">Cite</a>
</p>

> Release status: v0.1.0 (research preview, alpha quality).
> PyPI package: `causalrl` | Import name: `crl`

If CausalRL helps your work, please consider starring the repo.

---

## At a glance

| Focus | What you get |
| --- | --- |
| Estimands & assumptions | A clear, auditable story for what you can identify and why. |
| Diagnostics-first | Overlap, ESS, and weight pathologies surfaced early. |
| Benchmarks | Synthetic bandit and MDP suites with ground-truth values. |
| Sensitivity | Bounded-confounding analysis for bandits and sequential settings. |
| Reports | Reproducible tables and exportable HTML artifacts. |

## Why CausalRL

- Estimand-first API: make causal guarantees explicit and traceable.
- Diagnostics as a first-class output, not an afterthought.
- Synthetic benchmarks for fast comparisons and regression tests.
- Clean data contracts for bandit, trajectory, and transition logs.

## When to use this library

- Offline evaluation when online deployment is risky, expensive, or impossible.
- Diagnostics-first iteration when you need to validate overlap and stability.
- Synthetic ground-truth settings to compare estimators and tune workflows.

---

## Installation

PyPI (when available):

```bash
python -m pip install causalrl
```

From source:

```bash
git clone https://github.com/gsaco/causalrl
cd causalrl
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install "causalrl[docs]"
python -m pip install "causalrl[benchmarks]"
python -m pip install "causalrl[notebooks]"
python -m pip install "causalrl[behavior]"
python -m pip install "causalrl[d4rl]"
python -m pip install "causalrl[rlu]"
```

---

## Quickstart

Bandit OPE:

```python
from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.importance_sampling import ISEstimator, WISEstimator

benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = benchmark.sample(num_samples=1000, seed=1)

estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=1.0,
    horizon=1,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
)

for estimator in [ISEstimator(estimand), WISEstimator(estimand)]:
    report = estimator.estimate(dataset)
    print(report.value, report.diagnostics)
```

MDP OPE:

```python
from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.dr import DoublyRobustEstimator
from crl.estimators.fqe import FQEEstimator
from crl.estimators.importance_sampling import ISEstimator, PDISEstimator, WISEstimator

benchmark = SyntheticMDP(SyntheticMDPConfig(seed=0))
dataset = benchmark.sample(num_trajectories=200, seed=1)

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

for estimator in estimators:
    report = estimator.estimate(dataset)
    print(report.value, report.diagnostics)
```

Runnable scripts:

```bash
python examples/quickstart/bandit_ope.py
python examples/quickstart/mdp_ope.py
```

End-to-end pipeline:

```python
from crl.ope import evaluate

report = evaluate(dataset=dataset, policy=benchmark.target_policy)
report.to_dataframe()
report.save_html("report.html")
```

---

## CLI

```bash
crl ope --config configs/ope.yaml --out results/
```

---

## Data contracts

- `BanditDataset`: i.i.d. contexts with one action and reward (+ optional propensities).
- `TrajectoryDataset`: finite-horizon episodes with masks and propensities.
- `TransitionDataset`: (s, a, r, s', done) tuples with optional episode id/timestep.

---

## Estimator selection guide

```text
Do you know behavior propensities?
  yes -> Short horizon: IS or WIS
      -> Long horizon: PDIS, DR, or FQE
  no  -> FQE (model-based), then check diagnostics and sensitivity
Need confidence bounds? -> add high-confidence or bootstrap intervals
Concern about overlap? -> inspect ESS and weight diagnostics first
```

See the estimator docs for method assumptions and failure modes:
https://gsaco.github.io/causalrl/reference/estimators/

---

## Estimator coverage

- Importance sampling: IS, WIS, PDIS, MIS.
- Doubly robust family: DR, WDR, MRDR, MAGIC.
- Model-based: FQE.
- Alternative estimators: DualDICE, DoubleRL.
- High-confidence bounds: HCOPE.

---

## Diagnostics and reporting

- Overlap, effective sample size, and weight tail diagnostics.
- Shift diagnostics when behavior propensities are available.
- HTML reports with embedded figures for sharing and review.

---

## Benchmarks and reproducibility

- Synthetic benchmarks with known ground truth for bandits and MDPs.
- Reproduce runs with `python -m experiments.run_benchmarks --suite all --out results/`.
- Benchmark tutorials live in https://gsaco.github.io/causalrl/tutorials/.

---

## Notebooks and examples

- Walkthrough notebooks in `notebooks/` and `docs/notebooks/`.
- Runnable scripts in `examples/` (see `examples/README.md`).

---

## Documentation

- Live docs: https://gsaco.github.io/causalrl/
- Build locally: `mkdocs serve`

---

## Project status

CausalRL is a research preview (alpha). Confounding and transport settings are
exposed as experimental interfaces and may change.

---

## Citing

If you use CausalRL in your research, please cite it using
[`CITATION.cff`](CITATION.cff) or [`CITATION.bib`](CITATION.bib).

```bibtex
@software{causalrl,
  title = {causalrl: Causal Reinforcement Learning (CRL) toolbox},
  author = {Saco, Gabriel},
  year = {2026},
  version = {0.1.0},
  url = {https://github.com/gsaco/causalrl},
  note = {Research preview},
}
```

---

## Roadmap

- Confounding-robust OPE and sensitivity analysis for partial identification.
- Transport and transfer across environments and behavior policies.
