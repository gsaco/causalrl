# CausalRL

[![CI](https://github.com/gsaco/causalrl/actions/workflows/ci.yml/badge.svg)](https://github.com/gsaco/causalrl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://gsaco.github.io/causalrl/)
[![Coverage](https://codecov.io/gh/gsaco/causalrl/branch/main/graph/badge.svg)](https://codecov.io/gh/gsaco/causalrl)
[![PyPI](https://img.shields.io/badge/PyPI-coming%20soon-lightgrey)](https://pypi.org/project/causalrl/)
[![License](https://img.shields.io/github/license/gsaco/causalrl)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

CausalRL is an estimand-first toolkit for causal reinforcement learning and
off-policy evaluation that pairs identification assumptions with diagnostics so
you can trust or debug a policy value estimate.

## Why CausalRL

- Explicit estimands and assumptions make causal guarantees clear and auditable.
- Diagnostics-first reports surface overlap, ESS, and weight pathologies early.
- Synthetic benchmarks with ground truth support method selection and regression
  tests.

## When to use this library

- Offline evaluation when online deployment is risky, expensive, or impossible.
- Diagnostics-first iteration when you need to validate overlap and stability
  before trusting an estimate.
- Synthetic ground-truth settings to compare estimators and tune workflows.

## Installation

```bash
python -m pip install causalrl
```

Optional extras:

```bash
python -m pip install "causalrl[docs]"
python -m pip install "causalrl[benchmarks]"
python -m pip install "causalrl[notebooks]"
```

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
https://gsaco.github.io/causalrl/api/estimators/

## Benchmarks and reproducibility

- Synthetic benchmarks with known ground truth are built in for bandits and MDPs.
- Reproduce runs with `python -m experiments.run_benchmarks --suite all --out results/`.
- Benchmark tutorials live in https://gsaco.github.io/causalrl/tutorials/.

## Documentation

- Live docs: https://gsaco.github.io/causalrl/
- Build locally: `mkdocs serve`

## Citing

If you use CausalRL in your research, please cite it using
[`CITATION.cff`](CITATION.cff).

## Roadmap

- Confounding-robust OPE and sensitivity analysis for partial identification.
- Transport and transfer across environments and behavior policies.
