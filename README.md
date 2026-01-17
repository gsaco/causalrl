# causalrl

Causal Reinforcement Learning (CRL) is the intersection of sequential decision-making and causal inference.
This package provides estimand-first, diagnostics-first tools for offline policy evaluation and synthetic
benchmarks with ground truth.

## What CRL is (taxonomy)

- Causal bandits: single-step decisions with explicit interventions
- Offline RL and off-policy evaluation in MDPs under sequential ignorability
- Confounded and partially identified settings (interfaces are explicit, advanced methods marked experimental)
- Transport and mechanism shift (planned)

## Quickstart

```bash
python -m pip install -e ".[dev]"
```

## Minimal examples

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

## Scope and non-goals

- This release targets unconfounded OPE with explicit estimands and diagnostics.
- Includes a bandit propensity sensitivity analysis for bounded confounding.
- Confounded and transport settings are documented as experimental interfaces only.
- Representation learning claims are treated as heuristics unless identified.

## Citations (verified sources)

- Deng, Jiang, Long, Zhang (2023). Causal Reinforcement Learning: A Survey. https://arxiv.org/abs/2307.01452
- da Costa Cunha, Liu, French, Mian (2025). Unifying Causal Reinforcement Learning. https://arxiv.org/abs/2512.18135
- Jiang and Li (2016). Doubly Robust Off-policy Value Evaluation. https://proceedings.mlr.press/v48/jiang16.html
- Kallus and Uehara (2020). Double Reinforcement Learning. https://jmlr.org/papers/v21/19-827.html
- Uehara, Shi, Kallus (2022). Review of OPE in RL. https://arxiv.org/abs/2212.06355
- Hernan and Robins. Causal Inference: What If. https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
- Robins, Hernan, Brumback (2000). Marginal Structural Models and Causal Inference in Epidemiology.

## Benchmarks

Run the synthetic benchmarks and write a table:

```bash
python experiments/run_benchmarks.py --output-dir results
```

## Documentation

Build the docs locally:

```bash
mkdocs serve
```
