# CausalRL

CausalRL is an estimand-first toolkit for causal reinforcement learning and
off-policy evaluation (OPE). It makes assumptions explicit, surfaces overlap
diagnostics, and provides reproducible benchmarks for comparing estimators.

**Package**: `causalrl` · **Import**: `crl` · **Version**: 0.1.0 ·
[GitHub](https://github.com/gsaco/causalrl)

## Who it is for

- Researchers comparing OPE estimators with clear assumptions.
- Practitioners who need diagnostics before trusting policy value estimates.
- Anyone building reproducible baselines for offline evaluation.

## 60-second quickstart

```bash
python -m pip install causalrl
```

```python
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.ope import evaluate

benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = benchmark.sample(num_samples=1000, seed=1)
report = evaluate(dataset=dataset, policy=benchmark.target_policy)

print(report.summary_table())
```

This prints a table with estimates and diagnostics for the default estimators.
Use the diagnostics to check overlap and effective sample size before trusting
any estimate.

!!! note "Scope"
    The current `evaluate` pipeline assumes discrete action spaces. See
    [Limitations](concepts/limitations.md) for details.

## Core workflow (mental model)

1. **Define the estimand** (policy, horizon, discount).
2. **Declare assumptions** (sequential ignorability, overlap, Markov for MDPs).
3. **Run estimators** (IS/WIS/PDIS/DR/FQE, depending on data).
4. **Interpret diagnostics** (overlap, ESS, weight tails).

## Data contracts (read first)

Use the dataset contracts in `crl.data` and follow the shape rules exactly:

- Bandits: `LoggedBanditDataset`
- Trajectories: `TrajectoryDataset`
- Transitions: `TransitionDataset`

Start here: [Dataset Format and Validation](concepts/dataset_format.md)

## Learn by example

- [Installation](getting-started/installation.md)
- [Examples](tutorials/examples.md)
- [Estimator Reference](reference/estimators/index.md)
- [Public API](reference/api/public_api.md)
- [Sample HTML report](assets/reports/intro_bandit_report.html)

## Estimator selection

See the [Estimator Selection Guide](explanation/estimator_selection.md) for a
practical decision tree and recommended defaults.
