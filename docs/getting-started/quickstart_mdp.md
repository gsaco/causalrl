# Quickstart: MDP OPE

This walkthrough evaluates a target policy on trajectory data.

## 1) Generate data

```python
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig

benchmark = SyntheticMDP(SyntheticMDPConfig(seed=0))
dataset = benchmark.sample(num_trajectories=200, seed=1)
```

## 2) Define the estimand

```python
from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.estimands.policy_value import PolicyValueEstimand

estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=dataset.discount,
    horizon=dataset.horizon,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV]),
)
```

## 3) Run estimators

```python
from crl.estimators.dr import DoublyRobustEstimator
from crl.estimators.fqe import FQEEstimator
from crl.estimators.importance_sampling import ISEstimator, PDISEstimator, WISEstimator

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

## 4) Compare to ground truth

```python
true_value = benchmark.true_policy_value(benchmark.target_policy)
print("true", true_value)
```

## Interpret the results

- If overlap is weak, IS and PDIS will be unstable.
- DR and FQE can be more stable but depend on model fit.
- Always read diagnostics before trusting an estimate.

## Next steps

- Run the full quickstart script: `python examples/quickstart/mdp_ope.py`
- See the decision tree: [Estimator Selection Guide](../explanation/estimator_selection.md)
- Learn how to read diagnostics: [Diagnostics Interpretation](../how-to/diagnostics_interpretation.md)
