# Quickstart: Bandit OPE

This is the smallest end-to-end loop: synthesize data, define an estimand,
run estimators, and read diagnostics.

## 1) Generate data

```python
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig

benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = benchmark.sample(num_samples=1000, seed=1)
```

!!! note "Behavior propensities"
    Synthetic benchmarks include `behavior_action_probs`. If your real data
    does not, see [Behavior Propensities Missing](../how-to/behavior_propensities_missing.md).

## 2) Define the estimand

```python
from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import BEHAVIOR_POLICY_KNOWN, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.estimands.policy_value import PolicyValueEstimand

estimand = PolicyValueEstimand(
    policy=benchmark.target_policy,
    discount=1.0,
    horizon=1,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN]),
)
```

## 3) Run estimators and read diagnostics

```python
from crl.estimators.importance_sampling import ISEstimator, WISEstimator

for estimator in [ISEstimator(estimand), WISEstimator(estimand)]:
    report = estimator.estimate(dataset)
    print(report.value, report.diagnostics)
```

Key questions to ask:

- Do overlap diagnostics show support violations?
- Is ESS low or weight tails heavy?
- Do estimates move drastically across IS and WIS?

## 4) Compare to ground truth (for synthetic data)

```python
true_value = benchmark.true_policy_value(benchmark.target_policy)
print("true", true_value)
```

## Next steps

- Read the dataset contract: [Dataset Format and Validation](../concepts/dataset_format.md)
- Run the full quickstart script: `python examples/quickstart/bandit_ope.py`
- Learn how to export results: [Export and Reporting](../how-to/export_and_reporting.md)
