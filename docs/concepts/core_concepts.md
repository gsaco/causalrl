# Core Concepts

This page summarizes the core objects you will use in CausalRL and how they
connect in the OPE workflow. All names below map to real classes/functions in
the package.

## Datasets

CausalRL expects explicit dataset objects with validated shapes:

- **`LoggedBanditDataset`** (`crl.data`) for contextual bandits.
- **`TrajectoryDataset`** (`crl.data`) for finite-horizon MDP trajectories.
- **`TransitionDataset`** (`crl.data`) for (s, a, r, s', done) logs that can be
  grouped into trajectories.

Each dataset exposes `.discount`, `.horizon`, and optional
`behavior_action_probs` for logged propensities.

## Policies

Policies implement action probabilities for observed actions:

- **`Policy`** protocol (`crl.policies`) defines `action_probs` and
  `action_prob`.
- Use `Policy.from_sklearn` or `Policy.from_torch` to wrap existing models.
- Reference implementations include `TabularPolicy`, `StochasticPolicy`, and
  `CallablePolicy`.

## Estimands and assumptions

The central estimand is the **policy value**:

- **`PolicyValueEstimand`** binds the target policy with horizon, discount, and
  an `AssumptionSet`.
- Assumptions are explicit via **`Assumption`** and **`AssumptionSet`**.

## Estimators

Estimators return an **`EstimatorReport`** with the value estimate, standard
error, confidence interval, diagnostics, and warnings. Examples include:

- Importance sampling: `ISEstimator`, `WISEstimator`, `PDISEstimator`
- Doubly robust: `DoublyRobustEstimator`, `WeightedDoublyRobustEstimator`
- Model-based: `FQEEstimator`
- Specialized: `MAGICEstimator`, `MRDREstimator`, `DualDICEEstimator`

## Diagnostics and reports

Diagnostics quantify overlap, effective sample size, and tail behavior of
importance weights. Reports are aggregated in an **`OpeReport`**, which can
render tables and HTML outputs.

## End-to-end pipeline

For most workflows you can use `evaluate`:

```python
from crl.ope import evaluate

report = evaluate(dataset=dataset, policy=policy)
```

This selects default estimators for the dataset type and returns an `OpeReport`
with diagnostics and metadata.
