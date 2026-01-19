# Behavior Propensities Missing

Many real datasets do not include logged behavior propensities. You have two
options:

1. Estimate propensities from data.
2. Use estimators that do not require propensities (when applicable).

## Option 1: estimate propensities

CausalRL includes behavior policy estimation utilities (requires scikit-learn).
Install with `causalrl[behavior]`.

```python
from crl.behavior import fit_behavior_policy

fit = fit_behavior_policy(dataset, method="logit", seed=0)
with_props = fit.apply(dataset)
```

This attaches estimated propensities and records metadata so reports can warn
that propensities were modeled.

## Option 2: choose prop-free estimators

- For MDPs, model-based estimators like FQE can run without propensities.
- For bandits, you typically need propensities to use IS/WIS.
  `evaluate` will skip propensity-based estimators when propensities are missing.

## Best practices

- Always log true behavior propensities when possible.
- If you estimate them, run diagnostics and be explicit in reporting.
- Sensitivity analysis can help quantify robustness under misspecification.
