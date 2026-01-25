# Double Reinforcement Learning (Bandit)

Implementation: `crl.estimators.double_rl.DoubleRLEstimator`

## Assumptions

- Sequential ignorability
- Overlap/positivity

## Requires

- `LoggedBanditDataset`
- `behavior_action_probs` optional (estimated if contexts are discrete)

## Diagnostics to check

- `overlap.support_violations`
- `ess.ess_ratio`

## Formula

Orthogonalized score using outcome and propensity models in a bandit setting.

## Uncertainty

- Normal-approximation CI by default.
- Bootstrap CI available via `bootstrap=True`.

## Failure modes

- Misspecified nuisance models can increase variance.

## Minimal example

```python
from crl.estimators.double_rl import DoubleRLEstimator

report = DoubleRLEstimator(estimand).estimate(dataset)
```

## References

- Kallus & Uehara (2020)
