# Double Reinforcement Learning

## Assumptions

- Sequential ignorability
- Overlap/positivity

## Requires

- LoggedBanditDataset
- `behavior_action_probs` optional (can be estimated for discrete contexts)

## Diagnostics to check

- `overlap.support_violations`
- `ess.ess_ratio`
- `model` diagnostics from reward/behavior models

## Formula

For bandits, the orthogonalized estimator is

$\hat V = \frac{1}{n} \sum_i \left[ \hat m(x_i) + \frac{\pi(a_i|x_i)}{\hat\mu(a_i|x_i)} (r_i - \hat q(x_i, a_i)) \right]$,

with cross-fitting for nuisance models.

## Fails when

- Bias if nuisance models are poor.
- Requires sufficient overlap for stable reweighting.

## Minimal example

```python
from crl.estimators.double_rl import DoubleRLEstimator

report = DoubleRLEstimator(estimand).estimate(dataset)
```

## References

- Kallus & Uehara (2020)

## Notebook

- [02_bandit_ope_walkthrough.ipynb](../../notebooks/02_bandit_ope_walkthrough.ipynb)
