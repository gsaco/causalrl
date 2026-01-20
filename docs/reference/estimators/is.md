# Importance Sampling (IS)

## Assumptions

- Sequential ignorability
- Overlap/positivity

## Requires

- LoggedBanditDataset or TrajectoryDataset
- `behavior_action_probs` for logged actions

## Diagnostics to check

- `overlap.support_violations`
- `ess.ess_ratio`
- `weights.tail_fraction`

## Formula

For trajectory return $G_i$ and importance weight $w_i = \prod_t \pi(a_t|s_t) / \mu(a_t|s_t)$,

$\hat V = \frac{1}{n} \sum_{i=1}^n w_i G_i$.

## Fails when

- High variance with weak overlap or long horizons.
- Heavy-tailed weights can dominate the estimate.

## Minimal example

```python
from crl.estimators.importance_sampling import ISEstimator

report = ISEstimator(estimand).estimate(dataset)
```

## References

- Robins, Hernan, Brumback (2000)

## Notebook

- [02_bandit_ope_walkthrough.ipynb](https://github.com/gsaco/causalrl/blob/v4/notebooks/02_bandit_ope_walkthrough.ipynb)
- [03_mdp_ope_walkthrough.ipynb](https://github.com/gsaco/causalrl/blob/v4/notebooks/03_mdp_ope_walkthrough.ipynb)
