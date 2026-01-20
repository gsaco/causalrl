# Weighted Importance Sampling (WIS)

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

With weights $w_i$ and returns $G_i$,

$\hat V = \frac{\sum_i w_i G_i}{\sum_i w_i}$.

## Fails when

- Bias in small samples due to normalization.
- Still sensitive to extreme weights.

## Minimal example

```python
from crl.estimators.importance_sampling import WISEstimator

report = WISEstimator(estimand).estimate(dataset)
```

## References

- Precup, Sutton, Dasgupta (2000)

## Notebook

- [02_bandit_ope_walkthrough.ipynb](https://github.com/gsaco/causalrl/blob/v4/notebooks/02_bandit_ope_walkthrough.ipynb)
- [03_mdp_ope_walkthrough.ipynb](https://github.com/gsaco/causalrl/blob/v4/notebooks/03_mdp_ope_walkthrough.ipynb)
