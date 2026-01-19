# Per-Decision Importance Sampling (PDIS)

## Assumptions

- Sequential ignorability
- Overlap/positivity

## Requires

- TrajectoryDataset (finite horizon)
- `behavior_action_probs` for logged actions

## Diagnostics to check

- `overlap.support_violations`
- `ess.ess_ratio`
- `weights.tail_fraction`

## Formula

$\hat V = \frac{1}{n} \sum_i \sum_t \left( \prod_{k \le t} \frac{\pi(a_{ik}|s_{ik})}{\mu(a_{ik}|s_{ik})} \right) \gamma^t r_{it}$.

## Fails when

- Variance grows with horizon if overlap is weak.

## Minimal example

```python
from crl.estimators.importance_sampling import PDISEstimator

report = PDISEstimator(estimand).estimate(dataset)
```

## References

- Precup, Sutton, Dasgupta (2000)

## Notebook

- [03_mdp_ope_walkthrough.ipynb](../../notebooks/03_mdp_ope_walkthrough.ipynb)
