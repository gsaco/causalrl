# Per-Decision Importance Sampling (PDIS)

Implementation: `crl.estimators.importance_sampling.PDISEstimator`

## Assumptions

- Sequential ignorability
- Overlap/positivity

## Requires

- `TrajectoryDataset`
- `behavior_action_probs` for logged actions

## Diagnostics to check

- `overlap.support_violations`
- `ess.ess_ratio`
- `weights.tail_fraction`

## Formula

$\hat V = \frac{1}{n} \sum_i \sum_t \gamma^t \rho_{i,t} r_{it}$.

## Uncertainty

- Normal-approximation CI by default.
- Bootstrap CI available via `bootstrap=True`.

## Failure modes

- Variance grows with horizon under weak overlap.
- Stepwise ratios can still be heavy-tailed.

## Minimal example

```python
from crl.estimators.importance_sampling import PDISEstimator

report = PDISEstimator(estimand, clip_rho=10.0).estimate(dataset)
```

## References

- Precup, Sutton, Singh (2000)
