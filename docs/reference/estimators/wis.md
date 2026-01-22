# Weighted Importance Sampling (WIS)

Implementation: `crl.estimators.importance_sampling.WISEstimator`

## Assumptions

- Sequential ignorability
- Overlap/positivity

## Requires

- `LoggedBanditDataset` or `TrajectoryDataset`
- `behavior_action_probs` for logged actions

## Diagnostics to check

- `overlap.support_violations`
- `ess.ess_ratio`
- `weights.tail_fraction`

## Formula

Self-normalized IS:

$\hat V = \frac{\sum_i w_i G_i}{\sum_i w_i}$.

## Uncertainty

- Normal-approximation CI by default.
- Bootstrap CI available via `bootstrap=True`.

## Failure modes

- Bias from normalization in small samples.
- Heavy-tailed ratios can still dominate variance.

## Minimal example

```python
from crl.estimators.importance_sampling import WISEstimator

report = WISEstimator(estimand, clip_rho=10.0).estimate(dataset)
```

## References

- Swaminathan & Joachims (2015)
