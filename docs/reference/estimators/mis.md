# Marginalized Importance Sampling (MIS)

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov

## Requires

- TrajectoryDataset
- Discrete `state_space_n` (for tabular marginals)

## Diagnostics to check

- `overlap.support_violations`
- `ess.ess_ratio`
- `weights.tail_fraction`

## Formula

MIS replaces cumulative ratios with marginal action ratios at each time step:

$\hat V = \frac{1}{n} \sum_i \sum_t \gamma^t \frac{\pi(a_{it}|s_{it})}{\hat\mu_t(a_{it}|s_{it})} r_{it}$.

## Fails when

- Requires discrete state representation or density estimation.
- Sensitive to small counts per state/time.

## Minimal example

```python
from crl.estimators.mis import MarginalizedImportanceSamplingEstimator

report = MarginalizedImportanceSamplingEstimator(estimand).estimate(dataset)
```

## References

- Xie et al. (2019)

## Notebook

- [06_long_horizon_mis_vs_is.ipynb](../../notebooks/06_long_horizon_mis_vs_is.ipynb)
