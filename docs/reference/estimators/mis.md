# Marginalized Importance Sampling (MIS)

Implementation: `crl.estimators.mis.MarginalizedImportanceSamplingEstimator`

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov property

## Requires

- `TrajectoryDataset` with `state_space_n`

## Diagnostics to check

- `overlap.support_violations`
- `ess.ess_ratio`

## Formula

MIS replaces cumulative importance ratios with marginal state-action ratios computed from counts.

## Uncertainty

- Normal-approximation CI by default.
- Bootstrap CI available via `bootstrap=True`.

## Failure modes

- Requires discrete state space; sparse counts can inflate variance.

## Minimal example

```python
from crl.estimators.mis import MarginalizedImportanceSamplingEstimator

report = MarginalizedImportanceSamplingEstimator(estimand).estimate(dataset)
```

## References

- Xie et al. (2019)
