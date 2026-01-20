# High-Confidence OPE Lower Bounds

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Bounded rewards

## Requires

- LoggedBanditDataset or TrajectoryDataset
- `behavior_action_probs` for logged actions
- Reward bound (or inferred bound with warning)

## Diagnostics to check

- `overlap.support_violations`
- `ess.ess_ratio`

## Formula

Using an empirical Bernstein bound, compute a lower confidence bound for
importance-sampled returns:

$\text{LCB} = \bar X - \sqrt{\frac{2 \hat\sigma^2 \log(2/\delta)}{n}} - \frac{7 R_{\max} \log(2/\delta)}{3(n-1)}$.

## Fails when

- Conservative when variance is large.
- Requires a valid reward bound.

## Minimal example

```python
from crl.estimators.high_confidence import HighConfidenceISEstimator

report = HighConfidenceISEstimator(estimand).estimate(dataset)
```

## References

- Thomas et al. (2015)

## Notebook

- [04_confidence_intervals_and_hcope.ipynb](https://github.com/gsaco/causalrl/blob/v4/notebooks/04_confidence_intervals_and_hcope.ipynb)
