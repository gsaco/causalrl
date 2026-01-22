# High-Confidence Off-Policy Evaluation (HCOPE)

Implementation: `crl.estimators.high_confidence.HighConfidenceISEstimator`

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Bounded rewards

## Requires

- `behavior_action_probs` for logged actions

## Diagnostics to check

- `overlap.support_violations`
- `ess.ess_ratio`
- `weights.tail_fraction`

## Formula (sketch)

Compute a clipped IS estimate and apply a concentration inequality
(empirical Bernstein or Hoeffding) with bias correction, selecting the
clipping parameter that maximizes the lower bound.

## Uncertainty

- Returns a lower bound (not a symmetric CI).
- Confidence level set by `delta` in `HighConfidenceISConfig`.

## Failure modes

- Requires a valid reward bound; inferred bounds are heuristic.
- Bounds can be vacuous with weak overlap.

## Minimal example

```python
from crl.estimators.high_confidence import HighConfidenceISEstimator, HighConfidenceISConfig

config = HighConfidenceISConfig(delta=0.05, bound="empirical_bernstein")
report = HighConfidenceISEstimator(estimand, config=config).estimate(dataset)
```

## References

- Thomas et al. (2015)
