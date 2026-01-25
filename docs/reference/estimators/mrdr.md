# MRDR

Implementation: `crl.estimators.mrdr.MRDREstimator`

## Estimand

$V^\pi = \mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t r_t\right]$.

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov property

## Inputs required

- `TrajectoryDataset`
- `behavior_action_probs` for logged actions
- Q-model fit (weighted linear model in CRL)

## Algorithm

MRDR fits Q-models with weights chosen to reduce DR variance, then applies the DR correction.

## Formula

$\hat V = \frac{1}{n} \sum_i \left[ \hat V(s_{i0}) + \sum_t \gamma^t \rho_{i,t} \left(r_{it} + \gamma \hat V(s_{i,t+1}) - \hat Q(s_{it}, a_{it})\right) \right]$.

## Diagnostics

- `overlap.support_violations`
- `ess.ess_ratio`
- `model.q_model_mse`

## Uncertainty

- Normal-approximation CI by default.
- Bootstrap CI available via `bootstrap=True`.

## Failure modes

- Sensitive to model misspecification if overlap is weak.
- Weighted regression can be unstable with extreme ratios.

## Minimal example

```python
from crl.estimators.mrdr import MRDREstimator

report = MRDREstimator(estimand).estimate(dataset)
```

## References

- Farajtabar et al. (2018)
