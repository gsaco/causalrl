# Doubly Robust (DR)

Implementation: `crl.estimators.dr.DoublyRobustEstimator`

## Estimand

$V^\pi = \mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t r_t\right]$.

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov property

## Inputs required

- `TrajectoryDataset`
- `behavior_action_probs` for logged actions
- Q-model fit (linear by default in CRL)

## Algorithm

1. Fit a Q-model on logged data (optionally cross-fitted).
2. Compute cumulative importance ratios along each trajectory.
3. Combine model-based value with an importance-weighted TD correction.

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

- Bias if both propensities and Q-model are misspecified.
- High variance under weak overlap.

## Minimal example

```python
from crl.estimators.dr import DoublyRobustEstimator

report = DoublyRobustEstimator(estimand).estimate(dataset)
```

## References

- Jiang & Li (2016)

## Notebook

- [03_mdp_ope_walkthrough.ipynb](https://github.com/gsaco/causalrl/blob/v4/notebooks/03_mdp_ope_walkthrough.ipynb)
