# Weighted Doubly Robust (WDR)

Implementation: `crl.estimators.wdr.WeightedDoublyRobustEstimator`

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

WDR replaces trajectory weights with normalized weights per time step to reduce variance.

## Formula

$\hat V = \sum_i \bar w_{i0} \hat V(s_{i0}) + \sum_t \sum_i \gamma^t \bar w_{it} \left(r_{it} + \gamma \hat V(s_{i,t+1}) - \hat Q(s_{it}, a_{it})\right)$.

## Diagnostics

- `overlap.support_violations`
- `ess.ess_ratio`
- `model.q_model_mse`

## Uncertainty

- Normal-approximation CI by default.
- Bootstrap CI available via `bootstrap=True`.

## Failure modes

- Normalization can introduce bias in small samples.
- Sensitive to model misspecification when overlap is weak.

## Minimal example

```python
from crl.estimators.wdr import WeightedDoublyRobustEstimator

report = WeightedDoublyRobustEstimator(estimand).estimate(dataset)
```

## References

- Thomas & Brunskill (2016)

## Notebook

- [03_mdp_ope_walkthrough.ipynb](https://github.com/gsaco/causalrl/blob/main/notebooks/03_mdp_ope_walkthrough.ipynb)
