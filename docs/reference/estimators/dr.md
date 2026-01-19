# Doubly Robust (DR)

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov + correct value model specification

## Requires

- TrajectoryDataset
- `behavior_action_probs` for logged actions
- Q-model fit (linear by default in CRL)

## Diagnostics to check

- `overlap.support_violations`
- `ess.ess_ratio`
- `model.q_model_mse`

## Formula

$\hat V = \frac{1}{n} \sum_i \left[ \hat V(s_{i0}) + \sum_t w_{it} \left(r_{it} + \gamma \hat V(s_{i,t+1}) - \hat Q(s_{it}, a_{it})\right) \right]$.

## Fails when

- Bias if both the reward/Q model and propensities are misspecified.

## Minimal example

```python
from crl.estimators.dr import DoublyRobustEstimator

report = DoublyRobustEstimator(estimand).estimate(dataset)
```

## References

- Jiang & Li (2016)

## Notebook

- [03_mdp_ope_walkthrough.ipynb](../../notebooks/03_mdp_ope_walkthrough.ipynb)
