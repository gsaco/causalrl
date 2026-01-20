# Weighted Doubly Robust (WDR)

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov + value model

## Requires

- TrajectoryDataset
- `behavior_action_probs` for logged actions
- Q-model fit (linear by default in CRL)

## Diagnostics to check

- `overlap.support_violations`
- `ess.ess_ratio`
- `model.q_model_mse`

## Formula

WDR replaces trajectory weights with normalized weights per time step:

$\hat V = \sum_i \bar w_{i0} \hat V(s_{i0}) + \sum_t \sum_i \bar w_{it} \left(r_{it} + \gamma \hat V(s_{i,t+1}) - \hat Q(s_{it}, a_{it})\right)$.

## Fails when

- Still sensitive to model misspecification.
- Normalization can introduce bias in small samples.

## Minimal example

```python
from crl.estimators.wdr import WeightedDoublyRobustEstimator

report = WeightedDoublyRobustEstimator(estimand).estimate(dataset)
```

## References

- Thomas & Brunskill (2016)

## Notebook

- [03_mdp_ope_walkthrough.ipynb](https://github.com/gsaco/causalrl/blob/v4/notebooks/03_mdp_ope_walkthrough.ipynb)
