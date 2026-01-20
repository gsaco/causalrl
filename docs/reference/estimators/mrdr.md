# MRDR

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

MRDR modifies the Q-model loss to minimize DR variance. In linear models, this
corresponds to weighted regression with importance ratios.

## Fails when

- Still biased if both propensities and model are misspecified.
- Instability when overlap is weak.

## Minimal example

```python
from crl.estimators.mrdr import MRDREstimator

report = MRDREstimator(estimand).estimate(dataset)
```

## References

- Farajtabar et al. (2018)

## Notebook

- [03_mdp_ope_walkthrough.ipynb](https://github.com/gsaco/causalrl/blob/v4/notebooks/03_mdp_ope_walkthrough.ipynb)
