# MAGIC

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

MAGIC mixes truncated DR estimators with data-driven weights that target low
MSE. Conceptually,

$\hat V_{\text{MAGIC}} = \sum_j \alpha_j \hat V_j$ with $\sum_j \alpha_j = 1$.

## Fails when

- Sensitive to poor value models.
- Weight selection can be noisy in small samples.

## Minimal example

```python
from crl.estimators.magic import MAGICEstimator

report = MAGICEstimator(estimand).estimate(dataset)
```

## References

- Thomas & Brunskill (2016)

## Notebook

- [03_mdp_ope_walkthrough.ipynb](https://github.com/gsaco/causalrl/blob/v4/notebooks/03_mdp_ope_walkthrough.ipynb)
