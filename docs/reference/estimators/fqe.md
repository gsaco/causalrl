# Fitted Q Evaluation (FQE)

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov + function approximation

## Requires

- TrajectoryDataset
- `behavior_action_probs` optional (used only for diagnostics)

## Diagnostics to check

- `model.q_model_mse`
- `overlap.support_violations` (if propensities provided)

## Formula

FQE fits a Q-function by iterative Bellman regression on logged data, then
estimates $V^\pi$ by averaging $\hat V(s_0)$ under the target policy.

## Fails when

- Extrapolation error for out-of-distribution state-action pairs.
- Sensitive to model capacity and optimization.

## Minimal example

```python
from crl.estimators.fqe import FQEEstimator

report = FQEEstimator(estimand).estimate(dataset)
```

## Bootstrap notes

- IID bootstrap ignores temporal dependence and can be optimistic.
- Trajectory or block bootstraps are preferable for sequential data.

## References

- Le et al. (2019)
- Hao et al. (2021) for bootstrap inference

## Notebook

- [03_mdp_ope_walkthrough.ipynb](https://github.com/gsaco/causalrl/blob/v4/notebooks/03_mdp_ope_walkthrough.ipynb)
- [04_confidence_intervals_and_hcope.ipynb](https://github.com/gsaco/causalrl/blob/v4/notebooks/04_confidence_intervals_and_hcope.ipynb)
