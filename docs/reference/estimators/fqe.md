# Fitted Q Evaluation (FQE)

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov + function approximation

## Formula

FQE fits a Q-function by iterative Bellman regression on logged data, then
estimates $V^\pi$ by averaging $\hat V(s_0)$ under the target policy.

## Failure modes

- Extrapolation error for out-of-distribution state-action pairs.
- Sensitive to model capacity and optimization.

## Bootstrap notes

- IID bootstrap ignores temporal dependence and can be optimistic.
- Trajectory or block bootstraps are preferable for sequential data.

## References

- Le et al. (2019)
- Hao et al. (2021) for bootstrap inference

## Notebook

- [02_quickstart_mdp_ope.ipynb](../../notebooks/02_quickstart_mdp_ope.ipynb)
- [04_confidence_intervals_safe_selection.ipynb](../../notebooks/04_confidence_intervals_safe_selection.ipynb)
