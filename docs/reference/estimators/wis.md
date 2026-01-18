# Weighted Importance Sampling (WIS)

## Assumptions

- Sequential ignorability
- Overlap/positivity

## Formula

With weights $w_i$ and returns $G_i$,

$\hat V = \frac{\sum_i w_i G_i}{\sum_i w_i}$.

## Failure modes

- Bias in small samples due to normalization.
- Still sensitive to extreme weights.

## References

- Precup, Sutton, Dasgupta (2000)

## Notebook

- [02_bandit_ope_walkthrough.ipynb](../../notebooks/02_bandit_ope_walkthrough.ipynb)
- [03_mdp_ope_walkthrough.ipynb](../../notebooks/03_mdp_ope_walkthrough.ipynb)
