# MAGIC

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov + value model

## Formula

MAGIC mixes truncated DR estimators with data-driven weights that target low
MSE. Conceptually,

$\hat V_{\text{MAGIC}} = \sum_j \alpha_j \hat V_j$ with $\sum_j \alpha_j = 1$.

## Failure modes

- Sensitive to poor value models.
- Weight selection can be noisy in small samples.

## References

- Thomas & Brunskill (2016)

## Notebook

- [03_mdp_ope_walkthrough.ipynb](../../notebooks/03_mdp_ope_walkthrough.ipynb)
