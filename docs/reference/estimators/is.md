# Importance Sampling (IS)

## Assumptions

- Sequential ignorability
- Overlap/positivity

## Formula

For trajectory return $G_i$ and importance weight $w_i = \prod_t \pi(a_t|s_t) / \mu(a_t|s_t)$,

$\hat V = \frac{1}{n} \sum_{i=1}^n w_i G_i$.

## Failure modes

- High variance with weak overlap or long horizons.
- Heavy-tailed weights can dominate the estimate.

## References

- Robins, Hernan, Brumback (2000)

## Notebook

- [01_quickstart_bandit_ope.ipynb](../../notebooks/01_quickstart_bandit_ope.ipynb)
- [02_quickstart_mdp_ope.ipynb](../../notebooks/02_quickstart_mdp_ope.ipynb)
