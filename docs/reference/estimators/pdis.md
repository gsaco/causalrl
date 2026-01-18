# Per-Decision Importance Sampling (PDIS)

## Assumptions

- Sequential ignorability
- Overlap/positivity

## Formula

$\hat V = \frac{1}{n} \sum_i \sum_t \left( \prod_{k \le t} \frac{\pi(a_{ik}|s_{ik})}{\mu(a_{ik}|s_{ik})} \right) \gamma^t r_{it}$.

## Failure modes

- Variance grows with horizon if overlap is weak.

## References

- Precup, Sutton, Dasgupta (2000)

## Notebook

- [03_mdp_ope_walkthrough.ipynb](../../notebooks/03_mdp_ope_walkthrough.ipynb)
