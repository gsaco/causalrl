# Doubly Robust (DR)

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov + correct value model specification

## Formula

$\hat V = \frac{1}{n} \sum_i \left[ \hat V(s_{i0}) + \sum_t w_{it} \left(r_{it} + \gamma \hat V(s_{i,t+1}) - \hat Q(s_{it}, a_{it})\right) \right]$.

## Failure modes

- Bias if both the reward/Q model and propensities are misspecified.

## References

- Jiang & Li (2016)

## Notebook

- [03_mdp_ope_walkthrough.ipynb](../../notebooks/03_mdp_ope_walkthrough.ipynb)
