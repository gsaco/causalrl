# Weighted Doubly Robust (WDR)

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov + value model

## Formula

WDR replaces trajectory weights with normalized weights per time step:

$\hat V = \sum_i \bar w_{i0} \hat V(s_{i0}) + \sum_t \sum_i \bar w_{it} \left(r_{it} + \gamma \hat V(s_{i,t+1}) - \hat Q(s_{it}, a_{it})\right)$.

## Failure modes

- Still sensitive to model misspecification.
- Normalization can introduce bias in small samples.

## References

- Thomas & Brunskill (2016)

## Notebook

- [03_mdp_ope_walkthrough.ipynb](../../notebooks/03_mdp_ope_walkthrough.ipynb)
