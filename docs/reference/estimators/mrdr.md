# MRDR

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov + value model

## Formula

MRDR modifies the Q-model loss to minimize DR variance. In linear models, this
corresponds to weighted regression with importance ratios.

## Failure modes

- Still biased if both propensities and model are misspecified.
- Instability when overlap is weak.

## References

- Farajtabar et al. (2018)

## Notebook

- [02_quickstart_mdp_ope.ipynb](../../notebooks/02_quickstart_mdp_ope.ipynb)
