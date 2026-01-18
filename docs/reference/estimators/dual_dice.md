# DualDICE

## Assumptions

- Sequential ignorability
- Markov

## Formula

DualDICE solves a density ratio estimation problem for the discounted
stationary distribution ratio $w = d_\pi / d_\mu$ and estimates

$V^\pi \approx \frac{1}{1-\gamma} \mathbb{E}_{d_\mu}[w(s,a) r(s,a)]$.

## Failure modes

- Requires stable density ratio estimation.
- Sensitive to feature misspecification for function approximation.

## References

- Nachum et al. (2019)

## Notebook

- [03_mdp_ope_walkthrough.ipynb](../../notebooks/03_mdp_ope_walkthrough.ipynb)
