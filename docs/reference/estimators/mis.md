# Marginalized Importance Sampling (MIS)

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov

## Formula

MIS replaces cumulative ratios with marginal action ratios at each time step:

$\hat V = \frac{1}{n} \sum_i \sum_t \gamma^t \frac{\pi(a_{it}|s_{it})}{\hat\mu_t(a_{it}|s_{it})} r_{it}$.

## Failure modes

- Requires discrete state representation or density estimation.
- Sensitive to small counts per state/time.

## References

- Xie et al. (2019)

## Notebook

- [06_long_horizon_mis_vs_is.ipynb](../../notebooks/06_long_horizon_mis_vs_is.ipynb)
