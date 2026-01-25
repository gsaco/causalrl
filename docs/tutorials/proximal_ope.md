# Proximal OPE (Advanced)

This tutorial demonstrates a confounded bandit where standard OPE is biased and
proximal estimation uses proxy variables to improve robustness.

Use `ProximalPolicyValueEstimand` and `ProximalOPEEstimator` to make proximal
assumptions explicit and inspect bridge diagnostics.

!!! warning "Experimental"
    The current proximal implementation is a simplified linear bridge without
    cross-fitting or instrument-strength diagnostics. Treat results as
    exploratory and validate assumptions carefully.

## Notebook

- [07_proximal_ope_confounded_pomdp.ipynb](https://github.com/gsaco/causalrl/blob/main/notebooks/07_proximal_ope_confounded_pomdp.ipynb)
