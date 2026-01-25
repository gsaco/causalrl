# Proximal OPE

Proximal methods use proxy variables to identify causal effects when standard
ignorability fails. They require strong assumptions about proxy quality and
bridge functions.

## Assumptions

- Proxy variables are correlated with the unobserved confounder.
- Bridge functions are identifiable and well-posed.
- Sufficient variability to solve the moment conditions.

## Practical guidance

- Use `ProximalPolicyValueEstimand` to make proximal assumptions explicit.
- Inspect bridge fit diagnostics (MSE, conditioning) before trusting estimates.

## Limitations

- Sensitive to proxy misspecification.
- Often requires careful feature engineering.
