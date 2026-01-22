# Estimands

An estimand is the causal quantity you want to identify and estimate. CausalRL
makes the estimand explicit and pairs it with assumptions.

## Policy value

`PolicyValueEstimand` represents the expected return of a target policy.

```python
from crl.estimands.policy_value import PolicyValueEstimand
from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import OVERLAP, SEQUENTIAL_IGNORABILITY

estimand = PolicyValueEstimand(
    policy=target_policy,
    discount=0.99,
    horizon=10,
    assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
)
```

## Policy contrast

`PolicyContrastEstimand` captures differences between two policies.

```python
from crl.estimands.policy_value import PolicyContrastEstimand

contrast = PolicyContrastEstimand(
    treatment=policy_a,
    control=policy_b,
    discount=0.99,
    horizon=10,
)
```

## Sensitivity estimand

`SensitivityPolicyValueEstimand` encodes a gamma-bounded confounding model and
produces partial identification bounds.

## Proximal estimand

`ProximalPolicyValueEstimand` encodes proximal assumptions about proxy
variables and bridge functions.

## Why estimands matter

- They define what you can claim.
- They connect assumptions to estimators.
- They force clarity when data are incomplete.
