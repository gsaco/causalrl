# Diagnostics

Overlap and effective sample size diagnostics help you detect instability before
trusting an off-policy estimate.

## Checklist

- Are target actions supported by the behavior policy?
- Is the ESS ratio large enough to trust IS-based estimators?
- Do weight tails indicate extreme variance?

## Notebook

- [08_estimator_selection_and_debugging.ipynb](../notebooks/08_estimator_selection_and_debugging.ipynb)
