# Bandit OPE Tutorial (Script)

This tutorial mirrors the `examples/quickstart/bandit_ope.py` script. It is
designed for researchers who want a minimal, reproducible bandit OPE run that
still exposes diagnostics and uncertainty.

## Run the script

```bash
python examples/quickstart/bandit_ope.py
```

## What it does

1. Generates a synthetic contextual bandit dataset (ground truth known).
2. Defines a `PolicyValueEstimand` with overlap and ignorability assumptions.
3. Runs IS and WIS.
4. Prints a report table with estimates and diagnostics.

## What to look for

- **Overlap diagnostics**: are importance ratios heavy-tailed?
- **ESS**: small values suggest unstable estimates.
- **IS vs WIS gap**: large gaps hint at variance issues.

## Next steps

- See the full notebook workflow: `notebooks/02_bandit_ope_walkthrough.ipynb`
- Use `evaluate(...)` for multi-estimator reports.
