# Sensitivity Analysis

Sensitivity analysis quantifies how conclusions change when key assumptions are
relaxed, especially unobserved confounding.

## What it gives you

- A range of plausible policy values instead of a single point.
- A way to report robustness to hidden bias.

## When to use it

- You cannot guarantee sequential ignorability.
- Overlap is weak or behavior logging is noisy.

## Practical guidance

- Report both the point estimate and the sensitivity bounds.
- Be explicit about the sensitivity parameter and its meaning.
- Use the `SensitivityPolicyValueEstimand` to make the model explicit.
- Use synthetic benchmarks to build intuition before real data.
