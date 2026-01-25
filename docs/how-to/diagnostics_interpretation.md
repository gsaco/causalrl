# Diagnostics Interpretation

Diagnostics are how you decide whether to trust an estimate.

## Core diagnostics

- **Overlap**: indicates whether the target policy is supported by logged data.
- **ESS (effective sample size)**: a proxy for variance inflation.
- **Weight tails**: flags heavy-tailed importance weights.

## Quick reading guide

1. If overlap shows support violations, treat estimates as unreliable.
2. If ESS is very low, expect high variance and unstable estimates.
3. If weight tails are heavy, IS/PDIS may be dominated by few trajectories.

## Where to find diagnostics

Every estimator returns an `EstimatorReport` with `report.diagnostics` and
`report.warnings`. For end-to-end evaluations, use `OpeReport` and inspect each
estimator report.

## What to do when diagnostics are bad

- Try a lower-variance estimator (WIS, DR, FQE).
- Improve behavior policy logging or estimation.
- Use sensitivity analysis to bound possible errors.
