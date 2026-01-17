# Overlap Diagnostics

**Estimand**
- Not applicable.

**Assumptions**
- Requires logged propensities and target action probabilities.

**Method**
- Summarize behavior propensities and target/behavior ratios, highlight support violations.

**Diagnostics**
- Minimum propensity, fraction below threshold, ratio quantiles.

**Failure modes**
- Diagnostics are descriptive and do not prove ignorability.

**API**
- `crl.diagnostics.compute_overlap_metrics`

**References**
- Robins, Hernan, Brumback (2000).
