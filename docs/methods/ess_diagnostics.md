# Effective Sample Size (ESS)

**Estimand**
- Not applicable.

**Assumptions**
- Non-negative importance weights.

**Method**
- ESS = (sum w)^2 / sum w^2.

**Diagnostics**
- ESS and ESS ratio (ESS / n).

**Failure modes**
- ESS can be misleading under extreme weight distributions.

**API**
- `crl.diagnostics.effective_sample_size`

**References**
- Uehara, Shi, Kallus (2022).
