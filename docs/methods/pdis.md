# Per-Decision Importance Sampling (PDIS)

**Estimand**
- Policy value under intervention, V^pi.

**Assumptions**
- Sequential ignorability and overlap.

**Method**
- Apply cumulative importance ratios at each time step and sum weighted rewards.

**Diagnostics**
- Overlap metrics, ESS, and weight tail statistics.

**Failure modes**
- Variance still grows with horizon under weak overlap.

**API**
- `crl.estimators.PDISEstimator`

**References**
- Jiang and Li (2016).
