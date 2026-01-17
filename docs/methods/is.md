# Importance Sampling (IS)

**Estimand**
- Policy value under intervention, V^pi.

**Assumptions**
- Sequential ignorability and overlap.

**Method**
- Weight each trajectory return by the product of target/behavior action ratios.

**Diagnostics**
- Overlap metrics, ESS, and weight tail statistics.

**Failure modes**
- High variance and instability under weak overlap.

**API**
- `crl.estimators.ISEstimator`

**References**
- Robins, Hernan, Brumback (2000).
