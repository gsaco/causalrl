# Weighted Importance Sampling (WIS)

**Estimand**
- Policy value under intervention, V^pi.

**Assumptions**
- Sequential ignorability and overlap.

**Method**
- Normalize importance weights to reduce variance.

**Diagnostics**
- Overlap metrics, ESS, and weight tail statistics.

**Failure modes**
- Biased when weights are normalized in small samples.

**API**
- `crl.estimators.WISEstimator`

**References**
- Uehara, Shi, Kallus (2022).
