# Doubly Robust (DR)

**Estimand**
- Policy value under intervention, V^pi.

**Assumptions**
- Sequential ignorability, overlap, and correct model specification (one of Q or propensities).

**Method**
- Combine model-based Q estimates with importance-weighted residual correction.

**Diagnostics**
- Overlap metrics, ESS, and model residual checks.

**Failure modes**
- Bias when both models are misspecified; instability under weak overlap.

**API**
- `crl.estimators.DoublyRobustEstimator`

**References**
- Jiang and Li (2016).
- Kallus and Uehara (2020).
