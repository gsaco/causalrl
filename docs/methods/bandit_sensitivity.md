# Bandit Propensity Sensitivity

**Estimand**
- Policy value bounds under bounded confounding.

**Assumptions**
- Logged propensities are within a multiplicative factor gamma of the truth.

**Method**
- Adjust importance weights by gamma to compute lower and upper bounds.

**Diagnostics**
- Robustness curve of bounds over gamma values.

**Failure modes**
- Bounds may be conservative and do not imply point identification.

**API**
- `crl.sensitivity.BanditPropensitySensitivity`

**References**
- Namkoong et al. (2020).
