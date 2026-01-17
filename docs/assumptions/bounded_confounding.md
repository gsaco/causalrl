# Bounded Confounding

**Assumption**
- Unobserved confounding is bounded by a sensitivity parameter.

**Applies to**
- Bandit propensity sensitivity analysis.

**Definition**
- True propensities deviate from logged propensities by at most a multiplicative
  factor gamma.

**Diagnostics**
- Sensitivity curves over gamma values.

**Failure modes**
- Bounds can be loose and are not point identified.

**References**
- Namkoong et al. (2020).
