# Correct Model Specification

**Assumption**
- The function approximator used for Q or reward models is well specified.

**Applies to**
- DR and FQE.

**Definition**
- The regression model class contains the true conditional expectations.

**Diagnostics**
- Residual plots, cross-validation, and sensitivity to model class.

**Failure modes**
- Bias if model misspecification persists across folds.

**References**
- Kallus and Uehara (2020).
