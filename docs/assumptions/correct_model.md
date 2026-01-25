# Q-Model Realizability

**Assumption**
- The function approximator used for Q or reward models is well specified.

**Applies to**
- FQE and other model-based estimators.
- DR-family estimators benefit when the Q-model is accurate, but they do not require this assumption for consistency.

**Definition**
- The regression model class contains the true conditional expectations.

**Diagnostics**
- Residual plots, cross-validation, and sensitivity to model class.

**Failure modes**
- Bias if model misspecification persists across folds.

**References**
- Kallus and Uehara (2020).
