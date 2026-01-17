# Fitted Q Evaluation (FQE)

**Estimand**
- Policy value under intervention, V^pi.

**Assumptions**
- Markov property, sequential ignorability, overlap, and correct function approximation.

**Method**
- Fit a Q-function by regression on Bellman targets under the target policy.

**Diagnostics**
- Overlap metrics and Bellman residual checks.

**Failure modes**
- Extrapolation error for out-of-distribution actions.

**API**
- `crl.estimators.FQEEstimator`

**References**
- Uehara, Shi, Kallus (2022).
