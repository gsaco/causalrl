# Overlap (Positivity)

**Assumption**
- The behavior policy assigns non-zero probability to actions the target policy takes.

**Applies to**
- IS, WIS, PDIS, DR, FQE.

**Definition**
- For all states with positive probability under the target policy, the behavior
  policy probability of those actions is bounded away from zero.

**Diagnostics**
- Overlap metrics and weight tail diagnostics.

**Failure modes**
- Importance weights explode and variance dominates.

**References**
- Robins, Hernan, Brumback (2000).
