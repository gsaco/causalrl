# Behavior Policy Known

This assumption states that the behavior policy propensities used by estimators
are known (logged) or correctly specified if estimated. Without it, importance-
weighted estimators can be biased even when overlap holds.

## When it matters

- IS / WIS / PDIS
- DR / WDR / MRDR / MAGIC
- High-confidence OPE bounds

If you estimate propensities, document the model and treat this assumption as
additional modeling risk rather than a guarantee.
