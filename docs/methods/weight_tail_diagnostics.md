# Weight Tail Diagnostics

**Estimand**
- Not applicable.

**Assumptions**
- Non-negative importance weights.

**Method**
- Report tail quantiles and extreme-weight fractions.

**Diagnostics**
- Maximum weight, q99, and tail fraction above a threshold.

**Failure modes**
- Tail summaries can understate multi-modal weight behavior.

**API**
- `crl.diagnostics.weight_tail_stats`

**References**
- Uehara, Shi, Kallus (2022).
