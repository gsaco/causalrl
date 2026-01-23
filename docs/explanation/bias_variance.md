# Bias-Variance Tradeoffs

IS-based estimators are unbiased when propensities are correct and overlap
holds, but they can have high variance. WIS/PDIS can be biased in finite
samples. Model-based estimators reduce variance but introduce model bias. DR
methods aim to balance both by combining model estimates with importance
weights.

Key takeaways:

- Long horizons magnify variance in IS/PDIS.
- WDR and MAGIC can stabilize estimates when models are reasonable.
- FQE is attractive when propensities are unknown but model fit is reliable.
