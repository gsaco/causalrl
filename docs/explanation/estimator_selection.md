# Estimator Selection Guide

Use this guide to pick a default estimator before tuning.

```text
Do you know behavior propensities?
  yes -> Short horizon: IS or WIS
      -> Long horizon: PDIS, DR, WDR, or FQE
  no  -> FQE or DualDICE (density ratio)
Need confidence bounds?
  -> High-confidence IS or bootstrap FQE
Concern about overlap?
  -> Inspect ESS and weight diagnostics first
Concern about confounding?
  -> Sensitivity analysis or proximal OPE
```

## Recommended defaults

- Bandit: WIS + Double RL for a robust baseline.
- MDP: WDR + FQE, then check diagnostics.
