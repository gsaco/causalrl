# Estimator Selection Guide

Use this guide to pick a default estimator before tuning.

```text
Do you know behavior propensities?
  yes -> Short horizon: IS or WIS
      -> Long horizon: PDIS, DR, WDR, or MIS
  no  -> FQE, DualDICE, GenDICE, or DRL
Need confidence bounds?
  -> High-confidence IS or bootstrap FQE
Concern about overlap?
  -> Inspect ESS and weight diagnostics first
Concern about confounding?
  -> Sensitivity analysis or proximal OPE
```

## Recommended defaults

- Bandit: WIS + Double RL for a robust baseline.
- MDP: WDR + FQE or DRL, then check diagnostics.

## Diagnostics-driven selection

You can also use the selector API to rank candidates using stability heuristics:

```python
from crl.selectors import select_estimator

best = select_estimator(dataset, estimand, candidates=["is", "wis", "dr", "wdr", "fqe"]) 
```
