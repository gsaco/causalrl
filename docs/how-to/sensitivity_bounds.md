# Sensitivity Bounds

Compute partial-identification bounds for bandit OPE when ignorability may fail.

```python
from crl.sensitivity.bandits import sensitivity_bounds

bounds = sensitivity_bounds(dataset, policy, gammas=[1.0, 1.5, 2.0])
```

Plot the curve:

```python
from crl.viz.plots import plot_sensitivity_curve

rows = [
    {"gamma": g, "lower": lo, "upper": up}
    for g, lo, up in zip(bounds.gammas, bounds.lower, bounds.upper)
]
plot_sensitivity_curve(rows)
```
