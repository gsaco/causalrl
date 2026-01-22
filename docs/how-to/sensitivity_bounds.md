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

## Sequential sensitivity (Gamma model)

For trajectories, use the Namkoong-style bounds:

```python
from crl.sensitivity.namkoong2020 import confounded_ope_bounds

curve = confounded_ope_bounds(dataset, policy, gammas=[1.0, 1.25, 1.5, 2.0])
```

## Estimand-first integration

You can bundle sensitivity into the OPE pipeline with a sensitivity estimand:

```python
import numpy as np
from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import BOUNDED_CONFOUNDING
from crl.estimands.sensitivity_policy_value import SensitivityPolicyValueEstimand
from crl.ope import evaluate

sensitivity = SensitivityPolicyValueEstimand(
    policy=policy,
    discount=dataset.discount,
    horizon=dataset.horizon,
    gammas=np.linspace(1.0, 2.0, 6),
    assumptions=AssumptionSet([BOUNDED_CONFOUNDING]),
)
report = evaluate(dataset=dataset, policy=policy, sensitivity=sensitivity)
```
