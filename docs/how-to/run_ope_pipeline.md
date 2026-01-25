# Run the OPE Pipeline

Use the high-level pipeline to evaluate a policy end-to-end.

## Python API

```python
from crl.ope import evaluate_ope

report = evaluate_ope(dataset=dataset, policy=policy, estimators="default")
report.to_dataframe()
report.save_html("report.html")
```

## Sensitivity integration

```python
import numpy as np
from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import BOUNDED_CONFOUNDING
from crl.estimands.sensitivity_policy_value import SensitivityPolicyValueEstimand

sensitivity = SensitivityPolicyValueEstimand(
    policy=policy,
    discount=dataset.discount,
    horizon=dataset.horizon,
    gammas=np.linspace(1.0, 2.0, 6),
    assumptions=AssumptionSet([BOUNDED_CONFOUNDING]),
)
report = evaluate_ope(dataset=dataset, policy=policy, sensitivity=sensitivity)
```

## Estimated behavior policy

```python
from crl.behavior import fit_behavior_policy

fit = fit_behavior_policy(dataset, method="logit")
dataset = fit.apply(dataset)
```

## CLI

```bash
crl ope --config configs/ope.yaml --out results/run_001/
```

Alias:

```bash
crl --config configs/ope.yaml --out results/run_001/
```

See `configs/ope.yaml` for the expected schema.
