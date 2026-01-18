# Run the OPE Pipeline

Use the high-level pipeline to evaluate a policy end-to-end.

## Python API

```python
from crl.ope import evaluate

report = evaluate(dataset=dataset, policy=policy, estimators="default")
report.to_dataframe()
report.save_html("report.html")
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

See `configs/ope.yaml` for the expected schema.
