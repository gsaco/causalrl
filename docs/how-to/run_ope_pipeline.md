# Run the OPE Pipeline

Use the high-level pipeline to evaluate a policy end-to-end.

## Python API

```python
from crl.ope import evaluate

report = evaluate(dataset=dataset, policy=policy, estimators="default")
report.to_dataframe()
report.save_html("report.html")
```

## CLI

```bash
crl ope --config configs/ope.yaml --out results/run_001/
```

See `configs/ope.yaml` for the expected schema.
