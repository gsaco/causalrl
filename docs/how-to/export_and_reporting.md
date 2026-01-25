# Export and Reporting

CausalRL provides machine-readable and human-readable outputs.

## EstimatorReport

```python
report = estimator.estimate(dataset)

# Python structures
payload = report.to_dict()
json_text = report.to_json()
frame = report.to_dataframe()

# Files
report.save_json("estimate.json")
report.save_html("estimate.html")
```

The JSON payload includes a `schema_version` field for downstream parsing.

## OpeReport (pipeline)

```python
from crl.ope import evaluate_ope

report = evaluate_ope(dataset=dataset, policy=policy)
report.to_dataframe()  # summary table
report.to_html("report.html")
```

## Recommendations

- Store both JSON and HTML artifacts for reproducibility.
- Save the diagnostics alongside estimates to preserve the trust story.
