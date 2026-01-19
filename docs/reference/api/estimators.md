# Estimators

Estimator results are returned as `EstimatorReport` objects with a stable schema
and export utilities:

- `report.to_dict()` includes `schema_version`, `value`, `stderr`, `ci`, and diagnostics.
- `report.to_dataframe()` produces a one-row pandas table.
- `report.save_json(path)` and `report.save_html(path)` persist reports.

::: crl.estimators
