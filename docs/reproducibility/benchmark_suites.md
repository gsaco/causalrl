# Benchmark Suites

The benchmark runner produces comparable results across estimators.

## Run a suite

```bash
python -m experiments.run_benchmarks --suite all --output-dir results/
```

Or run the canonical suite:

```bash
python -m experiments.run_benchmarks --config configs/benchmark_suites/default.yaml --seed 0
```

## Outputs

- `results.csv` / `aggregate.csv`
- `report.html`
- `metadata.json` (git SHA, versions, seeds, config path)

## Best practices

- Pin seeds and config files.
- Store results with the git SHA and version.
- Use synthetic benchmarks to validate estimator changes.
