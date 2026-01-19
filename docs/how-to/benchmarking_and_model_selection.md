# Benchmarking and Model Selection

Synthetic benchmarks are the fastest way to compare estimators and debug
workflows.

## Run the benchmark suite

```bash
python -m experiments.run_benchmarks --suite all --output-dir results/
```

Or run the canonical suite:

```bash
python -m experiments.run_benchmarks --config configs/benchmark_suites/default.yaml --seed 0
```

## Read the outputs

- `results/results.csv`
- `results/aggregate.csv`
- `results/report.html`
- `results/metadata.json`

## Use the estimator selection guide

Start with the decision tree and adjust based on diagnostics:

- [Estimator Selection Guide](../explanation/estimator_selection.md)

## Practical tips

- Always compare against ground truth on synthetic benchmarks.
- Keep configs and seeds fixed when comparing estimators.
