# Benchmark Runner Tutorial

Use the benchmark harness to produce reproducible estimator comparisons on
synthetic datasets with known ground truth.

## Run the benchmark suite

```bash
python experiments/run_benchmarks.py --output-dir results
```

## What you get

- `results/benchmark_results.csv` for spreadsheet/plotting workflows.
- `results/benchmark_results.jsonl` for structured logging and reproducibility.

## Tips for reproducibility

- Pin seeds in your configs.
- Record estimator configs in the output metadata.
- Use the CI smoke benchmarks for quick regressions.
