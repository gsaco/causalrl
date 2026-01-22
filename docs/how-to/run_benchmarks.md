# Run Benchmarks

Benchmarks are configured with YAML files under `configs/benchmarks/`.

```bash
python -m experiments.run_benchmarks --suite all --output-dir results/
```

To run a pinned suite:

```bash
python -m experiments.run_benchmarks --config configs/benchmark_suites/default.yaml --seed 0
```

## CI vs full runs

- Smoke benchmarks (CI): `make benchmarks-smoke`
- Full benchmarks (manual): `make benchmarks-full`

Outputs include `results.csv`, `aggregate.csv`, figures, `metadata.json`, and an HTML summary.
