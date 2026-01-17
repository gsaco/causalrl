# Run Benchmarks

Benchmarks are configured with YAML files under `configs/benchmarks/`.

```bash
python -m experiments.run_benchmarks --suite all --out results/
```

Outputs include `results.csv`, `aggregate.csv`, figures, and an HTML summary.
