# Seeds and Configs

Reproducibility depends on both randomness control and configuration tracking.

## Set seeds explicitly

```python
from crl.utils.seeding import set_seed

set_seed(0)
```

## Record configs

- Keep benchmark config files under `configs/`.
- Save estimator configs in report metadata (many estimators do this already).
- The benchmark runner writes `metadata.json` with seeds and config paths.

## Recommended practice

- Store `seed`, `config`, and package version with every run.
- Prefer deterministic examples when validating changes.
