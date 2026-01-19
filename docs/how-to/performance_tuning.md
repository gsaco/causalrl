# Performance Tuning

This guide focuses on practical knobs that reduce runtime and memory.

## FQE training knobs

`FQEConfig` exposes the main levers:

- `hidden_sizes`: network width and depth
- `batch_size`: lower to fit in memory
- `num_epochs`: reduce for faster iterations
- `num_iterations`: fewer fitted-Q iterations
- `weight_decay`: regularization for stability
- `seed`: reproducibility

Example:

```python
from crl.estimators.fqe import FQEConfig, FQEEstimator

config = FQEConfig(
    hidden_sizes=(32, 32),
    batch_size=64,
    num_epochs=5,
    num_iterations=5,
)

estimator = FQEEstimator(estimand, config=config, device="cpu")
```

## Diagnostics cost

Diagnostics can add overhead. If you only need estimates:

```python
estimator = FQEEstimator(estimand, run_diagnostics=False)
```

## Bootstrap CI cost

Bootstrap can be expensive. Reduce the number of resamples:

```python
from crl.estimators.bootstrap import BootstrapConfig

estimator = FQEEstimator(
    estimand,
    bootstrap=True,
    bootstrap_config=BootstrapConfig(num_bootstrap=50),
)
```
