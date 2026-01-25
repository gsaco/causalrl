# DualDICE

Implementation: `crl.estimators.dual_dice.DualDICEEstimator`

## Assumptions

- Sequential ignorability
- Markov property

## Requires

- `TrajectoryDataset` with `state_space_n`

## Diagnostics to check

- Density-ratio convergence (if available)

## Formula

DualDICE estimates the discounted occupancy ratio and reweights rewards.

## Uncertainty

- CI reported but may be unreliable for density-ratio estimators; interpret cautiously.

## Failure modes

- Sparse state-action coverage can destabilize ratio estimates.
- Dense feature construction scales as `O((S·A)^2)` memory; large discrete spaces can be impractical.

## Scaling notes

DualDICE constructs dense one-hot features for all state-action pairs. When
`state_space_n * action_space_n` grows large, memory usage can explode.
The implementation emits a warning above a conservative threshold to flag
potential OOM risk.

## Minimal example

```python
from crl.estimators.dual_dice import DualDICEEstimator

report = DualDICEEstimator(estimand).estimate(dataset)
```

## References

- Nachum et al. (2019)
