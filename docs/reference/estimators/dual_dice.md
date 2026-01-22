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

## Minimal example

```python
from crl.estimators.dual_dice import DualDICEEstimator

report = DualDICEEstimator(estimand).estimate(dataset)
```

## References

- Nachum et al. (2019)
