# GenDICE

Implementation: `crl.estimators.gen_dice.GenDICEEstimator`

## Assumptions

- Sequential ignorability
- Markov property

## Requires

- `TrajectoryDataset` with `state_space_n`

## Algorithm

Generalized density-ratio estimation with linear features and regularization.

## Uncertainty

- CI reported but may be unreliable for density-ratio estimators; interpret cautiously.

## Failure modes

- Sparse state-action coverage can destabilize ratio estimates.

## Minimal example

```python
from crl.estimators.gen_dice import GenDICEEstimator

report = GenDICEEstimator(estimand).estimate(dataset)
```

## References

- Yang et al. (2020)
