# DualDICE

## Assumptions

- Sequential ignorability
- Markov

## Requires

- TrajectoryDataset with discrete `state_space_n`
- Stable state-action feature representation

## Diagnostics to check

- `model` diagnostics from upstream modeling (DualDICE itself is diagnostic-light)

## Formula

DualDICE solves a density ratio estimation problem for the discounted
stationary distribution ratio $w = d_\pi / d_\mu$ and estimates

$V^\pi \approx \frac{1}{1-\gamma} \mathbb{E}_{d_\mu}[w(s,a) r(s,a)]$.

## Fails when

- Requires stable density ratio estimation.
- Sensitive to feature misspecification for function approximation.

## Minimal example

```python
from crl.estimators.dual_dice import DualDICEEstimator

report = DualDICEEstimator(estimand).estimate(dataset)
```

## References

- Nachum et al. (2019)

## Notebook

- [03_mdp_ope_walkthrough.ipynb](https://github.com/gsaco/causalrl/blob/v4/notebooks/03_mdp_ope_walkthrough.ipynb)
