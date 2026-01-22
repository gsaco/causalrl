# Double Reinforcement Learning (MDP)

Implementation: `crl.estimators.drl.DRLEstimator`

## Estimand

$V^\pi = \mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t r_t\right]$.

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov property

## Requires

- `TrajectoryDataset` with `state_space_n`

## Algorithm

1. Estimate a Q-function for the target policy.
2. Estimate per-time state-action density ratios.
3. Combine the Q-model and ratio-weighted TD correction.

## Uncertainty

- Normal-approximation CI by default.
- Bootstrap CI available via `bootstrap=True`.

## Failure modes

- Sparse coverage can destabilize ratio estimates.
- Misspecified Q-model or density ratios can induce bias.

## Minimal example

```python
from crl.estimators.drl import DRLEstimator

report = DRLEstimator(estimand).estimate(dataset)
```

## References

- Kallus & Uehara (2020)
