# MAGIC

Implementation: `crl.estimators.magic.MAGICEstimator`

## Estimand

$V^\pi = \mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t r_t\right]$.

## Assumptions

- Sequential ignorability
- Overlap/positivity
- Markov property

## Inputs required

- `TrajectoryDataset`
- `behavior_action_probs` for logged actions
- Q-model fit (linear by default in CRL)

## Algorithm

MAGIC mixes truncated DR estimators over multiple horizons to reduce variance.

## Formula (sketch)

For truncation horizon $m$:

$\hat V_m = \hat V(s_0) + \sum_{t=0}^{m-1} \gamma^t \rho_t\left(r_t + \gamma \hat V(s_{t+1}) - \hat Q(s_t,a_t)\right)$.

MAGIC returns a variance-weighted mixture of $\hat V_m$ across horizons.

## Diagnostics

- `overlap.support_violations`
- `ess.ess_ratio`
- `model.q_model_mse`

## Uncertainty

- Normal-approximation CI by default.
- Bootstrap CI available via `bootstrap=True`.

## Failure modes

- Sensitive to poor Q-models and heavy-tailed ratios.
- Mixture weights can be unstable in small samples.

## Minimal example

```python
from crl.estimators.magic import MAGICEstimator

report = MAGICEstimator(estimand).estimate(dataset)
```

## References

- Thomas & Brunskill (2016)
