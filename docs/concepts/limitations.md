# Limitations

CausalRL is an OPE toolkit, not a replacement for careful causal reasoning.

## When not to use it

- If you cannot justify sequential ignorability or any sensitivity model.
- If the behavior policy never explores the target policy's actions.
- If you need guaranteed policy optimization, not just evaluation.

## Common failure modes

- **Poor overlap** leads to unstable IS/PDIS estimates.
- **Model misspecification** can bias DR/FQE.
- **Long horizons** amplify variance for importance-weighted methods.

## Scope constraints

- `evaluate` currently assumes discrete action spaces; continuous-action OPE is not implemented yet.
- Transition-only logs must include `episode_id` and `timestep` to reconstruct trajectories, or you must build a `TrajectoryDataset` directly.

## What to do instead

- Log behavior propensities during data collection.
- Run diagnostics and show them alongside results.
- Use sensitivity analysis when confounding is plausible.

## Versioning caveat

This library is still in 0.x. Public APIs follow SemVer and deprecation
warnings, but breaking changes may still occur before 1.0.
