# MDP OPE Tutorial (Script)

This mirrors `examples/quickstart/mdp_ope.py` and focuses on trajectory-based
estimators in a finite-horizon MDP.

## Run the script

```bash
python -m examples.quickstart.mdp_ope
```

## What it does

1. Generates synthetic trajectories with known ground truth.
2. Declares Markov and overlap assumptions.
3. Compares IS, WIS, PDIS, DR, and FQE.
4. Prints a summary table for quick inspection.

## Interpretation tips

- **Horizon effects**: ESS often decays over time.
- **DR/FQE**: lower variance if the value model is well specified.
- **PDIS**: more stable than full-trajectory IS in long horizons.

## Next steps

- See the long-horizon comparison notebook: `notebooks/11_mdp_long_horizon_comparison.ipynb`
- Review estimator selection guidance: [Estimator Selection Guide](../explanation/estimator_selection.md)
