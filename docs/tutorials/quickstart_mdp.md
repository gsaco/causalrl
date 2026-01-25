# Quickstart (MDP)

This tutorial covers finite-horizon OPE in an MDP setting. It demonstrates how
trajectory data changes estimator behavior and diagnostics.

## What you will do

- Sample trajectories from the synthetic MDP benchmark.
- Compare importance-sampling and model-based estimators.
- Generate publication-ready plots.

## Walkthrough

```python
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.ope import evaluate_ope

benchmark = SyntheticMDP(SyntheticMDPConfig(seed=0, horizon=5))
dataset = benchmark.sample(num_trajectories=300, seed=1)
report = evaluate_ope(dataset=dataset, policy=benchmark.target_policy)
```

## Notebook

- [03_mdp_ope_walkthrough.ipynb](https://github.com/gsaco/causalrl/blob/main/notebooks/03_mdp_ope_walkthrough.ipynb)
