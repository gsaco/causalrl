# Configuration Schemas

## OPE config (`configs/ope.yaml`)

```yaml
benchmark:
  type: bandit | mdp
  seed: 0
  num_samples: 1000        # bandit only
  num_trajectories: 200    # mdp only
  config: {}               # passed to SyntheticBanditConfig/SyntheticMDPConfig
estimators: ["is", "wis", "double_rl"]
diagnostics: default
seed: 0
```

## Benchmark suite configs (`configs/benchmarks/*.yaml`)

```yaml
suite: horizon
benchmarks:
  - name: mdp_short_horizon
    type: mdp
    num_trajectories: 200
    estimators: ["dr", "wdr", "fqe"]
    behavior_known: true
    config:
      horizon: 3
```
