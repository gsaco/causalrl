# Dataset Format and Validation

This page is the single source of truth for how to build datasets that CausalRL
accepts. Every estimator and diagnostic assumes these contracts are respected.

## Bandit data: LoggedBanditDataset

Use `LoggedBanditDataset` for contextual bandits (one action per context).

Required fields and shapes:

- `contexts`: shape `(n, d)` or `(n,)`.
- `actions`: shape `(n,)`, integer action indices.
- `rewards`: shape `(n,)`.
- `behavior_action_probs`: optional shape `(n,)`, values in `[0, 1]`.
- `action_space_n`: positive integer.

Minimal example:

```python
import numpy as np
from crl.data.datasets import LoggedBanditDataset

contexts = np.random.normal(size=(100, 5))
actions = np.random.randint(0, 3, size=100)
rewards = np.random.normal(size=100)
behavior_probs = np.full(100, 1.0 / 3.0)

dataset = LoggedBanditDataset(
    contexts=contexts,
    actions=actions,
    rewards=rewards,
    behavior_action_probs=behavior_probs,
    action_space_n=3,
)
```

Convenience constructors:

```python
dataset = LoggedBanditDataset.from_numpy(
    contexts=contexts,
    actions=actions,
    rewards=rewards,
    behavior_action_probs=behavior_probs,
)
```

Validation highlights:

- `actions` must be integers in `[0, action_space_n)`.
- `behavior_action_probs` must be in `[0, 1]` and finite.
- All arrays must have the same length.

## Trajectory data: TrajectoryDataset

Use `TrajectoryDataset` for finite-horizon episodes.

Required fields and shapes:

- `observations`: shape `(n, t, d)` or `(n, t)`.
- `actions`: shape `(n, t)`, integer action indices.
- `rewards`: shape `(n, t)`.
- `next_observations`: same shape as `observations`.
- `mask`: shape `(n, t)`, boolean. Valid steps must be a prefix.
- `behavior_action_probs`: optional shape `(n, t)`, values in `[0, 1]` on mask.
- `discount`: float in `[0, 1]`.
- `action_space_n`: positive integer.
- `state_space_n`: optional integer (only for discrete states).

Minimal example:

```python
import numpy as np
from crl.data.datasets import TrajectoryDataset

num_traj = 10
horizon = 5
obs = np.random.normal(size=(num_traj, horizon, 3))
next_obs = np.random.normal(size=(num_traj, horizon, 3))
actions = np.random.randint(0, 4, size=(num_traj, horizon))
rewards = np.random.normal(size=(num_traj, horizon))
mask = np.ones((num_traj, horizon), dtype=bool)
behavior_probs = np.full((num_traj, horizon), 1.0 / 4.0)

dataset = TrajectoryDataset(
    observations=obs,
    actions=actions,
    rewards=rewards,
    next_observations=next_obs,
    behavior_action_probs=behavior_probs,
    mask=mask,
    discount=0.99,
    action_space_n=4,
)
```

Convenience constructors:

```python
dataset = TrajectoryDataset.from_dataframe(
    df,
    observation_columns=["obs"],
    next_observation_columns=["next_obs"],
    behavior_prob_column="propensity",
    discount=0.99,
    action_space_n=4,
)
```

Validation highlights:

- `mask` must be contiguous: valid steps form a prefix for each trajectory.
- `actions`, `rewards`, `mask`, and `behavior_action_probs` (if present) must
  share shape `(n, t)`.
- `actions` must be integer indices in `[0, action_space_n)` on valid steps.

## Transition data: TransitionDataset

Use `TransitionDataset` for flat `(s, a, r, s', done)` logs.

Required fields and shapes:

- `states`: shape `(n, d)` or `(n,)`.
- `actions`: shape `(n,)` (or `(n, d)` for continuous actions).
- `rewards`: shape `(n,)`.
- `next_states`: same shape as `states`.
- `dones`: shape `(n,)`.
- `behavior_action_probs`: optional shape `(n,)`.
- `discount`: float in `[0, 1]`.
- `action_space_n`: optional; required for discrete actions.
- `episode_ids` and `timesteps`: optional; required for `to_trajectory()`.

Minimal example:

```python
import numpy as np
from crl.data.transition import TransitionDataset

states = np.random.normal(size=(100, 4))
next_states = np.random.normal(size=(100, 4))
actions = np.random.randint(0, 2, size=100)
rewards = np.random.normal(size=100)
dones = np.zeros(100, dtype=bool)
behavior_probs = np.full(100, 0.5)

dataset = TransitionDataset(
    states=states,
    actions=actions,
    rewards=rewards,
    next_states=next_states,
    dones=dones,
    behavior_action_probs=behavior_probs,
    discount=0.99,
    action_space_n=2,
)
```

To convert transitions to trajectories:

```python
traj = dataset.to_trajectory()
```

This requires `episode_ids` and `timesteps` to be present and discrete actions.

## Common validation errors and fixes

- **Actions out of range**: check `action_space_n` and action indexing.
- **Mask not contiguous**: ensure each trajectory uses a prefix of valid steps.
- **Missing propensities**: IS/PDIS/DR require them; FQE can run without but
  diagnostics will be limited.
- **Shape mismatch**: verify `(n, t)` or `(n, t, d)` consistency across arrays.

## Researcher checklist

- Confirm the dataset contract for your estimator.
- Log behavior propensities whenever possible.
- Inspect diagnostics before trusting estimates.
- Use synthetic benchmarks to sanity check workflows.
