# Trajectory Dataset from Parquet

This guide shows how to build a `TrajectoryDataset` when your trajectories are
stored in a parquet file.

## Recommended layout

Store one row per (episode, timestep):

- `episode_id` (int)
- `timestep` (int)
- `obs_*` columns (state features)
- `action` (int)
- `reward` (float)
- `next_obs_*` columns (next state features)
- `behavior_prob` (optional)

## Example

```python
import pandas as pd
from crl.data.datasets import TrajectoryDataset

# Example parquet format: one row per step
steps = pd.read_parquet("trajectories.parquet")
obs_cols = [c for c in steps.columns if c.startswith("obs_")]
next_cols = [c for c in steps.columns if c.startswith("next_obs_")]

trajectories = TrajectoryDataset.from_dataframe(
    steps,
    observation_columns=obs_cols,
    next_observation_columns=next_cols,
    action_column="action",
    reward_column="reward",
    behavior_prob_column="behavior_prob",
    discount=0.99,
    action_space_n=4,
)
```

## Notes

- `read_parquet` requires a parquet engine (pyarrow or fastparquet).
- Ensure each episode has the same horizon when reshaping.
- If horizons vary, pad and set `mask` to indicate valid steps.
