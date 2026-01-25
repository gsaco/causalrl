# Logged Bandit from a DataFrame

This guide shows how to build a `LoggedBanditDataset` from a pandas DataFrame.

## Expected columns

You need:

- context features (one or more columns)
- `action` column with integer action indices
- `reward` column
- optional `behavior_prob` column for logged propensities

## Example

```python
from crl.data.datasets import LoggedBanditDataset

# df columns: ["x1", "x2", "action", "reward", "behavior_prob"]
bandit = LoggedBanditDataset.from_dataframe(
    df,
    context_columns=["x1", "x2"],
    action_column="action",
    reward_column="reward",
    behavior_prob_column="behavior_prob",
)
```

## Common pitfalls

- **Action indices are not 0..K-1**: remap your actions before creating the dataset.
- **Propensities missing**: IS/WIS require `behavior_action_probs`. See
  [Behavior Propensities Missing](behavior_propensities_missing.md).
- **Contexts are mixed types**: cast to numeric or one-hot encode categorical features.
