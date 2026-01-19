# Integrate Logged Data

This guide shows the shortest path from real logs to a usable OPE report.

## Step 1: Build a dataset

For contextual bandits, use `LoggedBanditDataset.from_dataframe`:

```python
from crl.data.datasets import LoggedBanditDataset

bandit = LoggedBanditDataset.from_dataframe(
    df,
    context_columns=["x1", "x2"],
    action_column="action",
    reward_column="reward",
    behavior_prob_column="propensity",
)
```

For trajectories, use `TrajectoryDataset.from_dataframe` with long-form rows:

```python
from crl.data.datasets import TrajectoryDataset

traj = TrajectoryDataset.from_dataframe(
    df,
    observation_columns=["obs"],
    next_observation_columns=["next_obs"],
    action_column="action",
    reward_column="reward",
    behavior_prob_column="propensity",
    discount=0.99,
    action_space_n=4,
)
```

## Step 2: Wrap your policy

If you already have a model that outputs probabilities:

```python
from crl.policies.base import Policy

policy = Policy.from_sklearn(model, action_space_n=4)
```

If you have a torch model returning logits:

```python
from crl.policies.base import Policy

policy = Policy.from_torch(model, action_space_n=4, device="cpu")
```

## Step 3: Run evaluation

```python
from crl.ope import evaluate

report = evaluate(dataset=bandit, policy=policy)
summary = report.to_dataframe()
```

## Missing propensities

If `behavior_action_probs` are unavailable, `evaluate` will skip propensity-based
estimators and fall back to estimators that do not require them (for MDPs, FQE).
Use diagnostics cautiously and prefer logging propensities in production.
