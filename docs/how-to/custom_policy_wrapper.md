# Custom Policy Wrapper

Policies must provide action probabilities for each observation.

## Option 1: wrap a callable

Use `StochasticPolicy` when you already have a function returning probabilities.

```python
import numpy as np
from crl.policies.discrete import StochasticPolicy

def prob_fn(obs: np.ndarray) -> np.ndarray:
    # return shape (n, action_space_n)
    logits = np.ones((obs.shape[0], 3))
    return logits / logits.sum(axis=1, keepdims=True)

policy = StochasticPolicy(prob_fn=prob_fn, action_space_n=3)
```

## Option 2: deterministic callable

Use `CallablePolicy` when you have a function that returns action indices.

```python
import numpy as np
from crl.policies.discrete import CallablePolicy

def action_fn(obs: np.ndarray) -> np.ndarray:
    return np.zeros(obs.shape[0], dtype=int)

policy = CallablePolicy(action_fn=action_fn, action_space_n=3, returns="actions")
```

## Option 3: sklearn or torch models

```python
from crl.policies.base import Policy

# scikit-learn classifier with predict_proba
policy = Policy.from_sklearn(model, action_space_n=3)

# torch model returning logits
policy = Policy.from_torch(torch_model, action_space_n=3, device="cpu")
```

## Option 4: implement a minimal class

```python
import numpy as np
from crl.policies.base import Policy

class MyPolicy(Policy):
    def action_probs(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations)
        if obs.ndim == 1:
            obs = obs.reshape(-1, 1)
        probs = np.ones((obs.shape[0], 2)) * 0.5
        return probs
```

## Notes

- `action_probs` must return shape `(n, num_actions)`.
- Rows must sum to 1.
- Use integer action indices when building datasets.
