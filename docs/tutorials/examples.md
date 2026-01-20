# Examples

Use these examples as a starting point. They are grounded in the code and are
kept minimal so you can adapt them quickly.

## Quickstart scripts (recommended)

- Bandit OPE: [`examples/quickstart/bandit_ope.py`](https://github.com/gsaco/causalrl/blob/v4/examples/quickstart/bandit_ope.py)
- MDP OPE: [`examples/quickstart/mdp_ope.py`](https://github.com/gsaco/causalrl/blob/v4/examples/quickstart/mdp_ope.py)

Run them locally:

```bash
python examples/quickstart/bandit_ope.py
python examples/quickstart/mdp_ope.py
```

## 10-line end-to-end example

```python
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.ope import evaluate

benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = benchmark.sample(num_samples=500, seed=1)
report = evaluate(dataset=dataset, policy=benchmark.target_policy)

print(report.summary_table())
```

## Notebook gallery (source)

The repo includes notebooks that walk through diagnostics, sensitivity analysis,
and estimator selection. See the source directory:

- [`notebooks/`](https://github.com/gsaco/causalrl/tree/v4/notebooks)
