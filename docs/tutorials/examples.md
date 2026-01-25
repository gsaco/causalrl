# Examples

Use these examples as a starting point. They are grounded in the code and are
kept minimal so you can adapt them quickly.

## Quickstart scripts (recommended)

- Bandit OPE: [`examples/quickstart/bandit_ope.py`](https://github.com/gsaco/causalrl/blob/main/examples/quickstart/bandit_ope.py)
- MDP OPE: [`examples/quickstart/mdp_ope.py`](https://github.com/gsaco/causalrl/blob/main/examples/quickstart/mdp_ope.py)

Run them locally:

```bash
python -m examples.quickstart.bandit_ope
python -m examples.quickstart.mdp_ope
```

## 10-line end-to-end example

```python
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.ope import evaluate_ope

benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = benchmark.sample(num_samples=500, seed=1)
report = evaluate_ope(dataset=dataset, policy=benchmark.target_policy)

print(report.summary_table())
```

## Notebook gallery (source)

The repo includes notebooks that walk through diagnostics, sensitivity analysis,
and estimator selection. Each notebook is paired with a `.py` script via
Jupytext so you can diff and run them easily. See the source directory:

- [`notebooks/`](https://github.com/gsaco/causalrl/tree/main/notebooks)

Recommended notebooks:

- `00_introduction.ipynb` — estimand-first tour + report export
- `02_bandit_ope_walkthrough.ipynb` — diagnostics-driven bandit OPE
- `03_mdp_ope_walkthrough.ipynb` — trajectory estimators and horizon effects
- `10_bandit_ope_end_to_end.ipynb` — full workflow with sensitivity analysis
