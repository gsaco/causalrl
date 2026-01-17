# Examples

This directory contains runnable scripts that demonstrate common CausalRL
workflows. Each script is self-contained and uses synthetic benchmarks with
known ground truth so you can compare estimators directly.

## Quickstart

- `examples/quickstart/bandit_ope.py`: contextual bandit OPE with IS and WIS,
  showing how to set estimands and assumptions.
- `examples/quickstart/mdp_ope.py`: MDP OPE with IS, WIS, PDIS, DR, and FQE,
  illustrating trajectory-based evaluation and estimator comparison.

Run the scripts with:

```bash
python examples/quickstart/bandit_ope.py
python examples/quickstart/mdp_ope.py
```
