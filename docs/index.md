# causalrl

causalrl is an estimand-first toolkit for causal reinforcement learning and
off-policy evaluation. The API emphasizes identification assumptions,
automatic diagnostics, and benchmark-driven development.

## Highlights

- Explicit estimands and assumption sets for each estimator
- Diagnostics-first reports with overlap and ESS metrics
- Synthetic benchmarks with known ground truth
- Focused dependencies with a PyTorch backend for model-based baselines

## Installation

```bash
python -m pip install causalrl
```

Optional extras:

```bash
python -m pip install "causalrl[docs]"
python -m pip install "causalrl[benchmarks]"
```

## Getting started

- Run the quickstart scripts in `examples/quickstart/`
- Work through the rendered notebooks under the Notebooks section
- Browse the API reference for full docstrings
