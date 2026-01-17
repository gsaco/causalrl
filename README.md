# causalrl

CausalRL is an estimand-first toolkit for causal reinforcement learning and
off-policy evaluation. It pairs identification assumptions with diagnostics so
you can trust or debug an estimate before using it.

Highlights:
- Explicit estimands and assumption sets for each estimator
- Diagnostics-first reports with overlap and ESS metrics
- Synthetic benchmarks with ground-truth policy values
- Baselines for IS, WIS, PDIS, DR, and FQE

## Installation

```bash
python -m pip install causalrl
```

Optional extras:

```bash
python -m pip install "causalrl[docs]"
python -m pip install "causalrl[benchmarks]"
```

## Documentation

- Live docs: https://gsaco.github.io/causalrl/
- Build locally: `mkdocs serve`

## Examples

Quickstart scripts:

```bash
python examples/quickstart/bandit_ope.py
python examples/quickstart/mdp_ope.py
```

Notebooks:

- `notebooks/01_bandit_ope.ipynb`
- `notebooks/02_mdp_ope.ipynb`
- `notebooks/03_diagnostics.ipynb`
- `notebooks/04_sensitivity.ipynb`

## Development

```bash
python -m pip install -e ".[dev]"
```

## Citation

If you use causalrl in your research, please cite it using
[`CITATION.cff`](CITATION.cff).

## Project status

See [`docs/project_status.md`](docs/project_status.md) for scope and roadmap.
