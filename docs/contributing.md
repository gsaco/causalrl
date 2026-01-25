# Contributing

Thanks for contributing to CausalRL. This page mirrors the essentials from
`CONTRIBUTING.md`.

## Development setup

```bash
python -m pip install -e ".[dev]"
```

## Tests and checks

```bash
pytest -q
ruff check .
mypy crl
mkdocs build
```

## Pull requests

- Keep changes focused and add tests for new functionality.
- Document new estimators with method and assumption cards.
- Ensure benchmarks remain deterministic.

Full guide: https://github.com/gsaco/causalrl/blob/main/CONTRIBUTING.md
