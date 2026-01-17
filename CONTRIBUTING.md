# Contributing

Contribution guidelines for causalrl.

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
