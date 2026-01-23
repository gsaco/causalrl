# Environment

Keep your environment consistent to reproduce results.

## Python version

CausalRL targets Python 3.10+.

## Record dependencies

```bash
python -m pip freeze > requirements.txt
```

The benchmark runner also captures package versions in `metadata.json`.

## Optional extras

Install extras based on your workflow (source install):

```bash
python -m pip install -e ".[docs]"
```

Available extras:

- `.[docs]`
- `.[benchmarks]`
- `.[notebooks]`
- `.[behavior]`
