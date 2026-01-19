# Installation

CausalRL is available as a Python package (`causalrl`) and imported as `crl`.

## Install from PyPI

```bash
python -m pip install causalrl
```

## Install from source

```bash
git clone https://github.com/gsaco/causalrl
cd causalrl
python -m pip install -e .
```

## Optional extras

Choose extras when you need them:

```bash
python -m pip install "causalrl[docs]"
python -m pip install "causalrl[benchmarks]"
python -m pip install "causalrl[notebooks]"
python -m pip install "causalrl[behavior]"
python -m pip install "causalrl[d4rl]"
python -m pip install "causalrl[rlu]"
```

## Sanity check

```bash
python - <<'PY'
import crl
print(crl.__version__)
PY
```
