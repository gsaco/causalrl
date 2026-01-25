# Installation

CausalRL is imported as `crl`. PyPI publishing is planned; for now install from
source.

## PyPI status

Not yet published (as of January 23, 2026).

## Install from source

```bash
git clone https://github.com/gsaco/causalrl
cd causalrl
python -m pip install -e .
```

## Optional extras

Choose extras when you need them:

```bash
python -m pip install -e ".[docs]"
python -m pip install -e ".[benchmarks]"
python -m pip install -e ".[notebooks]"
python -m pip install -e ".[behavior]"
python -m pip install -e ".[d4rl]"
python -m pip install -e ".[rlu]"
```

!!! tip "Recommended extras"
    - Use `.[behavior]` if you need to estimate behavior propensities.
    - Use `.[notebooks]` to run the walkthrough notebooks locally.
    - Use `.[docs]` if you plan to build the documentation site.

!!! warning "Common pitfalls"
    - Behavior policy estimation requires `scikit-learn`.
    - FQE and some estimators require `torch`.
    - `evaluate` currently assumes discrete action spaces.
    - The walkthrough notebooks call behavior estimation; install `.[notebooks]` (includes scikit-learn) or `.[behavior]` before running them.

## Workflow schematic

<figure class="crl-figure">
  <img src="../../assets/diagrams/workflow.svg" alt="Workflow diagram from logged data to report" loading="lazy" />
  <figcaption>Typical CausalRL workflow: data -> estimand -> estimator -> diagnostics -> sensitivity -> report.</figcaption>
</figure>

## Sanity check

```bash
python - <<'PY'
import crl
print(crl.__version__)
PY
```
