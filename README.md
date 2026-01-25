# CausalRL

<p align="center">
  <img src="docs/assets/branding/causalrl-banner.svg" alt="CausalRL banner" width="100%" />
</p>

<p align="center">
  <strong>Estimand-first causal reinforcement learning and off-policy evaluation</strong><br />
  Assumptions in the open. Diagnostics by default. Reports you can audit.
</p>

<p align="center">
  <a href="https://github.com/gsaco/causalrl/actions/workflows/ci.yml"><img src="https://github.com/gsaco/causalrl/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://gsaco.github.io/causalrl/"><img src="https://github.com/gsaco/causalrl/actions/workflows/pages.yml/badge.svg?branch=main" alt="Docs" /></a>
  <a href="https://codecov.io/gh/gsaco/causalrl"><img src="https://codecov.io/gh/gsaco/causalrl/branch/main/graph/badge.svg" alt="Coverage" /></a>
  <a href="https://pypi.org/project/causalrl/"><img src="https://img.shields.io/badge/PyPI-coming%20soon-lightgrey" alt="PyPI" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/gsaco/causalrl" alt="License" /></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python" /></a>
</p>

<p align="center">
  <a href="https://gsaco.github.io/causalrl/">Docs</a> |
  <a href="https://gsaco.github.io/causalrl/tutorials/">Tutorials</a> |
  <a href="examples/README.md">Examples</a> |
  <a href="docs/project_status.md">Project status</a> |
  <a href="CITATION.cff">Cite</a>
</p>

> Release status: v0.1.0 (research preview, alpha quality).
> Import name: `crl`

---

## What is CausalRL

CausalRL is a research-grade library for off-policy evaluation (OPE) that makes
causal assumptions explicit. It combines estimand-first design, diagnostics-first
reporting, and reproducible benchmarks so you can tell not just what a policy is
worth, but whether you should trust the estimate.

### The three pillars

| Pillar | Why it matters | What you get |
| --- | --- | --- |
| Estimands | Know what quantity you are estimating. | Explicit estimands and assumptions for identification. |
| Diagnostics | Know when an estimate is fragile. | Overlap, ESS, weight tails, and shift checks. |
| Evidence | Know how results were produced. | Reproducible configs, benchmarks, and report artifacts. |

---

## 60-second quickstart

Install from source (PyPI not yet published):

```bash
git clone https://github.com/gsaco/causalrl
cd causalrl
python -m pip install -e .
```

Run a synthetic bandit OPE in a few lines:

```python
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.ope import evaluate_ope

benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = benchmark.sample(num_samples=1000, seed=1)

report = evaluate_ope(dataset=dataset, policy=benchmark.target_policy)
print(report.summary_table())
report.save_html("report.html")
```

Runnable scripts:

```bash
python -m examples.quickstart.bandit_ope
python -m examples.quickstart.mdp_ope
```

---

## End-to-end pipeline (reports you can share)

```python
from crl.ope import evaluate_ope

report = evaluate_ope(dataset=dataset, policy=policy)
report.save_html("report.html")
report.save_bundle("results/run_001")
```

Bundle structure:

```text
results/run_001/
  report.html
  report.json
  summary.csv
  metadata.json
  figures/
```

---

## Diagnostics that matter

- Overlap and support violations
- Effective sample size (ESS)
- Weight tails and instability flags
- Shift diagnostics (when propensities are available)
- Sensitivity bounds for unobserved confounding

---

## Benchmarks and reproducibility

- Synthetic bandit and MDP suites with ground-truth values
- Deterministic seeds and versioned configs
- `python -m experiments.run_benchmarks --suite all --out results/`

---

## Learn the library

- Docs: https://gsaco.github.io/causalrl/
- Tutorials: https://gsaco.github.io/causalrl/tutorials/
- Estimator reference: https://gsaco.github.io/causalrl/reference/estimators/

Recommended path:

1. Quickstart (bandit and MDP)
2. Diagnostics guide
3. Sensitivity analysis
4. Benchmarking workflow

---

## Citation

If you use CausalRL in academic work, cite via `CITATION.cff` or the GitHub
"Cite this repository" button.

---

## License

MIT. See `LICENSE`.
