**CausalRL** is a research-grade Python library for **off-policy evaluation (OPE)** that makes causal assumptions explicit. It goes beyond point estimates—combining estimand-first design, diagnostics-first reporting, and reproducible benchmarks so you can tell not just *what* a policy is worth, but *whether you should trust the estimate.*

</td>
</tr>
</table>

> 📦 **v0.2.0** (research preview, alpha) &nbsp;·&nbsp; Import: `import crl`

---

## ✨ Why CausalRL?

<table>
<tr>
<td align="center" width="25%">

### 🎯
### Estimand-First
Every estimator is tied to a formal estimand with explicit identification assumptions

</td>
<td align="center" width="25%">

### 🔍
### Diagnostics by Default
Overlap, ESS, weight tails, and shift checks run automatically with every evaluation

</td>
<td align="center" width="25%">

### 📊
### 20+ Estimators
IS, DR, WDR, MAGIC, MRDR, MIS, FQE, DualDICE, GenDICE, DRL, and more

</td>
<td align="center" width="25%">

### 📈
### Sensitivity Analysis
Bounded-confounding curves for robustness to hidden confounders

</td>
</tr>
</table>

<table>
<tr>
<td align="center" width="25%">

### 📦
### D4RL Compatible
Load D4RL and RL Unplugged datasets with built-in adapters

</td>
<td align="center" width="25%">

### 📝
### Audit-Ready Reports
HTML reports with tables, figures, and full metadata bundles

</td>
<td align="center" width="25%">

### 🧪
### Ground-Truth Benchmarks
Synthetic bandit/MDP suites with known true values for validation

</td>
<td align="center" width="25%">

### ⚡
### Production Ready
Type-checked, tested, with deterministic seeding throughout

</td>
</tr>
</table>

---

## 🚀 Quickstart

### Installation

```bash
# Install from PyPI
pip install causalrl

# With all extras
pip install "causalrl[all]"

# Clone and install from source
git clone https://github.com/gsaco/causalrl
cd causalrl
pip install -e .
```

### Your First OPE Evaluation

```python
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.ope import evaluate_ope

# Create a synthetic benchmark with known ground truth
benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
dataset = benchmark.sample(num_samples=1000, seed=1)

# Run end-to-end evaluation
report = evaluate_ope(dataset=dataset, policy=benchmark.target_policy)

# View results
print(report.summary_table())

# Generate audit-ready HTML report
report.save_html("report.html")
```

**Output:**
```
              Estimator    Value     Std      ESS  OverlapWarning
0                    IS   0.8234  0.0821   412.3           False
1                   WIS   0.8156  0.0634   412.3           False
2                    DR   0.8189  0.0512   412.3           False
3                   WDR   0.8167  0.0498   412.3           False
Ground Truth: 0.8200
```

### CLI

```bash
# Quick bandit OPE demo
python -m examples.quickstart.bandit_ope

# MDP evaluation
python -m examples.quickstart.mdp_ope

# Run full benchmark suite
python -m experiments.run_benchmarks --suite all --out results/
```

---

## 📊 Sample Outputs

<table>
<tr>
<td align="center" width="50%">

<img src="docs/assets/figures/bandit_estimator_comparison_web.png" alt="Estimator comparison with confidence intervals" width="100%"/>

**Estimator Comparison**<br/>
<sub>Point estimates with uncertainty quantification</sub>

</td>
<td align="center" width="50%">

<img src="docs/assets/figures/bandit_overlap_ratios_web.png" alt="Overlap diagnostics" width="100%"/>

**Overlap Diagnostics**<br/>
<sub>Importance weight ratio distribution</sub>

</td>
</tr>
<tr>
<td align="center" width="50%">

<img src="docs/assets/figures/bandit_sensitivity_bounds_web.png" alt="Sensitivity analysis bounds" width="100%"/>

**Sensitivity Analysis**<br/>
<sub>Bounds under hidden confounding</sub>

</td>
<td align="center" width="50%">

<img src="docs/assets/figures/mdp_ess_by_time_web.png" alt="ESS by time step" width="100%"/>

**Temporal ESS**<br/>
<sub>Effective sample size across horizon</sub>

</td>
</tr>
</table>

---

## 🧠 The Three Pillars

| Pillar | Why It Matters | What You Get |
|--------|----------------|--------------|
| **Estimands** | Know *what quantity* you're estimating—not just which estimator | Explicit estimands with identification assumptions via `AssumptionSet` |
| **Diagnostics** | Know *when* an estimate is fragile before acting on it | Overlap checks, ESS, weight tails, shift diagnostics, sensitivity curves |
| **Evidence** | Know *how* results were produced for auditing and reproducibility | Versioned configs, deterministic seeds, structured report bundles |

---

## 📦 Estimator Suite

<details>
<summary><strong>Click to expand full estimator list</strong></summary>

| Category | Estimators | Notes |
|----------|------------|-------|
| **Importance Sampling** | IS, WIS, SN-IS | Propensity-based weighting |
| **Doubly Robust** | DR, WDR | Combines regression with IS |
| **Model-Assisted** | MAGIC, MRDR | Variance reduction via modeling |
| **Marginalized** | MIS | State-marginal importance sampling |
| **Value Function** | FQE | Fitted Q-Evaluation |
| **DICE Family** | DualDICE, GenDICE | Distribution correction estimation |
| **Double RL** | DRL | Double reinforcement learning |
| **High-Confidence** | HCOPE bounds | Concentration-based bounds |

</details>

---

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Dataset   │ ──▶ │  Estimand   │ ──▶ │ Estimators  │ ──▶ │   Report    │
│             │     │ + Assump.   │     │ + Diagnostics│    │  (HTML/JSON)│
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                                        │
      ▼                                        ▼
┌─────────────┐                         ┌─────────────┐
│  Benchmarks │                         │ Sensitivity │
│ (Synth/D4RL)│                         │  Analysis   │
└─────────────┘                         └─────────────┘
```

---

## 📚 Learn the Library

**Recommended learning path:**

1. 📖 [Installation Guide](https://gsaco.github.io/causalrl/getting-started/installation.md)
2. 🚀 [Quickstart Tutorial](https://gsaco.github.io/causalrl/tutorials/examples.md)
3. 🔍 [Diagnostics Guide](https://gsaco.github.io/causalrl/how-to/)
4. 📈 [Sensitivity Analysis](https://gsaco.github.io/causalrl/tutorials/)
5. 🧪 [Benchmarking Workflow](https://gsaco.github.io/causalrl/reproducibility/)

**Reference:**

- [Estimator Reference](https://gsaco.github.io/causalrl/reference/estimators/)
- [Public API](https://gsaco.github.io/causalrl/reference/api/public_api.md)
- [Dataset Format](https://gsaco.github.io/causalrl/concepts/dataset_format.md)

---

## 🤝 Contributing

We welcome contributions! Check out:

- [Contributing Guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Open Issues](https://github.com/gsaco/causalrl/issues)

---

## 📄 Citation

If you use CausalRL in academic work, please cite:

```bibtex
@software{causalrl,
  author = {Saco, Gabriel},
  title = {CausalRL: Estimand-first Causal Reinforcement Learning},
  year = {2024},
  url = {https://github.com/gsaco/causalrl}
}
```

Or use the **"Cite this repository"** button on GitHub.

---

## 📜 License

[MIT](LICENSE) © Gabriel Saco

---

<p align="center">
  <sub>Built with ❤️ for the causal inference and reinforcement learning communities</sub>
</p>
