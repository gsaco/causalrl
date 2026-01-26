# CausalRL

<div class="crl-hero" markdown="1">

<img src="assets/branding/causalrl-banner.svg" alt="CausalRL Banner" style="width: 100%; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 10px 40px rgba(0,0,0,0.3);" />

## Estimand-first causal RL and off-policy evaluation

<p class="crl-hero-subtitle">
Know what you're estimating. Know when to trust it. Know how it was produced.
</p>

<div class="crl-hero-actions" markdown="1">

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[View Gallery](results/gallery.md){ .md-button }
[Tutorials](tutorials/index.md){ .md-button }

</div>

</div>

---

**Package**: `causalrl` · **Import**: `crl` · **Version**: 0.2.0 · [:fontawesome-brands-github: GitHub](https://github.com/gsaco/causalrl)

---

## Why CausalRL?

<div class="grid cards" markdown="1">

-   :material-target: **Estimand-first Design**

    ---

    Every estimator is tied to a formal estimand with explicit identification assumptions. Know *what* you're estimating.

-   :material-magnify: **Diagnostics by Default**

    ---

    Overlap, ESS, weight tails, and shift checks run automatically. Know *when* to trust your estimates.

-   :material-chart-line: **20+ Estimators**

    ---

    IS, WIS, DR, WDR, MAGIC, MRDR, MIS, FQE, DualDICE, GenDICE, DRL—all in a unified pipeline.

-   :material-shield-check: **Sensitivity Analysis**

    ---

    Bounded-confounding curves for bandits and sequential settings. Quantify robustness to hidden confounders.

-   :material-package-variant: **D4RL & RL Unplugged**

    ---

    Built-in adapters for standard RL benchmarks. Load datasets with one line of code.

-   :material-file-document-check: **Audit-Ready Reports**

    ---

    HTML reports with tables, figures, and full metadata bundles. Share reproducible results.

-   :material-test-tube: **Ground-Truth Benchmarks**

    ---

    Synthetic bandit/MDP suites with known true values. Validate estimators before deployment.

-   :material-lightning-bolt: **Production Ready**

    ---

    Type-checked, tested, with deterministic seeding throughout. Built for research reliability.

</div>

---

## 60-Second Quickstart

=== "Installation"

    ```bash
    # Install from PyPI
    pip install causalrl

    # With all optional extras
    pip install "causalrl[all]"

    # Or install from source
    git clone https://github.com/gsaco/causalrl
    cd causalrl
    pip install -e .
    ```

=== "First Evaluation"

    ```python
    from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
    from crl.ope import evaluate_ope

    # Create benchmark with known ground truth
    benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
    dataset = benchmark.sample(num_samples=1000, seed=1)

    # Run end-to-end evaluation
    report = evaluate_ope(dataset=dataset, policy=benchmark.target_policy)
    
    # View results and generate report
    print(report.summary_table())
    report.save_html("report.html")
    ```

=== "CLI"

    ```bash
    # Bandit OPE demo
    python -m examples.quickstart.bandit_ope
    
    # MDP evaluation
    python -m examples.quickstart.mdp_ope
    
    # Full benchmark suite
    python -m experiments.run_benchmarks --suite all --out results/
    ```

!!! note "Scope"
    The current `evaluate` pipeline assumes discrete action spaces for importance sampling estimators. See [Limitations](concepts/limitations.md) for details on continuous actions.

---

## The Three Pillars

| Pillar | Why It Matters | What You Get |
|--------|----------------|--------------|
| **Estimands** | Know *what quantity* you're estimating—not just which estimator | Explicit estimands with identification assumptions via `AssumptionSet` |
| **Diagnostics** | Know *when* an estimate is fragile before acting on it | Overlap checks, ESS, weight tails, shift diagnostics, sensitivity curves |
| **Evidence** | Know *how* results were produced for auditing | Versioned configs, deterministic seeds, structured report bundles |

---

## Results Gallery

<div class="crl-gallery">
  <a href="assets/figures/bandit_estimator_comparison.svg" data-gallery="home">
    <img src="assets/figures/bandit_estimator_comparison_web.png" alt="Bandit estimator comparison" loading="lazy" />
  </a>
  <a href="assets/figures/bandit_overlap_ratios.svg" data-gallery="home">
    <img src="assets/figures/bandit_overlap_ratios_web.png" alt="Overlap ratios histogram" loading="lazy" />
  </a>
  <a href="assets/figures/bandit_sensitivity_bounds.svg" data-gallery="home">
    <img src="assets/figures/bandit_sensitivity_bounds_web.png" alt="Sensitivity bounds curve" loading="lazy" />
  </a>
  <a href="assets/figures/mdp_estimator_comparison.svg" data-gallery="home">
    <img src="assets/figures/mdp_estimator_comparison_web.png" alt="MDP estimator comparison" loading="lazy" />
  </a>
  <a href="assets/figures/mdp_ess_by_time.svg" data-gallery="home">
    <img src="assets/figures/mdp_ess_by_time_web.png" alt="Effective sample size" loading="lazy" />
  </a>
  <a href="assets/figures/bandit_weighted_rewards.svg" data-gallery="home">
    <img src="assets/figures/bandit_weighted_rewards_web.png" alt="Weighted reward distribution" loading="lazy" />
  </a>
</div>

[:material-arrow-right: See the full Results Gallery](results/gallery.md)

---

## Why Trust CausalRL?

<div class="grid cards" markdown="1">

-   :material-check-circle: **Explicit Assumptions**

    Every estimand declares its identification assumptions via `AssumptionSet`—no hidden requirements.

-   :material-check-circle: **Deterministic Benchmarks**

    Synthetic generators with fixed seeds produce identical results across runs.

-   :material-check-circle: **Comprehensive Testing**

    Test suite covering estimators, diagnostics, and full pipeline integration.

-   :material-check-circle: **Docs ↔ Code Parity**

    Automated checks keep formulas and APIs aligned with documentation.

</div>

---

## Data Contracts

Use the dataset contracts in `crl.data` and follow the shape rules exactly:

| Data Type | Class | Use Case |
|-----------|-------|----------|
| **Bandits** | `LoggedBanditDataset` | Single-step contextual decisions |
| **Trajectories** | `TrajectoryDataset` | Episode-based sequential data |
| **Transitions** | `TransitionDataset` | Step-by-step (s, a, r, s') tuples |

[:material-arrow-right: Dataset Format and Validation](concepts/dataset_format.md)

---

## Learn by Example

<div class="grid cards" markdown="1">

-   :material-rocket-launch: **Getting Started**

    [Installation](getting-started/installation.md) · [Quickstart](tutorials/examples.md)

-   :material-notebook: **Tutorials**

    [Notebook Walkthroughs](tutorials/index.md) · [Diagnostics](tutorials/diagnostics.md)

-   :material-book-open: **Reference**

    [Estimator Reference](reference/estimators/index.md) · [Public API](reference/api/public_api.md)

-   :material-chart-box: **Results**

    [Gallery](results/gallery.md) · [Sample HTML Report](results/report_preview.md)

</div>

---

## Estimator Selection

Not sure which estimator to use? See the [Estimator Selection Guide](explanation/estimator_selection.md) for a practical decision tree and recommended defaults.
