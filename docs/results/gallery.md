# Results Gallery

All figures on this page are generated from the repository codebase. Core
figures come from `scripts/build_docs_figures.py`, while end-to-end notebook
figures are emitted by the tutorial notebooks in `notebooks/`.

## Bandit estimator comparison

<figure class="crl-figure">
  <a href="../assets/figures/bandit_estimator_comparison.svg" data-gallery="results">
    <img src="../assets/figures/bandit_estimator_comparison_web.png" alt="Bandit estimator comparison with uncertainty" loading="lazy" />
  </a>
  <figcaption>Bandit OPE estimates with uncertainty for IS and WIS compared to ground truth.</figcaption>
</figure>

What it shows:

- Relative bias and uncertainty across estimators.
- Ground-truth reference line from the synthetic benchmark.
- Practical spread between IS and WIS under the same data.

Why it matters:

- Shows estimator stability before you trust an estimate.
- Highlights the value of diagnostics and CIs in small samples.

## MDP estimator comparison

<figure class="crl-figure">
  <a href="../assets/figures/mdp_estimator_comparison.svg" data-gallery="results">
    <img src="../assets/figures/mdp_estimator_comparison_web.png" alt="MDP estimator comparison with uncertainty" loading="lazy" />
  </a>
  <figcaption>MDP OPE estimates for IS, WIS, PDIS, and DR with uncertainty bands.</figcaption>
</figure>

What it shows:

- Trajectory-based estimators across multiple horizons.
- The effect of model-based correction (DR) on variance.
- A direct comparison against ground truth.

Why it matters:

- Demonstrates why horizon length makes diagnostics essential.
- Provides a baseline for choosing estimators on real data.

## Overlap diagnostics

<figure class="crl-figure">
  <a href="../assets/figures/bandit_overlap_ratios.svg" data-gallery="results">
    <img src="../assets/figures/bandit_overlap_ratios_web.png" alt="Overlap ratios histogram" loading="lazy" />
  </a>
  <figcaption>Distribution of target/behavior importance ratios for bandit data.</figcaption>
</figure>

What it shows:

- Whether target actions are supported by the behavior policy.
- Heavy tails that signal unstable importance weighting.

Why it matters:

- Poor overlap is the fastest path to unreliable OPE.

## Effective sample size over time

<figure class="crl-figure">
  <a href="../assets/figures/mdp_ess_by_time.svg" data-gallery="results">
    <img src="../assets/figures/mdp_ess_by_time_web.png" alt="Effective sample size by time step" loading="lazy" />
  </a>
  <figcaption>ESS drops over time for MDP trajectories under importance weighting.</figcaption>
</figure>

What it shows:

- How effective sample size decays with horizon.
- The variance cost of long sequences.

Why it matters:

- Long horizons require careful estimator choice and diagnostics.

## Weighted reward distribution

<figure class="crl-figure">
  <a href="../assets/figures/bandit_weighted_rewards.svg" data-gallery="results">
    <img src="../assets/figures/bandit_weighted_rewards_web.png" alt="Weighted rewards histogram" loading="lazy" />
  </a>
  <figcaption>Importance-weighted rewards plotted on a log scale.</figcaption>
</figure>

What it shows:

- How a few large weights can dominate estimates.
- The tail behavior that affects estimator stability.

Why it matters:

- Motivates clipping, diagnostics, and sensitivity checks.

## Sensitivity bounds

<figure class="crl-figure">
  <a href="../assets/figures/bandit_sensitivity_bounds.svg" data-gallery="results">
    <img src="../assets/figures/bandit_sensitivity_bounds_web.png" alt="Sensitivity bounds curve" loading="lazy" />
  </a>
  <figcaption>Bounded-confounding sensitivity curve for bandit OPE.</figcaption>
</figure>

What it shows:

- Lower and upper bounds as confounding strength increases.
- A compact summary of robustness to unobserved bias.

Why it matters:

- Sensitivity curves help quantify uncertainty beyond point estimates.

## End-to-end bandit workflow

<figure class="crl-figure">
  <a href="../assets/figures/bandit_end_to_end_estimator_comparison.pdf" data-gallery="results">
    <img src="../assets/figures/bandit_end_to_end_estimator_comparison.png" alt="End-to-end bandit estimator comparison" loading="lazy" />
  </a>
  <figcaption>Estimator comparison from the bandit end-to-end notebook.</figcaption>
</figure>

What it shows:

- A full pipeline run with diagnostics and sensitivity outputs.
- How estimates line up with the synthetic ground truth.

Why it matters:

- Mirrors the workflow researchers use to audit new policies.

## Long-horizon MDP comparison

<figure class="crl-figure">
  <a href="../assets/figures/mdp_long_horizon_estimator_comparison.pdf" data-gallery="results">
    <img src="../assets/figures/mdp_long_horizon_estimator_comparison.png" alt="Long-horizon MDP estimator comparison" loading="lazy" />
  </a>
  <figcaption>Estimator comparison in a longer-horizon MDP.</figcaption>
</figure>

What it shows:

- Increased variance for IS/PDIS as horizon grows.
- Stabilization from MIS/DICE and model-based estimators.

Why it matters:

- Motivates long-horizon diagnostics and estimator selection.
