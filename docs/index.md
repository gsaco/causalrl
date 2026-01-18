# CausalRL

CausalRL is an estimand-first toolkit for causal reinforcement learning and
off-policy evaluation, pairing identification assumptions with diagnostics and
reproducible benchmarks.

## Why CausalRL

- Diagnostics-first reporting to surface overlap and ESS issues early.
- Synthetic benchmarks with ground truth for method selection.
- A growing estimator suite grounded in core OPE literature.

## When to use this library

- Offline evaluation when online deployment is risky or expensive.
- Diagnostics-first iteration before promoting policies.
- Synthetic ground-truth experiments for estimator comparisons.

## Get started

- Follow the [Quickstart (Bandit)](tutorials/quickstart_bandit.md) tutorial.
- Run the [Quickstart (MDP)](tutorials/quickstart_mdp.md).
- Browse the [Estimator Reference](reference/estimators/index.md).
- View a [sample HTML report](assets/reports/intro_bandit_report.html).

## Estimator selection

See the [Estimator Selection Guide](explanation/estimator_selection.md) for a
decision tree and recommended defaults.
