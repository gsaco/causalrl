# CausalRL

CausalRL is an estimand-first toolkit for causal reinforcement learning and
off-policy evaluation. It pairs identification assumptions with diagnostics and
reproducible benchmarks so you can trust or debug a policy value estimate.

> Package name on PyPI: `causalrl` | Import name: `crl`

## A 10-minute researcher path

1. Install:

   ```bash
   python -m pip install causalrl
   ```

2. Run a quickstart:

   ```bash
   python examples/quickstart/bandit_ope.py
   python examples/quickstart/mdp_ope.py
   ```

3. Interpret outputs:

   - Check the estimate and the diagnostics (overlap, ESS, weight tails).
   - Compare estimators in a synthetic benchmark before you trust a number.

## The mental model (teach it once, reuse forever)

1. **Define the estimand**: what policy value you want and the horizon/discount.
2. **State the assumptions**: sequential ignorability, overlap, and Markov (MDPs).
3. **Run estimators**: IS/WIS/PDIS/DR/FQE, depending on data and horizon.
4. **Read diagnostics**: overlap, ESS, and weight pathologies guide trust.

If you remember only one rule: never trust a point estimate without its
assumptions and diagnostics.

## What you can do here

- Diagnostics-first reporting to surface overlap and ESS issues early.
- Synthetic benchmarks with ground truth for method selection.
- A growing estimator suite grounded in core OPE literature.

## Data contracts (required before you run anything)

Use the dataset contracts in `crl.data` and follow the shape rules exactly:

- Bandits: `LoggedBanditDataset`
- Trajectories: `TrajectoryDataset`
- Transitions: `TransitionDataset`

Start here: [Dataset Format and Validation](concepts/dataset_format.md)

## Learn by example

- [Installation](getting-started/installation.md)
- [Quickstart (Bandit)](getting-started/quickstart_bandit.md)
- [Quickstart (MDP)](getting-started/quickstart_mdp.md)
- [Estimator Reference](reference/estimators/index.md)
- [Public API](reference/api/public_api.md)
- [Sample HTML report](assets/reports/intro_bandit_report.html)

## Estimator selection

See the [Estimator Selection Guide](explanation/estimator_selection.md) for a
practical decision tree and recommended defaults.
