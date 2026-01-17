# Assumptions and Identifiability

Causal OPE relies on explicit assumptions. Each estimator lists its required
assumptions in the API and documentation.

## Core assumptions

- Sequential ignorability: no unmeasured confounding given observed history.
- Overlap: behavior policy supports target policy actions.
- Markov: state is sufficient for future evolution.
- Correct model: function approximation matches the data-generating process.
- Bounded rewards: needed for high-confidence bounds.

## Practical guidance

- Diagnose overlap before trusting IS-based estimators.
- Prefer model-based or DR estimators when overlap is weak.
- Use sensitivity analysis when ignorability is questionable.
