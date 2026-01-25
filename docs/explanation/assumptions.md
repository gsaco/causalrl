# Assumptions and Identifiability

Causal OPE relies on explicit assumptions. Each estimator lists its required
assumptions in the API and documentation.

## Core assumptions

- Sequential ignorability: no unmeasured confounding given observed history.
- Overlap: behavior policy supports target policy actions.
- Markov: state is sufficient for future evolution.
- Behavior policy known: propensities are known or correctly specified.
- Q-model realizability: value function lies in the chosen model class.
- Bridge identifiability: proximal bridge functions are well-posed.
- Bounded rewards: needed for high-confidence bounds.
- Bounded confounding: hidden bias is limited by a sensitivity parameter.

## Practical guidance

- Diagnose overlap before trusting IS-based estimators.
- Prefer model-based or DR estimators when overlap is weak.
- Use sensitivity analysis when ignorability is questionable.
