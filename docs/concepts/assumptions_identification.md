# Assumptions and Identification

CausalRL requires explicit assumptions. These determine which estimators are
valid and how to interpret results.

## Core assumptions

- **Sequential ignorability**: actions are conditionally independent of
  potential outcomes given observed history.
- **Overlap (positivity)**: the behavior policy assigns nonzero probability to
  actions that the target policy may take.
- **Markov (MDPs)**: the future depends only on the current state and action.
- **Behavior policy known** (`BEHAVIOR_POLICY_KNOWN`): propensities are known or correctly specified.
- **Q-model realizability**: value function lies in the chosen model class.
- **Bridge identifiability**: proximal bridge functions are well-posed.
- **Bounded rewards**: required for concentration bounds.

## What estimators require

- IS/WIS/PDIS: sequential ignorability + overlap.
- DR/WDR/MRDR/MAGIC: sequential ignorability + overlap + Markov.
- DRL: sequential ignorability + overlap + Markov.
- FQE: sequential ignorability + overlap + Markov + Q-model realizability.
- MIS: sequential ignorability + overlap + Markov.
- DualDICE/GenDICE: sequential ignorability + Markov + discrete state support.
- HCOPE: sequential ignorability + overlap + bounded rewards.

## Why this matters

Assumptions are the bridge from data to causal statements. Without them, the
estimate is a number without a guarantee.
