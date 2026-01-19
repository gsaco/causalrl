# Assumptions and Identification

CausalRL requires explicit assumptions. These determine which estimators are
valid and how to interpret results.

## Core assumptions

- **Sequential ignorability**: actions are conditionally independent of
  potential outcomes given observed history.
- **Overlap (positivity)**: the behavior policy assigns nonzero probability to
  actions that the target policy may take.
- **Markov (MDPs)**: the future depends only on the current state and action.

## What estimators require

- IS/WIS/PDIS: sequential ignorability + overlap.
- DR/WDR/MRDR/MAGIC: sequential ignorability + overlap + model correctness.
- FQE: sequential ignorability + overlap + adequate function approximation.
- DualDICE/MIS: sequential ignorability + Markov + discrete state support.

## Why this matters

Assumptions are the bridge from data to causal statements. Without them, the
estimate is a number without a guarantee.
