# Markov Property

**Assumption**
- The state is a sufficient summary of the history for transitions and rewards.

**Applies to**
- DR and FQE.

**Definition**
- P(S_{t+1}, R_t | H_t, A_t) = P(S_{t+1}, R_t | S_t, A_t).

**Diagnostics**
- Model misspecification checks and residual diagnostics.

**Failure modes**
- State aliasing induces bias in model-based estimators.

**References**
- Jiang and Li (2016).
