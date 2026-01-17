# Bounded Rewards

**Assumption**
- Rewards are uniformly bounded.

**Applies to**
- IS and related estimators when deriving concentration bounds.

**Definition**
- There exists R_max such that |R_t| <= R_max almost surely.

**Diagnostics**
- Check empirical reward range and clip if necessary.

**Failure modes**
- Heavy-tailed rewards inflate variance and invalidate CI assumptions.

**References**
- Uehara, Shi, Kallus (2022).
