# Project Status

## Release status

- Current release: v0.2.0 (research preview, alpha quality).
- API may evolve as estimator coverage and diagnostics expand.

## Versioning and deprecations

- We follow SemVer; breaking changes are expected in 0.x, with tighter stability after 1.0.
- Deprecated public APIs warn for at least one minor release before removal.

## Current scope

- Unconfounded OPE with explicit estimands, diagnostics, and synthetic benchmarks
- Bandit and sequential propensity sensitivity analysis under bounded confounding
- Baseline and advanced estimators for IS, WIS, PDIS, DR, WDR, MAGIC, MRDR, FQE
- Estimator selection heuristics and behavior policy estimation

## Experimental and non-goals

- Confounded and transport settings are exposed as experimental interfaces only
- Representation learning claims are treated as heuristics unless identified

## Roadmap

- Transport and mechanism shift tooling
- Expanded confounded MDP benchmarks and sensitivity analyses
- Additional diagnostics for model misspecification
