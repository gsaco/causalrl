"""Minimal bandit OPE example."""

from __future__ import annotations

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.importance_sampling import ISEstimator, WISEstimator


def main() -> None:
    benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
    dataset = benchmark.sample(num_samples=1000, seed=1)
    true_value = benchmark.true_policy_value(benchmark.target_policy)

    estimand = PolicyValueEstimand(
        policy=benchmark.target_policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
    )

    for estimator in [ISEstimator(estimand), WISEstimator(estimand)]:
        report = estimator.estimate(dataset)
        print(
            f"{report.metadata['estimator']}: estimate={report.value:.3f}, "
            f"true={true_value:.3f}, warnings={report.warnings}"
        )


if __name__ == "__main__":
    main()
