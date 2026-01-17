"""Minimal MDP OPE example."""

from __future__ import annotations

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.dr import DoublyRobustEstimator
from crl.estimators.fqe import FQEEstimator
from crl.estimators.importance_sampling import ISEstimator, PDISEstimator, WISEstimator


def main() -> None:
    benchmark = SyntheticMDP(SyntheticMDPConfig(seed=0))
    dataset = benchmark.sample(num_trajectories=200, seed=1)
    true_value = benchmark.true_policy_value(benchmark.target_policy)

    estimand = PolicyValueEstimand(
        policy=benchmark.target_policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV]),
    )

    estimators = [
        ISEstimator(estimand),
        WISEstimator(estimand),
        PDISEstimator(estimand),
        DoublyRobustEstimator(estimand),
        FQEEstimator(estimand),
    ]

    for estimator in estimators:
        report = estimator.estimate(dataset)
        print(
            f"{report.metadata['estimator']}: estimate={report.value:.3f}, "
            f"true={true_value:.3f}, warnings={report.warnings}"
        )


if __name__ == "__main__":
    main()
