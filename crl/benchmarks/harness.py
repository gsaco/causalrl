"""Benchmark harness for synthetic bandit and MDP benchmarks."""

from __future__ import annotations

from typing import Any

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import (
    MARKOV,
    OVERLAP,
    Q_MODEL_REALIZABLE,
    SEQUENTIAL_IGNORABILITY,
    BEHAVIOR_POLICY_KNOWN,
)
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.dr import DoublyRobustEstimator
from crl.estimators.fqe import FQEEstimator
from crl.estimators.importance_sampling import ISEstimator, PDISEstimator, WISEstimator


def run_bandit_benchmark(
    num_samples: int = 1000,
    seed: int = 0,
    config: SyntheticBanditConfig | None = None,
) -> list[dict[str, Any]]:
    """Run IS/WIS on the synthetic bandit benchmark.

    Estimand:
        Policy value under intervention for the benchmark target policy.
    Assumptions:
        Sequential ignorability, overlap, and known behavior propensities.
    Inputs:
        num_samples: Number of logged bandit samples.
        seed: Random seed for sampling.
        config: Optional SyntheticBanditConfig override.
    Outputs:
        List of result dictionaries with estimate and true value.
    Failure modes:
        Small samples can yield high variance estimates.
    """

    bench = SyntheticBandit(config or SyntheticBanditConfig(seed=seed))
    dataset = bench.sample(num_samples=num_samples, seed=seed)
    true_value = bench.true_policy_value(bench.target_policy)

    estimand = PolicyValueEstimand(
        policy=bench.target_policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, BEHAVIOR_POLICY_KNOWN]),
    )

    estimators = [ISEstimator(estimand), WISEstimator(estimand)]
    results: list[dict[str, Any]] = []
    for estimator in estimators:
        report = estimator.estimate(dataset)
        results.append(
            {
                "benchmark": "synthetic_bandit",
                "estimator": report.metadata["estimator"],
                "estimate": report.value,
                "stderr": report.stderr,
                "true_value": true_value,
                "error": report.value - true_value,
            }
        )
    return results


def run_mdp_benchmark(
    num_trajectories: int = 200,
    seed: int = 0,
    config: SyntheticMDPConfig | None = None,
) -> list[dict[str, Any]]:
    """Run IS/WIS/PDIS/DR/FQE on the synthetic MDP benchmark.

    Estimand:
        Policy value under intervention for the benchmark target policy.
    Assumptions:
        Sequential ignorability, overlap, Markov property, and known behavior propensities.
    Inputs:
        num_trajectories: Number of logged trajectories.
        seed: Random seed for sampling.
        config: Optional SyntheticMDPConfig override.
    Outputs:
        List of result dictionaries with estimate and true value.
    Failure modes:
        Small samples can yield unstable estimates.
    """

    bench = SyntheticMDP(config or SyntheticMDPConfig(seed=seed))
    dataset = bench.sample(num_trajectories=num_trajectories, seed=seed)
    true_value = bench.true_policy_value(bench.target_policy)

    estimand = PolicyValueEstimand(
        policy=bench.target_policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet(
            [
                SEQUENTIAL_IGNORABILITY,
                OVERLAP,
                BEHAVIOR_POLICY_KNOWN,
                MARKOV,
                Q_MODEL_REALIZABLE,
            ]
        ),
    )

    estimators = [
        ISEstimator(estimand),
        WISEstimator(estimand),
        PDISEstimator(estimand),
        DoublyRobustEstimator(estimand),
        FQEEstimator(estimand),
    ]

    results: list[dict[str, Any]] = []
    for estimator in estimators:
        report = estimator.estimate(dataset)
        results.append(
            {
                "benchmark": "synthetic_mdp",
                "estimator": report.metadata["estimator"],
                "estimate": report.value,
                "stderr": report.stderr,
                "true_value": true_value,
                "error": report.value - true_value,
            }
        )
    return results


def run_all_benchmarks(
    num_samples: int = 1000,
    num_trajectories: int = 200,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run all benchmarks and return a combined result table.

    Estimand:
        Policy value under intervention for each benchmark target policy.
    Assumptions:
        Sequential ignorability, overlap, and known behavior propensities (plus Markov for MDP).
    Inputs:
        num_samples: Number of bandit samples.
        num_trajectories: Number of MDP trajectories.
        seed: Random seed for sampling.
    Outputs:
        Combined list of result dictionaries.
    Failure modes:
        Small samples can yield unstable estimates.
    """

    results = []
    results.extend(run_bandit_benchmark(num_samples=num_samples, seed=seed))
    results.extend(run_mdp_benchmark(num_trajectories=num_trajectories, seed=seed))
    return results
