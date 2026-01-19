"""Micro-benchmark for estimator runtime."""

from __future__ import annotations

import argparse
import time

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.fqe import FQEConfig, FQEEstimator


def main() -> None:
    parser = argparse.ArgumentParser(description="CRL micro-benchmark")
    parser.add_argument("--num-trajectories", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--num-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    benchmark = SyntheticMDP(
        SyntheticMDPConfig(seed=args.seed, horizon=args.horizon)
    )
    dataset = benchmark.sample(num_trajectories=args.num_trajectories, seed=args.seed)

    estimand = PolicyValueEstimand(
        policy=benchmark.target_policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV]),
    )

    config = FQEConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_iterations=args.num_iterations,
        seed=args.seed,
    )
    estimator = FQEEstimator(estimand, config=config, device=args.device)

    start = time.perf_counter()
    report = estimator.estimate(dataset)
    elapsed = time.perf_counter() - start

    print("elapsed_sec=%.3f" % elapsed)
    print("estimate=%.6f" % report.value)


if __name__ == "__main__":
    main()
