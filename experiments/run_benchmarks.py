"""CLI entrypoint for running synthetic benchmarks."""

from __future__ import annotations

import argparse

from crl.experiments.runner import run_benchmarks_to_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CRL synthetic benchmarks.")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-trajectories", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_benchmarks_to_table(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_trajectories=args.num_trajectories,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
