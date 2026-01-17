"""CLI entrypoint for running synthetic benchmarks."""

from __future__ import annotations

import argparse

from crl.experiments.runner import run_benchmark_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CRL synthetic benchmarks.")
    parser.add_argument("--suite", type=str, default="all")
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    run_benchmark_suite(args.suite, args.out, seeds)


if __name__ == "__main__":
    main()
