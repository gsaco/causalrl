"""CLI entrypoint for running synthetic benchmarks."""

from __future__ import annotations

import argparse

from crl.experiments.runner import run_benchmark_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CRL synthetic benchmarks.")
    parser.add_argument("--suite", type=str, default="all")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config-dir", type=str, default="configs/benchmarks")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--out", type=str, dest="output_dir", default=None)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        seeds = [int(args.seed)]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    output_dir = args.output_dir or "results"
    run_benchmark_suite(
        args.suite,
        output_dir,
        seeds,
        config_dir=args.config_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
