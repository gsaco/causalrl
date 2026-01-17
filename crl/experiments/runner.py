"""Experiment runner for synthetic benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from crl.benchmarks.harness import run_all_benchmarks


def run_benchmarks_to_table(
    output_dir: str | Path,
    num_samples: int = 1000,
    num_trajectories: int = 200,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run benchmarks and write CSV/JSONL outputs.

    Estimand:
        Policy value under intervention for each benchmark target policy.
    Assumptions:
        Sequential ignorability and overlap (plus Markov for MDP).
    Inputs:
        output_dir: Directory for result files.
        num_samples: Number of bandit samples.
        num_trajectories: Number of MDP trajectories.
        seed: RNG seed.
    Outputs:
        List of result dictionaries.
    Failure modes:
        Small samples can yield unstable estimates.
    """

    results = run_all_benchmarks(
        num_samples=num_samples, num_trajectories=num_trajectories, seed=seed
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=sorted(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    with (output_dir / "benchmark_results.jsonl").open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    return results
