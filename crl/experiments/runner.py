"""Experiment runner for synthetic benchmarks."""

from __future__ import annotations

import base64
import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.benchmarks.harness import run_all_benchmarks
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.data.datasets import LoggedBanditDataset, TrajectoryDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.ope import evaluate
from crl.version import __version__
from crl.viz.plots import plot_bias_variance_tradeoff, plot_estimator_comparison


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


def run_benchmark_suite(
    suite: str,
    output_dir: str | Path,
    seeds: list[int],
    config_dir: str | Path = "configs/benchmarks",
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run a benchmark suite defined by YAML configs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    specs = _load_suite_specs(suite, config_dir, config_path)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        for spec in specs:
            results.extend(_run_spec(spec, seed))

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "results.csv", index=False)

    aggregate = _aggregate_results(df)
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)

    _write_figures(aggregate, figures_dir)
    _write_html_report(aggregate, figures_dir, output_dir / "report.html")
    _write_metadata(
        output_dir,
        suite=suite,
        seeds=seeds,
        config_dir=config_dir,
        config_path=config_path,
        num_specs=len(specs),
    )

    return df


def _load_suite_specs(
    suite: str, config_dir: str | Path, config_path: str | Path | None
) -> list[dict[str, Any]]:
    if config_path is not None:
        path = Path(config_path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return [
            dict(spec, suite=data.get("suite", suite))
            for spec in data.get("benchmarks", [])
        ]
    config_dir = Path(config_dir)
    if suite == "all":
        all_path = config_dir / "all.yaml"
        data = yaml.safe_load(all_path.read_text(encoding="utf-8"))
        specs: list[dict[str, Any]] = []
        for name in data.get("suites", []):
            specs.extend(_load_suite_specs(name, config_dir))
        return specs
    suite_path = config_dir / f"{suite}.yaml"
    data = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    return [
        dict(spec, suite=data.get("suite", suite))
        for spec in data.get("benchmarks", [])
    ]


def _run_spec(spec: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    benchmark_type = spec["type"]
    behavior_known = spec.get("behavior_known", True)
    estimators = spec.get("estimators", "default")
    name = spec.get("name", f"{benchmark_type}_bench")
    dataset: LoggedBanditDataset | TrajectoryDataset

    if benchmark_type == "bandit":
        bandit_config = SyntheticBanditConfig(seed=seed, **spec.get("config", {}))
        bandit_bench = SyntheticBandit(bandit_config)
        dataset = bandit_bench.sample(
            num_samples=int(spec.get("num_samples", 1000)), seed=seed
        )
        if not behavior_known:
            dataset.behavior_action_probs = None
        policy = bandit_bench.target_policy
        true_value = bandit_bench.true_policy_value(policy)
        estimand = PolicyValueEstimand(
            policy=policy,
            discount=1.0,
            horizon=1,
            assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
        )
    else:
        mdp_config = SyntheticMDPConfig(seed=seed, **spec.get("config", {}))
        mdp_bench = SyntheticMDP(mdp_config)
        dataset = mdp_bench.sample(
            num_trajectories=int(spec.get("num_trajectories", 200)), seed=seed
        )
        if not behavior_known:
            dataset.behavior_action_probs = None
        policy = mdp_bench.target_policy
        true_value = mdp_bench.true_policy_value(policy)
        estimand = PolicyValueEstimand(
            policy=policy,
            discount=dataset.discount,
            horizon=dataset.horizon,
            assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV]),
        )

    report = evaluate(
        dataset=dataset,
        policy=policy,
        estimand=estimand,
        estimators=estimators,
        diagnostics="default",
        seed=seed,
    )

    rows: list[dict[str, Any]] = []
    for name_key, est_report in report.reports.items():
        rows.append(
            {
                "suite": spec.get("suite", "custom"),
                "benchmark": name,
                "type": benchmark_type,
                "estimator": name_key,
                "estimate": est_report.value,
                "stderr": est_report.stderr,
                "true_value": true_value,
                "error": est_report.value - true_value,
                "seed": seed,
                "behavior_known": behavior_known,
            }
        )
    return rows


def _aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["suite", "benchmark", "estimator"], as_index=False)
    agg = grouped.agg(
        estimate_mean=("estimate", "mean"),
        estimate_std=("estimate", "std"),
        error_mean=("error", "mean"),
        error_std=("error", "std"),
        true_value=("true_value", "mean"),
    )
    agg["bias"] = agg["error_mean"]
    agg["variance"] = agg["estimate_std"] ** 2
    agg["mse"] = agg["bias"] ** 2 + agg["variance"]
    return agg


def _write_figures(aggregate: pd.DataFrame, figures_dir: Path) -> None:
    from crl.viz import save_figure

    for benchmark in aggregate["benchmark"].unique():
        subset = aggregate[aggregate["benchmark"] == benchmark]
        rows = []
        for _, row in subset.iterrows():
            mean = float(row["estimate_mean"])
            std = (
                float(row["estimate_std"]) if not pd.isna(row["estimate_std"]) else 0.0
            )
            rows.append(
                {
                    "estimator": row["estimator"],
                    "value": mean,
                    "ci": (mean - std, mean + std),
                }
            )
        fig = plot_estimator_comparison(rows, truth=float(subset["true_value"].iloc[0]))
        save_figure(fig, figures_dir / f"{benchmark}")

    trade_rows = aggregate[["estimator", "bias", "variance"]].to_dict(orient="records")
    fig = plot_bias_variance_tradeoff(trade_rows)
    save_figure(fig, figures_dir / "bias_variance_tradeoff")


def _write_html_report(
    aggregate: pd.DataFrame, figures_dir: Path, output_path: Path
) -> None:
    html_parts = [
        "<html><head><meta charset='utf-8'><title>CRL Benchmarks</title></head><body>",
        "<h1>Benchmark Summary</h1>",
        aggregate.to_html(index=False),
        "<h2>Figures</h2>",
    ]
    for fig_path in sorted(figures_dir.glob("*.png")):
        img = _to_base64(fig_path)
        html_parts.append(f"<h3>{fig_path.stem}</h3>")
        html_parts.append(f"<img src='data:image/png;base64,{img}' />")
    html_parts.append("</body></html>")
    output_path.write_text("\n".join(html_parts), encoding="utf-8")


def _write_metadata(
    output_dir: Path,
    *,
    suite: str,
    seeds: list[int],
    config_dir: str | Path,
    config_path: str | Path | None,
    num_specs: int,
) -> None:
    metadata = {
        "suite": suite,
        "seeds": seeds,
        "num_specs": num_specs,
        "config_dir": str(config_dir),
        "config_path": None if config_path is None else str(config_path),
        "crl_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "package_versions": _package_versions(
            ["numpy", "pandas", "torch", "pyyaml", "causalrl"]
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _package_versions(names: list[str]) -> dict[str, str | None]:
    try:
        from importlib import metadata
    except Exception:
        return {name: None for name in names}
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = None
    return versions


def _to_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")
