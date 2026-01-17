"""CLI entrypoints for CRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml

from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.ope import evaluate

app = typer.Typer(help="CausalRL command line interface.")


@app.command()
def ope(config: str = typer.Option(..., help="Path to OPE config YAML."),
        out: str = typer.Option(..., help="Output directory for reports.")) -> None:
    """Run OPE based on a YAML config file."""

    cfg = _load_yaml(config)
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, policy, truth = _resolve_benchmark(cfg.get("benchmark", {}))
    report = evaluate(
        dataset=dataset,
        policy=policy,
        estimand=None,
        estimators=cfg.get("estimators", "default"),
        diagnostics=cfg.get("diagnostics", "default"),
        inference=cfg.get("inference"),
        seed=cfg.get("seed", 0),
    )

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = report.summary_table()
    summary.to_csv(output_dir / "summary.csv", index=False)

    from crl.viz import save_figure

    fig = report.plot_estimator_comparison(truth=truth)
    save_figure(fig, figures_dir / "estimator_comparison")

    report.to_html(str(output_dir / "report.html"))


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_benchmark(config: dict[str, Any]):
    benchmark_type = config.get("type", "bandit")
    if benchmark_type == "bandit":
        bench = SyntheticBandit(SyntheticBanditConfig(**config.get("config", {})))
        dataset = bench.sample(
            num_samples=int(config.get("num_samples", 1000)),
            seed=int(config.get("seed", 0)),
        )
        truth = bench.true_policy_value(bench.target_policy)
        return dataset, bench.target_policy, truth

    if benchmark_type == "mdp":
        bench = SyntheticMDP(SyntheticMDPConfig(**config.get("config", {})))
        dataset = bench.sample(
            num_trajectories=int(config.get("num_trajectories", 200)),
            seed=int(config.get("seed", 0)),
        )
        truth = bench.true_policy_value(bench.target_policy)
        return dataset, bench.target_policy, truth

    raise ValueError(f"Unknown benchmark type: {benchmark_type}")


if __name__ == "__main__":
    app()
