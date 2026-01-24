"""Config loaders for CRL."""

from __future__ import annotations

from typing import Any

import numpy as np
import yaml

from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.evaluation.spec import (
    DiagnosticsSpec,
    EvaluationSpec,
    InferenceSpec,
    ReportSpec,
    SensitivitySpec,
)


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_evaluation_spec(path: str) -> EvaluationSpec:
    cfg = load_yaml(path)
    benchmark_cfg = cfg.get("benchmark")
    if benchmark_cfg is None:
        raise ValueError("Config must include a 'benchmark' section.")

    dataset, policy = _resolve_benchmark(benchmark_cfg)

    estimators = cfg.get("estimators", "auto")

    diagnostics = _parse_diagnostics(cfg.get("diagnostics"))
    inference = _parse_inference(cfg.get("inference"))
    sensitivity = _parse_sensitivity(cfg.get("sensitivity"))
    report = _parse_report(cfg.get("report"))

    seed = int(cfg.get("seed", 0))

    return EvaluationSpec(
        dataset=dataset,
        policy=policy,
        estimators=estimators,
        diagnostics=diagnostics,
        inference=inference,
        sensitivity=sensitivity,
        report=report,
        seed=seed,
    )


def _parse_diagnostics(value: Any) -> DiagnosticsSpec:
    if value is None or value == "default":
        return DiagnosticsSpec()
    if isinstance(value, bool):
        return DiagnosticsSpec(enabled=value)
    if isinstance(value, list):
        return DiagnosticsSpec(suites=value)
    if isinstance(value, dict):
        return DiagnosticsSpec(
            enabled=bool(value.get("enabled", True)),
            suites=tuple(value.get("suites", ("overlap", "ess", "weights", "shift"))),
            fail_on=tuple(value.get("fail_on", ())),
            min_ess=value.get("min_ess"),
            max_weight=value.get("max_weight"),
        )
    raise ValueError(f"Unsupported diagnostics config: {value}")


def _parse_inference(value: Any) -> InferenceSpec:
    if value is None:
        return InferenceSpec()
    if isinstance(value, dict):
        return InferenceSpec(
            alpha=float(value.get("alpha", 0.05)),
            method=value.get("method", "asymptotic"),
            bootstrap_num=int(value.get("bootstrap_num", 200)),
            bootstrap_kind=value.get("bootstrap_kind", "trajectory"),
            bootstrap_block_size=int(value.get("bootstrap_block_size", 5)),
            seed=int(value.get("seed", 0)),
        )
    raise ValueError(f"Unsupported inference config: {value}")


def _parse_sensitivity(value: Any) -> SensitivitySpec:
    if value is None:
        return SensitivitySpec()
    if isinstance(value, bool):
        return SensitivitySpec(enabled=value)
    if isinstance(value, dict):
        gammas = value.get("gammas")
        gamma_arr = None
        if gammas is not None:
            gamma_arr = np.asarray(gammas, dtype=float)
        return SensitivitySpec(
            enabled=bool(value.get("enabled", True)),
            kind=value.get("kind", "bandit_gamma"),
            gammas=gamma_arr,
            baseline_value=value.get("baseline_value"),
        )
    raise ValueError(f"Unsupported sensitivity config: {value}")


def _parse_report(value: Any) -> ReportSpec:
    if value is None:
        return ReportSpec()
    if isinstance(value, dict):
        return ReportSpec(
            html=bool(value.get("html", True)),
            include_figures=bool(value.get("include_figures", True)),
            theme=value.get("theme", "auto"),
        )
    raise ValueError(f"Unsupported report config: {value}")


def _resolve_benchmark(
    config: dict[str, Any],
):
    benchmark_type = config.get("type", "bandit")
    if benchmark_type == "bandit":
        bandit_bench = SyntheticBandit(
            SyntheticBanditConfig(**config.get("config", {}))
        )
        bandit_dataset = bandit_bench.sample(
            num_samples=int(config.get("num_samples", 1000)),
            seed=int(config.get("seed", 0)),
        )
        return bandit_dataset, bandit_bench.target_policy
    if benchmark_type == "mdp":
        mdp_bench = SyntheticMDP(SyntheticMDPConfig(**config.get("config", {})))
        mdp_dataset = mdp_bench.sample(
            num_trajectories=int(config.get("num_trajectories", 200)),
            seed=int(config.get("seed", 0)),
        )
        return mdp_dataset, mdp_bench.target_policy

    raise ValueError(f"Unknown benchmark type: {benchmark_type}")


__all__ = ["load_yaml", "load_evaluation_spec"]
