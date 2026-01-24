"""End-to-end off-policy evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, overload

import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import (
    BEHAVIOR_POLICY_KNOWN,
    MARKOV,
    OVERLAP,
    SEQUENTIAL_IGNORABILITY,
    BOUNDED_CONFOUNDING,
)
from crl.core.datasets import BanditDataset, TrajectoryDataset, TransitionDataset
from crl.core.policy import Policy
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimands.sensitivity_policy_value import SensitivityPolicyValueEstimand
from crl.estimators.base import EstimatorReport, OPEEstimator
from crl.estimators.diagnostics import run_diagnostics
from crl.estimators.double_rl import DoubleRLEstimator
from crl.estimators.dr import DoublyRobustEstimator
from crl.estimators.drl import DRLEstimator
from crl.estimators.dual_dice import DualDICEEstimator
from crl.estimators.fqe import FQEEstimator
from crl.estimators.gen_dice import GenDICEEstimator
from crl.estimators.high_confidence import HighConfidenceISEstimator
from crl.estimators.importance_sampling import ISEstimator, PDISEstimator, WISEstimator
from crl.estimators.magic import MAGICEstimator
from crl.estimators.mis import MarginalizedImportanceSamplingEstimator
from crl.estimators.mrdr import MRDREstimator
from crl.estimators.utils import compute_action_probs
from crl.estimators.wdr import WeightedDoublyRobustEstimator
from crl.evaluation.spec import EvaluationSpec
from crl.reporting import EstimateRow, ReportData, ReportMetadata
from crl.reporting.export import save_bundle as save_report_bundle
from crl.reporting.html import render_html
from crl.reporting.warnings import normalize_warnings
from crl.utils.seeding import set_seed
from crl.utils.validation import validate_dataset
from crl.version import __version__


@dataclass
class OpeReport:
    """Aggregate report for an OPE evaluation run."""

    estimand: PolicyValueEstimand
    reports: dict[str, EstimatorReport]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_table(self) -> Any:
        """Return a pandas DataFrame summary if pandas is available."""

        rows: list[dict[str, Any]] = []
        for name, report in self.reports.items():
            row = report.to_dict()
            row["estimator"] = name
            rows.append(row)

        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pandas is required for summary_table().") from exc
        return pd.DataFrame(rows)

    def to_dataframe(self) -> Any:
        """Alias for summary_table()."""

        return self.summary_table()

    def plot_estimator_comparison(self, truth: float | None = None) -> Any:
        """Plot estimator comparison with confidence intervals."""

        from crl.viz.plots import plot_estimator_comparison

        df = self.summary_table()
        fig = plot_estimator_comparison(df, truth=truth)
        self.figures["estimator_comparison"] = fig
        return fig

    def plot_importance_weights(self, weights: np.ndarray, logy: bool = True) -> Any:
        """Plot importance weight distribution."""

        from crl.viz.plots import plot_importance_weights

        fig = plot_importance_weights(weights, logy=logy)
        self.figures["importance_weights"] = fig
        return fig

    def plot_effective_sample_size(
        self, weights: np.ndarray, by_time: bool = False
    ) -> Any:
        """Plot effective sample size diagnostics."""

        from crl.viz.plots import plot_effective_sample_size

        fig = plot_effective_sample_size(weights, by_time=by_time)
        self.figures["effective_sample_size"] = fig
        return fig

    def save_html(self, out_path: str) -> None:
        """Write a self-contained HTML report."""

        html = self.to_html()
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

    def to_html(self, out_path: str | None = None) -> str:
        """Return a self-contained HTML report string (optionally write to file)."""

        report_data = self.to_report_data()
        html = render_html(report_data, title="Off-Policy Evaluation Report")
        if out_path is not None:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)
        return html

    def to_report_data(self) -> ReportData:
        """Build a structured report payload."""

        metadata = ReportMetadata(
            run_name=self.metadata.get("run_name"),
            package_version=self.metadata.get("package_version") or __version__,
            git_sha=self.metadata.get("git_sha"),
            dataset_fingerprint=self.metadata.get("dataset_fingerprint"),
            dataset_summary=self.metadata.get("dataset_summary"),
            policy_name=self.metadata.get("policy_name"),
            baseline_policy_name=self.metadata.get("baseline_policy_name"),
            estimand=str(self.estimand),
            seed=self.metadata.get("seed"),
            configs={
                "inference": self.metadata.get("inference", {}),
                "diagnostics": self.metadata.get("diagnostics", {}),
                "sensitivity": self.metadata.get("sensitivity", {}),
                "estimators": list(self.reports.keys()),
            },
            environment=self.metadata.get("environment", {}),
        )

        estimates: list[EstimateRow] = []
        for name, report in self.reports.items():
            payload = report.to_dict()
            estimates.append(
                EstimateRow(
                    estimator=name,
                    value=payload.get("value"),
                    stderr=payload.get("stderr"),
                    ci=payload.get("ci"),
                    lower_bound=payload.get("lower_bound"),
                    upper_bound=payload.get("upper_bound"),
                    assumptions_checked=payload.get("assumptions_checked", []),
                    assumptions_flagged=payload.get("assumptions_flagged", []),
                    warnings=normalize_warnings(payload.get("warnings", [])),
                    diagnostics=payload.get("diagnostics", {}),
                    metadata=payload.get("metadata", {}),
                )
            )

        figures_payload: list[dict[str, Any]] = []
        for name, fig in self.figures.items():
            img = _figure_to_base64(fig)
            figures_payload.append(
                {
                    "id": name,
                    "title": name.replace("_", " ").title(),
                    "src": f"data:image/png;base64,{img}",
                }
            )

        return ReportData(
            mode="ope",
            metadata=metadata,
            estimates=estimates,
            diagnostics=self.diagnostics,
            sensitivity=self.diagnostics.get("sensitivity"),
            figures=figures_payload,
            warnings=normalize_warnings(self.metadata.get("warnings", [])),
        )

    def save_bundle(self, output_dir: str) -> None:
        """Write a bundle containing HTML, JSON, CSV, and figures."""

        report_data = self.to_report_data()
        html = render_html(report_data, title="Off-Policy Evaluation Report")
        save_report_bundle(
            output_dir,
            html=html,
            report_data=report_data.to_dict(),
            summary=self.summary_table(),
            figures=self.figures,
            metadata=self.metadata,
        )

    def __repr__(self) -> str:
        return f"OpeReport(num_estimators={len(self.reports)}, diagnostics_keys={list(self.diagnostics.keys())})"


@overload
def evaluate(spec: EvaluationSpec) -> Any:
    ...


@overload
def evaluate(
    dataset: BanditDataset | TrajectoryDataset | TransitionDataset,
    policy: Policy,
    estimand: PolicyValueEstimand | None = None,
    estimators: Iterable[str | OPEEstimator] | str = "default",
    diagnostics: list[str] | str = "default",
    inference: dict[str, Any] | None = None,
    sensitivity: SensitivityPolicyValueEstimand | None = None,
    seed: int = 0,
) -> OpeReport:
    ...


def evaluate(
    dataset: BanditDataset | TrajectoryDataset | TransitionDataset | EvaluationSpec,
    policy: Policy | None = None,
    estimand: PolicyValueEstimand | None = None,
    estimators: Iterable[str | OPEEstimator] | str = "default",
    diagnostics: list[str] | str = "default",
    inference: dict[str, Any] | None = None,
    sensitivity: SensitivityPolicyValueEstimand | None = None,
    seed: int = 0,
) -> Any:
    """Run an end-to-end OPE evaluation with reporting."""

    if isinstance(dataset, EvaluationSpec):
        return _evaluate_from_spec(dataset)
    if policy is None:
        raise ValueError("policy is required when dataset is provided directly.")
    return _evaluate_legacy(
        dataset=dataset,
        policy=policy,
        estimand=estimand,
        estimators=estimators,
        diagnostics=diagnostics,
        inference=inference,
        sensitivity=sensitivity,
        seed=seed,
    )


def _evaluate_legacy(
    dataset: BanditDataset | TrajectoryDataset | TransitionDataset,
    policy: Policy,
    *,
    estimand: PolicyValueEstimand | None = None,
    estimators: Iterable[str | OPEEstimator] | str = "default",
    diagnostics: list[str] | str = "default",
    inference: dict[str, Any] | None = None,
    sensitivity: SensitivityPolicyValueEstimand | None = None,
    diagnostics_config: Any | None = None,
    seed: int = 0,
) -> OpeReport:
    """Legacy evaluate implementation (dataset + policy arguments)."""

    set_seed(seed)
    if isinstance(dataset, TransitionDataset):
        if dataset.action_space_n is None:
            raise ValueError(
                "TransitionDataset with continuous actions is not supported in evaluate(); "
                "provide action_space_n for discrete actions or construct a TrajectoryDataset."
            )
        if dataset.episode_ids is None or dataset.timesteps is None:
            raise ValueError(
                "TransitionDataset requires episode_ids and timesteps to build trajectories "
                "for OPE; provide them or construct a TrajectoryDataset."
            )
        dataset = dataset.to_trajectory()
    validate_dataset(dataset)

    if estimand is None:
        assumptions = [SEQUENTIAL_IGNORABILITY, OVERLAP]
        if getattr(dataset, "behavior_action_probs", None) is not None:
            assumptions.append(BEHAVIOR_POLICY_KNOWN)
        if isinstance(dataset, TrajectoryDataset):
            assumptions.append(MARKOV)
        estimand = PolicyValueEstimand(
            policy=policy,
            discount=dataset.discount,
            horizon=dataset.horizon,
            assumptions=AssumptionSet(assumptions),
        )

    estimator_list, estimator_warnings = _resolve_estimators(
        estimators, estimand, dataset, strict=estimators != "default"
    )
    warnings: list[str] = []
    warnings.extend(estimator_warnings)
    if getattr(dataset, "behavior_action_probs", None) is None:
        skipped = []
        filtered: list[OPEEstimator] = []
        for estimator in estimator_list:
            if "behavior_action_probs" in getattr(estimator, "required_fields", []):
                skipped.append(type(estimator).__name__)
                continue
            filtered.append(estimator)
        if skipped:
            warnings.append(
                "Skipped estimators requiring behavior_action_probs: "
                + ", ".join(skipped)
                + "."
            )
        estimator_list = filtered
    if not estimator_list:
        raise ValueError("No compatible estimators available for this dataset.")
    reports: dict[str, EstimatorReport] = {}
    for estimator in estimator_list:
        report = estimator.estimate(dataset)
        name = report.metadata.get("estimator", type(estimator).__name__)
        reports[name] = report

    diagnostics_out: dict[str, Any] = {}
    figures_out: dict[str, Any] = {}
    if (
        diagnostics != []
        and getattr(dataset, "behavior_action_probs", None) is not None
    ):
        suite_names = diagnostics if isinstance(diagnostics, list) else None
        diagnostics_out = _compute_dataset_diagnostics(
            dataset, policy, suites=suite_names, config=diagnostics_config
        )
    if sensitivity is not None:
        sensitivity.require(["bounded_confounding"])
        bounds = sensitivity.compute_bounds(dataset)
        diagnostics_out["sensitivity"] = bounds.to_dict()
        try:
            from crl.viz.plots import plot_sensitivity_curve

            records = [
                {"gamma": float(g), "lower": float(lower), "upper": float(upper)}
                for g, lower, upper in zip(
                    bounds.gammas, bounds.lower, bounds.upper, strict=True
                )
            ]
            figures_out["sensitivity_bounds"] = plot_sensitivity_curve(records)
        except Exception:
            diagnostics_out.setdefault("sensitivity", {}).setdefault(
                "plot_error", "failed to create sensitivity plot"
            )

    dataset_summary = (
        dataset.summary() if hasattr(dataset, "summary") else getattr(dataset, "describe", lambda: {})()
    )
    if hasattr(dataset, "fingerprint"):
        dataset_fingerprint = dataset.fingerprint()
    else:
        try:
            from crl.data.fingerprint import fingerprint_dataset

            dataset_fingerprint = fingerprint_dataset(dataset)
        except Exception:
            dataset_fingerprint = None

    metadata = {
        "seed": seed,
        "inference": inference or {},
        "diagnostics": diagnostics,
        "warnings": warnings,
        "dataset_summary": dataset_summary,
        "dataset_fingerprint": dataset_fingerprint,
        "policy_name": getattr(policy, "name", type(policy).__name__),
        "package_version": __version__,
        "environment": _environment_metadata(),
    }

    return OpeReport(
        estimand=estimand,
        reports=reports,
        diagnostics=diagnostics_out,
        figures=figures_out,
        metadata=metadata,
    )


def _resolve_estimators(
    estimators: Iterable[str | OPEEstimator] | str,
    estimand: PolicyValueEstimand,
    dataset: BanditDataset | TrajectoryDataset,
    *,
    strict: bool = True,
) -> tuple[list[OPEEstimator], list[str]]:
    if estimators == "default":
        if isinstance(dataset, BanditDataset):
            default_estimators: list[type[OPEEstimator]] = [
                ISEstimator,
                WISEstimator,
                DoubleRLEstimator,
            ]
        else:
            default_estimators = [
                ISEstimator,
                WISEstimator,
                PDISEstimator,
                DoublyRobustEstimator,
                WeightedDoublyRobustEstimator,
                MAGICEstimator,
                MRDREstimator,
                MarginalizedImportanceSamplingEstimator,
                FQEEstimator,
                DualDICEEstimator,
                DRLEstimator,
                GenDICEEstimator,
            ]
        warnings: list[str] = []
        resolved: list[OPEEstimator] = []
        for estimator_cls in default_estimators:
            try:
                resolved.append(estimator_cls(estimand))
            except ValueError as exc:
                if strict:
                    raise
                warnings.append(
                    f"Skipped {estimator_cls.__name__}: {exc}"
                )
        return resolved, warnings
    if isinstance(estimators, str):
        estimators = [estimators]
    estimator_list: list[OPEEstimator] = []
    warnings: list[str] = []
    for estimator in estimators:
        if isinstance(estimator, OPEEstimator):
            estimator_list.append(estimator)
            continue
        try:
            estimator_list.append(_estimator_from_name(str(estimator), estimand))
        except ValueError as exc:
            if strict:
                raise
            warnings.append(f"Skipped {estimator}: {exc}")
    return estimator_list, warnings


def _estimator_from_name(name: str, estimand: PolicyValueEstimand) -> OPEEstimator:
    registry: dict[str, type[OPEEstimator]] = {
        "is": ISEstimator,
        "wis": WISEstimator,
        "pdis": PDISEstimator,
        "dr": DoublyRobustEstimator,
        "wdr": WeightedDoublyRobustEstimator,
        "magic": MAGICEstimator,
        "mrdr": MRDREstimator,
        "mis": MarginalizedImportanceSamplingEstimator,
        "fqe": FQEEstimator,
        "dualdice": DualDICEEstimator,
        "dual_dice": DualDICEEstimator,
        "gendice": GenDICEEstimator,
        "gen_dice": GenDICEEstimator,
        "double_rl": DoubleRLEstimator,
        "drl": DRLEstimator,
        "hcope": HighConfidenceISEstimator,
    }
    key = name.strip().lower()
    if key not in registry:
        raise ValueError(f"Unknown estimator name: {name}")
    return registry[key](estimand)


def _compute_dataset_diagnostics(
    dataset: BanditDataset | TrajectoryDataset,
    policy: Policy,
    *,
    suites: list[str] | None = None,
    config: Any | None = None,
) -> dict[str, Any]:
    from crl.diagnostics.calibration import behavior_calibration_from_metadata
    from crl.diagnostics.registry import run_suite
    from crl.diagnostics.slicing import action_overlap_slices, timestep_weight_slices

    if isinstance(dataset, BanditDataset):
        assert dataset.behavior_action_probs is not None
        target_probs = policy.action_prob(dataset.contexts, dataset.actions)
        ratios = target_probs / dataset.behavior_action_probs
        suites = suites or ["overlap", "ess", "weights", "shift"]
        diag_config = config or _default_diagnostics_config()
        diagnostics, _ = run_suite(
            suites,
            weights=ratios,
            target_action_probs=target_probs,
            behavior_action_probs=dataset.behavior_action_probs,
            mask=None,
            config=diag_config,
            contexts=dataset.contexts,
        )
        diagnostics["slices"] = {
            "actions": action_overlap_slices(
                dataset.actions,
                dataset.behavior_action_probs,
                target_probs,
                action_space_n=dataset.action_space_n,
            )
        }
        diagnostics["calibration"] = behavior_calibration_from_metadata(
            dataset.metadata
        )
        return diagnostics

    assert dataset.behavior_action_probs is not None
    target_probs = compute_action_probs(policy, dataset.observations, dataset.actions)
    ratios = np.where(dataset.mask, target_probs / dataset.behavior_action_probs, 1.0)
    weights = np.prod(ratios, axis=1)
    suites = suites or ["overlap", "ess", "weights"]
    diag_config = config or _default_diagnostics_config()
    diagnostics, _ = run_suite(
        suites,
        weights=weights,
        target_action_probs=target_probs,
        behavior_action_probs=dataset.behavior_action_probs,
        mask=dataset.mask,
        config=diag_config,
        contexts=None,
    )
    try:
        from crl.diagnostics.shift import state_shift_diagnostics

        flat_states = dataset.observations[dataset.mask]
        flat_weights = ratios[dataset.mask]
        diagnostics["shift"] = state_shift_diagnostics(
            flat_states, weights=flat_weights
        )
    except Exception:
        diagnostics["shift"] = {"error": "shift diagnostics failed"}
    diagnostics["slices"] = {
        "timesteps": timestep_weight_slices(ratios, dataset.mask)
    }
    diagnostics["calibration"] = behavior_calibration_from_metadata(
        dataset.metadata
    )
    return diagnostics


def _default_diagnostics_config():
    from crl.estimators.base import DiagnosticsConfig

    return DiagnosticsConfig()


def _evaluate_from_spec(spec: EvaluationSpec) -> Any:
    from crl.evaluation.result import EvaluationResult

    if spec.estimand != "policy_value":
        raise NotImplementedError(
            "EvaluationSpec.estimand='policy_contrast' is not yet supported."
        )

    dataset = spec.dataset
    policy = spec.policy

    estimators = spec.estimators
    if estimators == "auto":
        estimators_arg: Iterable[str | OPEEstimator] | str = "default"
    else:
        name_map = {"dual_dice": "dualdice", "gen_dice": "gendice"}
        estimators_arg = [
            name_map.get(str(name), str(name)) for name in list(estimators)
        ]

    if spec.diagnostics.enabled:
        diagnostics_arg: list[str] | str = list(spec.diagnostics.suites)
    else:
        diagnostics_arg = []

    sensitivity_estimand: SensitivityPolicyValueEstimand | None = None
    if spec.sensitivity.enabled:
        gammas = (
            spec.sensitivity.gammas
            if spec.sensitivity.gammas is not None
            else np.linspace(1.0, 3.0, 9)
        )
        sensitivity_estimand = SensitivityPolicyValueEstimand(
            policy=policy,
            discount=dataset.discount,
            horizon=dataset.horizon,
            gammas=np.asarray(gammas, dtype=float),
            assumptions=AssumptionSet([BOUNDED_CONFOUNDING]),
        )

    estimand_obj = None
    if spec.assumptions is not None:
        estimand_obj = PolicyValueEstimand(
            policy=policy,
            discount=dataset.discount,
            horizon=dataset.horizon,
            assumptions=spec.assumptions,
        )

    diag_config = None
    if spec.diagnostics.min_ess is not None or spec.diagnostics.max_weight is not None:
        from crl.estimators.base import DiagnosticsConfig

        diag_config = DiagnosticsConfig()
        if spec.diagnostics.max_weight is not None:
            diag_config.max_weight = spec.diagnostics.max_weight
        if spec.diagnostics.min_ess is not None:
            threshold = spec.diagnostics.min_ess
            if threshold > 1.0:
                size = getattr(dataset, "num_samples", None)
                if size is None:
                    size = getattr(dataset, "num_trajectories", None)
                if size:
                    diag_config.ess_threshold = float(threshold) / float(size)
            else:
                diag_config.ess_threshold = float(threshold)

    report = _evaluate_legacy(
        dataset=dataset,
        policy=policy,
        estimand=estimand_obj,
        estimators=estimators_arg,
        diagnostics=diagnostics_arg,
        inference=spec.inference.__dict__,
        sensitivity=sensitivity_estimand,
        diagnostics_config=diag_config,
        seed=spec.seed,
    )

    gate_warnings = _apply_quality_gates(report, spec)
    if gate_warnings:
        report.metadata.setdefault("warnings", [])
        report.metadata["warnings"] = list(report.metadata.get("warnings", [])) + gate_warnings
        report.metadata["quality_gates"] = {
            "failed": gate_warnings,
            "rules": list(spec.diagnostics.fail_on),
        }

    report.metadata.setdefault("spec", {})
    report.metadata["spec"] = {
        "estimators": estimators,
        "diagnostics": spec.diagnostics.__dict__,
        "inference": spec.inference.__dict__,
        "sensitivity": {
            "enabled": spec.sensitivity.enabled,
            "kind": spec.sensitivity.kind,
            "baseline_value": spec.sensitivity.baseline_value,
            "gammas": None
            if spec.sensitivity.gammas is None
            else spec.sensitivity.gammas.tolist(),
        },
    }

    return EvaluationResult(spec=spec, report=report)


def _apply_quality_gates(report: OpeReport, spec: EvaluationSpec) -> list[str]:
    warnings: list[str] = []
    diagnostics = report.diagnostics or {}
    gates = set(spec.diagnostics.fail_on)

    if "overlap_violation" in gates:
        overlap = diagnostics.get("overlap", {})
        if overlap.get("support_violations", 0) > 0:
            warnings.append("Quality gate failed: overlap_violation.")

    if "ess_too_low" in gates:
        ess_ratio = diagnostics.get("ess", {}).get("ess_ratio")
        if ess_ratio is not None:
            threshold = spec.diagnostics.min_ess
            if threshold is None:
                from crl.estimators.base import DiagnosticsConfig

                threshold = DiagnosticsConfig().ess_threshold
            if threshold is not None and ess_ratio < float(threshold):
                warnings.append("Quality gate failed: ess_too_low.")

    if "extreme_tail" in gates:
        tail = diagnostics.get("weights", {})
        tail_fraction = tail.get("tail_fraction")
        if tail_fraction is not None and tail_fraction > 0.01:
            warnings.append("Quality gate failed: extreme_tail.")

    return warnings


def _figure_to_base64(fig: Any) -> str:
    from crl.viz.style import figure_to_base64

    return figure_to_base64(fig, dpi=500)


def _environment_metadata() -> dict[str, Any]:
    import platform
    import sys

    versions: dict[str, str | None] = {}
    try:
        from importlib import metadata

        for name in ["numpy", "pandas", "torch", "causalrl"]:
            try:
                versions[name] = metadata.version(name)
            except Exception:
                versions[name] = None
    except Exception:
        versions = {}

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": versions,
    }

def evaluate_many(**kwargs: Any) -> Any:
    from crl.evaluation.suite import evaluate_many as _evaluate_many

    return _evaluate_many(**kwargs)


__all__ = ["OpeReport", "evaluate", "evaluate_many"]
