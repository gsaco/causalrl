"""End-to-end off-policy evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.core.datasets import BanditDataset, TrajectoryDataset, TransitionDataset
from crl.core.policy import Policy
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.base import EstimatorReport, OPEEstimator
from crl.estimators.diagnostics import run_diagnostics
from crl.estimators.double_rl import DoubleRLEstimator
from crl.estimators.dr import DoublyRobustEstimator
from crl.estimators.dual_dice import DualDICEEstimator
from crl.estimators.fqe import FQEEstimator
from crl.estimators.high_confidence import HighConfidenceISEstimator
from crl.estimators.importance_sampling import ISEstimator, PDISEstimator, WISEstimator
from crl.estimators.magic import MAGICEstimator
from crl.estimators.mis import MarginalizedImportanceSamplingEstimator
from crl.estimators.mrdr import MRDREstimator
from crl.estimators.utils import compute_action_probs
from crl.estimators.wdr import WeightedDoublyRobustEstimator
from crl.utils.seeding import set_seed
from crl.utils.validation import validate_dataset


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
        """Alias for to_html()."""

        self.to_html(out_path)

    def to_html(self, out_path: str) -> None:
        """Write a self-contained HTML report with embedded figures."""

        try:
            import pandas as pd  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pandas is required for HTML export.") from exc

        html_parts = [
            "<html><head><meta charset='utf-8'><title>CRL OPE Report</title></head><body>",
            "<h1>Off-Policy Evaluation Report</h1>",
            "<h2>Summary</h2>",
            self.summary_table().to_html(index=False, escape=False),
        ]

        if any(report.assumptions_checked for report in self.reports.values()):
            html_parts.append("<h2>Assumptions</h2>")
            for name, report in self.reports.items():
                html_parts.append(f"<h3>{name}</h3>")
                html_parts.append("<ul>")
                html_parts.append(
                    f"<li>Checked: {', '.join(report.assumptions_checked) or 'None'}</li>"
                )
                html_parts.append(
                    f"<li>Flagged: {', '.join(report.assumptions_flagged) or 'None'}</li>"
                )
                html_parts.append("</ul>")

        if any(report.warnings for report in self.reports.values()):
            html_parts.append("<h2>Warnings</h2>")
            for name, report in self.reports.items():
                if not report.warnings:
                    continue
                html_parts.append(f"<h3>{name}</h3>")
                html_parts.append("<ul>")
                for warning in report.warnings:
                    html_parts.append(f"<li>{warning}</li>")
                html_parts.append("</ul>")

        if self.diagnostics:
            html_parts.append("<h2>Diagnostics</h2>")
            html_parts.append("<pre>")
            html_parts.append(str(self.diagnostics))
            html_parts.append("</pre>")

        if self.figures:
            html_parts.append("<h2>Figures</h2>")
            for name, fig in self.figures.items():
                img = _figure_to_base64(fig)
                html_parts.append(f"<h3>{name}</h3>")
                html_parts.append(f"<img src='data:image/png;base64,{img}' />")

        html_parts.append("</body></html>")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

    def __repr__(self) -> str:
        return f"OpeReport(num_estimators={len(self.reports)}, diagnostics_keys={list(self.diagnostics.keys())})"


def evaluate(
    dataset: BanditDataset | TrajectoryDataset | TransitionDataset,
    policy: Policy,
    estimand: PolicyValueEstimand | None = None,
    estimators: Iterable[str | OPEEstimator] | str = "default",
    diagnostics: list[str] | str = "default",
    inference: dict[str, Any] | None = None,
    seed: int = 0,
) -> OpeReport:
    """Run an end-to-end OPE evaluation with reporting.

    Estimand:
        Policy value of the provided target policy.
    Assumptions:
        Sequential ignorability and overlap by default (plus Markov for MDPs).
    Inputs:
        dataset: Logged bandit, trajectory, or transition dataset.
        policy: Target policy to evaluate.
        estimand: Optional PolicyValueEstimand override.
        estimators: List of estimator instances or names (or "default").
        diagnostics: Diagnostics configuration (currently unused placeholder).
        inference: Optional inference configuration (stored in metadata).
        seed: Random seed for estimators that require it.
    Outputs:
        OpeReport containing estimator reports and diagnostics.
    Failure modes:
        Raises ValueError if requested estimators are unavailable.
    """

    set_seed(seed)
    if isinstance(dataset, TransitionDataset):
        dataset = dataset.to_trajectory()
    validate_dataset(dataset)

    if estimand is None:
        assumptions = [SEQUENTIAL_IGNORABILITY, OVERLAP]
        if isinstance(dataset, TrajectoryDataset):
            assumptions.append(MARKOV)
        estimand = PolicyValueEstimand(
            policy=policy,
            discount=dataset.discount,
            horizon=dataset.horizon,
            assumptions=AssumptionSet(assumptions),
        )

    estimator_list = _resolve_estimators(estimators, estimand, dataset)
    warnings: list[str] = []
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
    if (
        diagnostics != []
        and getattr(dataset, "behavior_action_probs", None) is not None
    ):
        diagnostics_out = _compute_dataset_diagnostics(dataset, policy)

    return OpeReport(
        estimand=estimand,
        reports=reports,
        diagnostics=diagnostics_out,
        figures={},
        metadata={
            "seed": seed,
            "inference": inference or {},
            "diagnostics": diagnostics,
            "warnings": warnings,
        },
    )


def _resolve_estimators(
    estimators: Iterable[str | OPEEstimator] | str,
    estimand: PolicyValueEstimand,
    dataset: BanditDataset | TrajectoryDataset,
) -> list[OPEEstimator]:
    if estimators == "default":
        if isinstance(dataset, BanditDataset):
            return [
                ISEstimator(estimand),
                WISEstimator(estimand),
                DoubleRLEstimator(estimand),
            ]
        return [
            ISEstimator(estimand),
            WISEstimator(estimand),
            PDISEstimator(estimand),
            DoublyRobustEstimator(estimand),
            WeightedDoublyRobustEstimator(estimand),
            MAGICEstimator(estimand),
            MRDREstimator(estimand),
            MarginalizedImportanceSamplingEstimator(estimand),
            FQEEstimator(estimand),
            DualDICEEstimator(estimand),
        ]
    if isinstance(estimators, str):
        estimators = [estimators]
    estimator_list: list[OPEEstimator] = []
    for estimator in estimators:
        if isinstance(estimator, OPEEstimator):
            estimator_list.append(estimator)
            continue
        estimator_list.append(_estimator_from_name(str(estimator), estimand))
    return estimator_list


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
        "double_rl": DoubleRLEstimator,
        "hcope": HighConfidenceISEstimator,
    }
    key = name.strip().lower()
    if key not in registry:
        raise ValueError(f"Unknown estimator name: {name}")
    return registry[key](estimand)


def _compute_dataset_diagnostics(
    dataset: BanditDataset | TrajectoryDataset,
    policy: Policy,
) -> dict[str, Any]:
    if isinstance(dataset, BanditDataset):
        assert dataset.behavior_action_probs is not None
        target_probs = policy.action_prob(dataset.contexts, dataset.actions)
        ratios = target_probs / dataset.behavior_action_probs
        diagnostics, _ = run_diagnostics(
            ratios,
            target_probs,
            dataset.behavior_action_probs,
            None,
            config=_default_diagnostics_config(),
        )
        try:
            from crl.diagnostics.shift import state_shift_diagnostics

            diagnostics["shift"] = state_shift_diagnostics(
                dataset.contexts, weights=ratios
            )
        except Exception:
            diagnostics["shift"] = {"error": "shift diagnostics failed"}
        return diagnostics

    assert dataset.behavior_action_probs is not None
    target_probs = compute_action_probs(policy, dataset.observations, dataset.actions)
    ratios = np.where(dataset.mask, target_probs / dataset.behavior_action_probs, 1.0)
    weights = np.prod(ratios, axis=1)
    diagnostics, _ = run_diagnostics(
        weights,
        target_probs,
        dataset.behavior_action_probs,
        dataset.mask,
        config=_default_diagnostics_config(),
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
    return diagnostics


def _default_diagnostics_config():
    from crl.estimators.base import DiagnosticsConfig

    return DiagnosticsConfig()


def _figure_to_base64(fig: Any) -> str:
    from crl.viz.style import figure_to_base64

    return figure_to_base64(fig, dpi=500)


__all__ = ["OpeReport", "evaluate"]
