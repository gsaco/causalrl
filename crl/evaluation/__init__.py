"""Evaluation APIs for CRL."""

from crl.evaluation.decision import DecisionResult, DecisionSpec
from crl.evaluation.result import EvaluationResult
from crl.evaluation.spec import (
    DiagnosticsSpec,
    EvaluationSpec,
    InferenceSpec,
    ReportSpec,
    SensitivitySpec,
)
from crl.evaluation.suite import EvaluationSuite, evaluate_many

__all__ = [
    "EvaluationSpec",
    "InferenceSpec",
    "DiagnosticsSpec",
    "SensitivitySpec",
    "ReportSpec",
    "EvaluationResult",
    "DecisionSpec",
    "DecisionResult",
    "EvaluationSuite",
    "evaluate_many",
]
