from __future__ import annotations

import pytest

from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.evaluation import DiagnosticsSpec, EvaluationSpec
from crl.ope import evaluate, evaluate_ope, run_spec


def test_run_spec_returns_result():
    bench = SyntheticBandit(SyntheticBanditConfig(seed=0))
    dataset = bench.sample(num_samples=200, seed=1)

    spec = EvaluationSpec(
        policy=bench.target_policy,
        dataset=dataset,
        estimators=("is",),
        diagnostics=DiagnosticsSpec(enabled=False),
    )
    result = run_spec(spec)
    df = result.report.to_dataframe()

    assert "IS" in set(df["estimator"])


def test_evaluate_spec_warns():
    bench = SyntheticBandit(SyntheticBanditConfig(seed=1))
    dataset = bench.sample(num_samples=100, seed=2)

    spec = EvaluationSpec(
        policy=bench.target_policy,
        dataset=dataset,
        estimators=("is",),
        diagnostics=DiagnosticsSpec(enabled=False),
    )

    with pytest.warns(DeprecationWarning):
        result = evaluate(spec)

    assert result.report is not None


def test_evaluate_ope_matches_direct_use():
    bench = SyntheticBandit(SyntheticBanditConfig(seed=2))
    dataset = bench.sample(num_samples=150, seed=3)

    report = evaluate_ope(dataset=dataset, policy=bench.target_policy)
    assert not report.to_dataframe().empty
