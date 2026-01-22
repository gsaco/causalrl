from __future__ import annotations

import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import (
    MARKOV,
    OVERLAP,
    Q_MODEL_REALIZABLE,
    SEQUENTIAL_IGNORABILITY,
)
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig
from crl.benchmarks.mdp_synth import SyntheticMDP, SyntheticMDPConfig
from crl.estimands.policy_value import PolicyValueEstimand
from crl.ope import evaluate


def test_golden_bandit_workflow():
    benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0, num_contexts=4))
    dataset = benchmark.sample(num_samples=200, seed=1)
    estimand = PolicyValueEstimand(
        policy=benchmark.target_policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
    )
    report = evaluate(
        dataset=dataset,
        policy=benchmark.target_policy,
        estimand=estimand,
        estimators=["is", "wis"],
        diagnostics="default",
        seed=0,
    )

    assert "IS" in report.reports
    is_report = report.reports["IS"]
    payload = is_report.to_dict()
    assert "schema_version" in payload
    assert "diagnostics" in payload
    assert "overlap" in is_report.diagnostics
    assert "ess" in is_report.diagnostics
    assert np.isfinite(is_report.value)


def test_golden_mdp_workflow():
    benchmark = SyntheticMDP(SyntheticMDPConfig(seed=0, horizon=4))
    dataset = benchmark.sample(num_trajectories=120, seed=1)
    estimand = PolicyValueEstimand(
        policy=benchmark.target_policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet(
            [SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV, Q_MODEL_REALIZABLE]
        ),
    )
    report = evaluate(
        dataset=dataset,
        policy=benchmark.target_policy,
        estimand=estimand,
        estimators=["is", "dr", "fqe"],
        diagnostics="default",
        seed=0,
    )

    assert "DR" in report.reports
    dr_report = report.reports["DR"]
    assert "overlap" in dr_report.diagnostics
    assert "ess" in dr_report.diagnostics
    assert np.isfinite(dr_report.value)
