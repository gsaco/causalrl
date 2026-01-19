import pytest

from crl.behavior import fit_behavior_policy
from crl.benchmarks.bandit_synth import SyntheticBandit, SyntheticBanditConfig


def test_fit_behavior_policy_bandit():
    pytest.importorskip("sklearn")

    benchmark = SyntheticBandit(SyntheticBanditConfig(seed=0))
    dataset = benchmark.sample(num_samples=200, seed=1)

    fit = fit_behavior_policy(dataset, method="logit", clip_min=1e-3, seed=0)
    assert fit.propensities.shape == dataset.actions.shape

    updated = fit.apply(dataset)
    assert updated.behavior_action_probs is not None
    assert updated.metadata.get("behavior_policy_source") == "estimated"
