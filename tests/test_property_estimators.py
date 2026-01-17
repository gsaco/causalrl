import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.data.datasets import LoggedBanditDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.base import DiagnosticsConfig
from crl.estimators.importance_sampling import ISEstimator, WISEstimator
from crl.policies.tabular import TabularPolicy


@st.composite
def bandit_data(draw):
    n = draw(st.integers(min_value=10, max_value=50))
    rewards = draw(st.lists(st.floats(min_value=-2, max_value=2, allow_nan=False, allow_infinity=False), min_size=n, max_size=n))
    rewards = np.array(rewards, dtype=float)
    actions = draw(st.lists(st.integers(min_value=0, max_value=1), min_size=n, max_size=n))
    actions = np.array(actions, dtype=int)
    behavior_probs = draw(st.lists(st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=n, max_size=n))
    behavior_probs = np.array(behavior_probs, dtype=float)
    contexts = np.zeros(n, dtype=int)
    return contexts, actions, rewards, behavior_probs


@settings(max_examples=30)
@given(bandit_data())
def test_is_wis_permutation_invariance(bandit_data):
    contexts, actions, rewards, behavior_probs = bandit_data
    dataset = LoggedBanditDataset(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        behavior_action_probs=behavior_probs,
        action_space_n=2,
    )
    policy = TabularPolicy(np.array([[0.5, 0.5]]))
    estimand = PolicyValueEstimand(
        policy=policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
    )

    is_value = ISEstimator(estimand).estimate(dataset).value
    wis_value = WISEstimator(estimand).estimate(dataset).value

    perm = np.random.permutation(contexts.shape[0])
    dataset_perm = LoggedBanditDataset(
        contexts=contexts[perm],
        actions=actions[perm],
        rewards=rewards[perm],
        behavior_action_probs=behavior_probs[perm],
        action_space_n=2,
    )
    is_value_perm = ISEstimator(estimand).estimate(dataset_perm).value
    wis_value_perm = WISEstimator(estimand).estimate(dataset_perm).value

    assert np.isclose(is_value, is_value_perm, atol=1e-6)
    assert np.isclose(wis_value, wis_value_perm, atol=1e-6)


@settings(max_examples=30)
@given(bandit_data())
def test_is_clipping_matches_manual(bandit_data):
    contexts, actions, rewards, behavior_probs = bandit_data
    dataset = LoggedBanditDataset(
        contexts=contexts,
        actions=actions,
        rewards=rewards,
        behavior_action_probs=behavior_probs,
        action_space_n=2,
    )
    policy = TabularPolicy(np.array([[0.5, 0.5]]))
    estimand = PolicyValueEstimand(
        policy=policy,
        discount=1.0,
        horizon=1,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP]),
    )

    config = DiagnosticsConfig(max_weight=1.0)
    report = ISEstimator(estimand, diagnostics_config=config).estimate(dataset)

    ratios = policy.action_prob(contexts, actions) / behavior_probs
    clipped = np.minimum(ratios, 1.0)
    expected = float(np.mean(clipped * rewards))

    assert np.isclose(report.value, expected, atol=1e-6)
