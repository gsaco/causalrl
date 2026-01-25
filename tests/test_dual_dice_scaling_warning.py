from __future__ import annotations

import numpy as np
import pytest

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, SEQUENTIAL_IGNORABILITY
from crl.data.datasets import TrajectoryDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators import dual_dice
from crl.estimators.dual_dice import DualDICEEstimator
from crl.policies.tabular import TabularPolicy


def test_dual_dice_warns_on_large_feature_space(monkeypatch):
    monkeypatch.setattr(dual_dice, "DUALDICE_FEATURE_WARN_THRESHOLD", 10)

    observations = np.array([[0]])
    actions = np.array([[0]])
    rewards = np.array([[1.0]])
    next_observations = np.array([[0]])
    mask = np.array([[True]])
    behavior_action_probs = np.array([[1.0]])

    dataset = TrajectoryDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        behavior_action_probs=behavior_action_probs,
        mask=mask,
        discount=0.9,
        action_space_n=4,
        state_space_n=3,
    )
    policy = TabularPolicy(np.full((3, 4), 0.25))
    estimand = PolicyValueEstimand(
        policy=policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, MARKOV]),
    )

    estimator = DualDICEEstimator(estimand)
    with pytest.warns(UserWarning):
        estimator.estimate(dataset)
