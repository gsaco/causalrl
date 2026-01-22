import warnings

import numpy as np

from crl.assumptions import AssumptionSet
from crl.assumptions_catalog import MARKOV, OVERLAP, SEQUENTIAL_IGNORABILITY
from crl.data.datasets import TrajectoryDataset
from crl.estimands.policy_value import PolicyValueEstimand
from crl.estimators.wdr import WDRConfig, WeightedDoublyRobustEstimator
from crl.policies.tabular import TabularPolicy


def test_wdr_zero_weight_timesteps_no_divide_warning():
    policy = TabularPolicy(np.array([[1.0, 0.0], [1.0, 0.0]]))

    observations = np.array([[0, 1], [0, 1], [1, 0]], dtype=int)
    next_observations = np.array([[1, 0], [1, 0], [0, 1]], dtype=int)
    actions = np.ones((3, 2), dtype=int)
    rewards = np.zeros((3, 2), dtype=float)
    mask = np.ones_like(actions, dtype=bool)
    behavior_action_probs = np.full_like(actions, 0.5, dtype=float)

    dataset = TrajectoryDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        behavior_action_probs=behavior_action_probs,
        mask=mask,
        discount=1.0,
        action_space_n=2,
        state_space_n=2,
    )

    estimand = PolicyValueEstimand(
        policy=policy,
        discount=dataset.discount,
        horizon=dataset.horizon,
        assumptions=AssumptionSet([SEQUENTIAL_IGNORABILITY, OVERLAP, MARKOV]),
    )

    estimator = WeightedDoublyRobustEstimator(
        estimand,
        config=WDRConfig(num_folds=2, num_iterations=1, ridge=1e-3, seed=0),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = estimator.estimate(dataset)

    assert np.isfinite(report.value)
    assert all(
        "where' used without 'out'" not in str(warning.message) for warning in caught
    )
