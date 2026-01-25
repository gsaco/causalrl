import numpy as np

from crl.estimators.dr_core import dr_values_from_qv
from crl.estimators.utils import compute_trajectory_returns


def test_dr_values_reduces_to_discounted_mc_when_rho_one():
    rewards = np.array(
        [
            [1.0, 0.5, -0.2],
            [0.3, -0.1, 0.9],
        ]
    )
    mask = np.array(
        [
            [True, True, False],
            [True, True, True],
        ]
    )
    discount = 0.9
    cumulative_rho = np.ones_like(rewards)

    n, t = rewards.shape
    v_hat = np.zeros((n, t + 1), dtype=float)
    q_hat = np.zeros_like(rewards)

    dr_values = dr_values_from_qv(
        rewards=rewards,
        mask=mask,
        discount=discount,
        cumulative_rho=cumulative_rho,
        v_hat=v_hat,
        q_hat=q_hat,
    )

    mc_returns = compute_trajectory_returns(rewards, mask, discount)
    assert np.allclose(dr_values, mc_returns)
