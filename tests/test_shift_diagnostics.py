import numpy as np

from crl.diagnostics.shift import state_shift_diagnostics


def test_state_shift_diagnostics_keys():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(50, 3))
    weights = rng.random(50)
    diag = state_shift_diagnostics(states, weights)
    assert "mmd_rbf" in diag
    assert "mean_shift_norm" in diag
    assert "cov_shift_fro" in diag
