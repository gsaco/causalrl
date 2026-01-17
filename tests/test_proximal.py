import numpy as np

from crl.benchmarks.confounded_bandit import ConfoundedBandit, ConfoundedBanditConfig
from crl.confounding.proximal import ProximalBanditEstimator


def test_proximal_bandit_estimator_runs():
    bench = ConfoundedBandit(ConfoundedBanditConfig(seed=0))
    data = bench.sample(num_samples=500, seed=1)
    estimate = ProximalBanditEstimator(bench.target_policy).estimate(data)
    assert np.isfinite(estimate)
