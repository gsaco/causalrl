import numpy as np

from crl.benchmarks.harness import run_all_benchmarks


def test_run_all_benchmarks_outputs():
    results = run_all_benchmarks(num_samples=200, num_trajectories=50, seed=0)
    assert len(results) == 7
    for record in results:
        assert np.isfinite(record["estimate"])
        assert np.isfinite(record["true_value"])
