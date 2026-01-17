import numpy as np

from crl.diagnostics.ess import effective_sample_size
from crl.diagnostics.overlap import compute_overlap_metrics
from crl.diagnostics.weights import weight_tail_stats


def test_overlap_and_weight_diagnostics():
    target_probs = np.array([0.5, 0.2, 0.1])
    behavior_probs = np.array([0.5, 0.4, 0.2])
    overlap = compute_overlap_metrics(target_probs, behavior_probs)
    assert "support_violations" in overlap

    weights = target_probs / behavior_probs
    ess = effective_sample_size(weights)
    tails = weight_tail_stats(weights)

    assert ess > 0
    assert tails["max"] >= tails["q99"]
