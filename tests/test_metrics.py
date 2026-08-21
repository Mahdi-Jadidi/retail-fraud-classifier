import numpy as np
from retail_fraud.metrics import tune_threshold


def test_threshold_search_finds_useful_cutoff():
    threshold, score = tune_threshold(np.array([0, 0, 1, 1]), np.array([.1, .2, .7, .8]))
    assert 0.05 <= threshold <= 0.95
    assert score == 1.0
