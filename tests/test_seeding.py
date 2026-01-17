import random

import numpy as np

from crl import set_seed


def test_set_seed_reproducible():
    set_seed(123)
    first_random = random.random()
    first_numpy = np.random.rand()

    set_seed(123)
    assert random.random() == first_random
    assert np.random.rand() == first_numpy
