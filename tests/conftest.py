import matplotlib
matplotlib.use("Agg")  # sin ventanas en tests ni en CI

import numpy as np
import pytest

from cnn.layers import set_default_dtype, set_seed


@pytest.fixture(autouse=True)
def _seed_and_dtype():
    np.random.seed(0)
    set_seed(0)
    set_default_dtype(np.float64)   # gradient checks en float64
    yield
    set_default_dtype(np.float32)
