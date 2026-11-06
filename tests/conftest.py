import matplotlib
matplotlib.use("Agg")  # sin ventanas en tests ni en CI

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    yield
