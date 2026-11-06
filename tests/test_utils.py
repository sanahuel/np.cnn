import numpy as np
import pytest

from cnn.utils import one_hot, train_test_split, load_digits, load_moons, load_mnist


def test_one_hot():
    oh = one_hot(np.array([0, 2, 1]), 3)
    assert oh.shape == (3, 3) and np.all(oh.sum(axis=1) == 1)
    assert oh.argmax(1).tolist() == [0, 2, 1]
    assert one_hot([0, 1], 2, dtype=np.float32).dtype == np.float32


def test_train_test_split_sizes_disjoint_reproducible():
    X, y = np.arange(100).reshape(50, 2), np.arange(50)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, seed=3)
    assert len(Xte) == 10 and len(Xtr) == 40
    assert set(ytr).isdisjoint(yte)
    assert np.array_equal(Xtr[:, 0] // 2, ytr)  # X and y stay aligned
    again = train_test_split(X, y, test_size=0.2, seed=3)
    assert np.array_equal(again[3], yte)


def test_load_digits():
    X, y = load_digits()
    assert X.shape == (1797, 64) and X.min() >= 0 and X.max() <= 1
    assert y.shape == (1797,)


def test_load_moons():
    X, y = load_moons(200, seed=0)
    assert X.shape == (200, 2) and set(np.unique(y)) == {0, 1}


@pytest.mark.slow
def test_load_mnist():
    (Xtr, ytr), (Xte, yte) = load_mnist()
    assert Xtr.shape == (60000, 784) and Xte.shape == (10000, 784)
    (Xtr, _), (Xte, _) = load_mnist(flatten=False)
    assert Xtr.shape == (60000, 1, 28, 28) and Xte.shape == (10000, 1, 28, 28)
