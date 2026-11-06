import os
from pathlib import Path

import numpy as np

from .layers import get_default_dtype

_OPENML = {"mnist": "mnist_784", "fashion": "Fashion-MNIST"}


def one_hot(y, n_classes=None, dtype=None):
    """Integer labels (N,) -> one-hot (N, C). Replaces the chain of elifs."""
    y = np.asarray(y, dtype=np.int64)
    n = int(y.max()) + 1 if n_classes is None else n_classes
    return np.eye(n, dtype=dtype or get_default_dtype())[y]


def train_test_split(X, y, test_size=0.2, seed=None):
    """Random split using numpy only. Returns X_train, X_test, y_train, y_test."""
    X, y = np.asarray(X), np.asarray(y)
    perm = np.random.default_rng(seed).permutation(len(X))
    n_test = int(round(len(X) * test_size))
    test, train = perm[:n_test], perm[n_test:]
    return X[train], X[test], y[train], y[test]


def load_moons(n_samples=1000, noise=0.1, seed=None):
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return X.astype(get_default_dtype()), y


def load_digits(normalize=True):
    """sklearn digits (8x8): X (1797, 64) in [0, 1] if normalize, y integer labels."""
    from sklearn.datasets import load_digits as _load
    X, y = _load(return_X_y=True)
    X = X.astype(get_default_dtype())
    return (X / 16.0 if normalize else X), y


def _data_home(data_home=None):
    home = Path(data_home or os.environ.get("NPDL_DATA", Path.home() / ".cache" / "npdl"))
    home.mkdir(parents=True, exist_ok=True)
    return home


def load_mnist(name="mnist", flatten=True, data_home=None):
    """MNIST or Fashion-MNIST (name='fashion') in [0, 1] with the standard 60k/10k split.

    Returns (X_train, y_train), (X_test, y_test). X is (N, 784), or (N, 1, 28, 28) if not flatten.
    """
    if name not in _OPENML:
        raise ValueError(f"name must be one of {sorted(_OPENML)}")
    home = _data_home(data_home)
    cache = home / f"{_OPENML[name]}.npz"
    if cache.exists():
        with np.load(cache) as f:
            X, y = f["X"], f["y"]
    else:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml(_OPENML[name], version=1, as_frame=False, parser="liac-arff",
                            data_home=str(home), return_X_y=True)
        X, y = X.astype(np.uint8), y.astype(np.int64)
        np.savez_compressed(cache, X=X, y=y)   # later loads take seconds
    X = X.astype(get_default_dtype()) / 255.0
    if not flatten:
        X = X.reshape(-1, 1, 28, 28)
    return (X[:60000], y[:60000]), (X[60000:], y[60000:])
