import numpy as np
import pytest

from cnn import (Sequential, Linear, ReLU, Tanh, Dropout, CrossEntropy, Adam, NN, FCLayer,
                set_seed, set_default_dtype)
from gradcheck import numerical_grad, rel_error


def _mlp(i=10, h=64, o=4):
    return Sequential([Linear(i, h), ReLU(), Linear(h, o)])


def test_end_to_end_gradcheck():
    model = Sequential([Linear(4, 6), Tanh(), Linear(6, 3)])
    x, y, loss = np.random.randn(5, 4), np.random.randint(0, 3, 5), CrossEntropy()
    loss(model.forward(x), y)
    model.backward(loss.backward())
    for p, g in model.parameters():
        g = g.copy()
        numeric = numerical_grad(lambda: CrossEntropy()(model.forward(x), y), p)
        assert rel_error(g, numeric) < 1e-5


def test_overfit_small_batch():
    X = np.random.randn(32, 10)
    y = np.random.randint(0, 4, 32)
    model = _mlp()
    hist = model.fit(X, y, epochs=300, batch_size=32, optimizer=Adam(lr=1e-2), verbose=False)
    assert hist["loss"][-1] < 0.05


def test_history_keys_and_lengths():
    X, y = np.random.randn(40, 10), np.random.randint(0, 4, 40)
    model = _mlp()
    h = model.fit(X, y, epochs=3, batch_size=16, verbose=False)
    assert set(h) == {"loss", "acc"} and all(len(v) == 3 for v in h.values())
    h = model.fit(X, y, epochs=2, batch_size=16, validation_data=(X, y), verbose=False)
    assert set(h) == {"loss", "acc", "val_loss", "val_acc"} and all(len(v) == 2 for v in h.values())
    assert model.history is h


def test_save_load_roundtrip(tmp_path):
    X = np.random.randn(8, 10)
    a = _mlp()
    a.save(tmp_path / "w.npz")
    set_seed(123)
    b = _mlp()
    assert not np.allclose(a.predict_proba(X), b.predict_proba(X))
    b.load(tmp_path / "w.npz")
    assert np.allclose(a.predict_proba(X), b.predict_proba(X))
    with pytest.raises(ValueError):
        _mlp(10, 32, 4).load(tmp_path / "w.npz")


def test_predict_proba_rows_sum_to_one():
    p = _mlp().predict_proba(np.random.randn(7, 10))
    assert p.shape == (7, 4) and np.allclose(p.sum(axis=1), 1.0)


def test_fit_uses_model_dtype():
    set_default_dtype(np.float32)
    model = _mlp()
    X, y = np.random.randn(16, 10), np.random.randint(0, 4, 16)  # float64 input
    model.fit(X, y, epochs=1, batch_size=8, verbose=False)
    assert model.forward(X.astype(np.float32)).dtype == np.float32


def test_dropout_disabled_at_inference():
    model = Sequential([Linear(10, 8), ReLU(), Dropout(0.5), Linear(8, 4)])
    X = np.random.randn(5, 10)
    assert np.allclose(model.predict_proba(X), model.predict_proba(X))


def test_sequential_unrolls_composites_and_repr():
    model = Sequential([FCLayer(4, 3, "relu"), FCLayer(3, 2, "softmax")])
    assert [type(l).__name__ for l in model.layers] == ["Linear", "ReLU", "Linear"]
    assert "parameters" in repr(model)


# --- legacy facade ---------------------------------------------------------
def _legacy_data(n=60):
    X = np.random.randn(n, 2, 1)
    y = (X[:, 0, 0] > 0).astype(int)
    return X, np.eye(2)[y].reshape(n, 2, 1)


def test_legacy_train_and_test_return_values(capsys):
    X, Y = _legacy_data()
    nn = NN([2, 8, 2])
    hist = nn.train(X, Y, epochs=2, lr=0.05)
    acc = nn.test(X, Y)
    out = capsys.readouterr().out
    assert isinstance(hist, dict) and len(hist["loss"]) == 2
    assert isinstance(acc, float) and 0.0 <= acc <= 1.0
    assert "Training for 2 epochs" in out and "Test accuracy" in out


def test_legacy_verbose_false_prints_nothing(capsys):
    X, Y = _legacy_data()
    nn = NN([2, 8, 2])
    nn.train(X, Y, epochs=1, lr=0.05, verbose=False)
    nn.test(X, Y, verbose=False)
    assert capsys.readouterr().out == ""


def test_legacy_functions_and_errors():
    NN([2, 4, 2])
    NN([2, 4, 2], "tanh")
    NN([2, 4, 2], "sigmoid")
    with pytest.raises(ValueError):
        NN([2, 4, 2], "foo")


def test_legacy_accepts_2d_inputs():
    X, Y = _legacy_data()
    nn = NN([2, 8, 2])
    nn.train(X[..., 0], Y[..., 0], epochs=1, lr=0.05, verbose=False)
    assert 0.0 <= nn.test(X[..., 0], Y[..., 0], verbose=False) <= 1.0
