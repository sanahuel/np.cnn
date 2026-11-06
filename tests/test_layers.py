import numpy as np
import pytest

from cnn.layers import (Linear, ReLU, Sigmoid, Tanh, Softmax, Dropout,
                       set_default_dtype, set_seed)
from gradcheck import check_layer


@pytest.mark.parametrize("init", ["he", "xavier"])
def test_linear_gradcheck(init):
    check_layer(Linear(5, 3, init=init), np.random.randn(4, 5))


def test_relu_gradcheck():
    x = np.random.randn(4, 5)
    x[np.abs(x) < 1e-3] = 0.1  # lejos del punto no derivable
    check_layer(ReLU(), x)


@pytest.mark.parametrize("layer_cls", [Sigmoid, Tanh, Softmax])
def test_activation_gradcheck(layer_cls):
    check_layer(layer_cls(), np.random.randn(4, 5))


def test_softmax_rows_sum_to_one_and_stable():
    out = Softmax().forward(np.random.randn(4, 5) * 1000)
    assert np.all(np.isfinite(out))
    assert np.allclose(out.sum(axis=1), 1.0)


def test_dropout_eval_is_identity():
    layer = Dropout(0.5).eval()
    x = np.random.randn(4, 5)
    assert np.array_equal(layer.forward(x), x)
    g = np.random.randn(4, 5)
    assert np.array_equal(layer.backward(g), g)


def test_dropout_train_backward_uses_mask():
    layer = Dropout(0.5).train()
    x = np.ones((10, 10))
    layer.forward(x)
    g = np.random.randn(10, 10)
    assert np.array_equal(layer.backward(g), g * layer.mask)


def test_dropout_is_inverted():
    out = Dropout(0.5).train().forward(np.ones((1000, 100)))
    assert abs(out.mean() - 1.0) < 0.05


def test_he_init_std():
    set_seed(1)
    w = Linear(1000, 1000, "he").W
    assert abs(w.std() / np.sqrt(2 / 1000) - 1) < 0.05


def test_default_dtype():
    set_default_dtype(np.float32)
    layer = Linear(3, 2)
    assert layer.W.dtype == np.float32 and layer.b.dtype == np.float32
