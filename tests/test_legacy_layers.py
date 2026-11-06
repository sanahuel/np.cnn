# Temporal: se elimina en la Fase 3.3, cuando lo sustituyan test_conv.py y test_model.py.
import numpy as np
import pytest

from cnn.model import FCLayer, ConvLayer
from gradcheck import numerical_grad, rel_error


@pytest.mark.parametrize("activation", ["relu", "sigmoid"])
def test_fclayer_gradients(activation):
    layer = FCLayer(5, 4, activation)
    x = np.random.randn(5, 1)
    R = np.random.randn(4, 1)
    if activation == "relu":  # alejar las preactivaciones del 0 (punto no derivable)
        layer.b[:] = np.where(np.abs(layer.W @ x) < 1e-2, 0.5, 0.0)
    loss = lambda: float(np.sum(layer.forward(x) * R))
    loss()
    dx, dW, db = layer.backward(R)
    assert rel_error(dx, numerical_grad(loss, x)) < 1e-5
    assert rel_error(dW, numerical_grad(loss, layer.W)) < 1e-5
    assert rel_error(db, numerical_grad(loss, layer.b)) < 1e-5


def test_softmax_layer_gradient_matches_cross_entropy():
    layer = FCLayer(5, 3, activation='softmax')
    x = np.random.randn(5, 1)
    y = np.array([[0.], [1.], [0.]])

    def loss():
        p = layer.forward(x)
        return float(-np.sum(y * np.log(p)))

    p = layer.forward(x)
    dx, dW, db = layer.backward(p - y)
    assert rel_error(dW, numerical_grad(loss, layer.W)) < 1e-5
    assert rel_error(db, numerical_grad(loss, layer.b)) < 1e-5
    assert rel_error(dx, numerical_grad(loss, x)) < 1e-5


def test_sigmoid_does_not_overflow():
    layer = FCLayer(1, 1, 'sigmoid')
    with np.errstate(over='raise'):
        out = layer.sigmoid(np.array([[-1e4], [1e4]]))
        layer.sigmoid(np.array([[-1e4], [1e4]]), derivative=True)
    assert np.allclose(out, [[0.0], [1.0]])


@pytest.mark.parametrize("padding,stride", [(0, 1), (1, 1), (1, 2), (0, 2)])
def test_convlayer_gradients(padding, stride):
    layer = ConvLayer(k_dim=3, stride=stride, padding=padding)
    x = np.random.randn(1, 7, 7)
    out = layer.forward(x)
    R = np.random.randn(*out.shape)
    loss = lambda: float(np.sum(layer.forward(x) * R))
    # analytic gradients first: numerical_grad calls forward again and overwrites the caches
    loss()
    dx, dW, db = layer.backward(R)
    assert dx.shape == x.shape  # padding removed
    assert rel_error(dx, numerical_grad(loss, x)) < 1e-5
    assert rel_error(dW, numerical_grad(loss, layer.filter)) < 1e-5
    assert rel_error(db, numerical_grad(loss, layer.bias)) < 1e-5


def test_chained_convs_with_padding():
    c1 = ConvLayer(k_dim=3, stride=1, padding=1)
    c2 = ConvLayer(k_dim=3, stride=1, padding=1)
    x = np.random.randn(1, 6, 6)
    R = np.random.randn(1, 6, 6)

    def loss():
        return float(np.sum(c2.forward(c1.forward(x)) * R))

    loss()                                    # forward para llenar las cachés
    d2, _, _ = c2.backward(R)
    d1, dW1, db1 = c1.backward(d2)            # el gradiente atraviesa el padding de c2

    assert d1.shape == x.shape
    assert rel_error(dW1, numerical_grad(loss, c1.filter)) < 1e-5
    assert rel_error(d1, numerical_grad(loss, x)) < 1e-5
