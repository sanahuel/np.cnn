import numpy as np
import pytest

from cnn import (Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, ConvLayer, CrossEntropy,
                 Adam, set_default_dtype)
from cnn.reference import conv2d_forward_loops, conv2d_backward_loops
from gradcheck import check_layer, numerical_grad, rel_error

CONV_CASES = [  # (C_in, C_out, k, stride, padding, H)
    (1, 1, 3, 1, 0, 6),
    (2, 3, 3, 1, 1, 6),
    (2, 3, 3, 2, 1, 7),   # stride that does not divide the size
    (3, 2, 2, 2, 0, 6),
    (1, 2, 5, 1, 2, 6),
    (2, 2, 3, 3, 1, 8),
]


@pytest.mark.parametrize("cin,cout,k,s,p,H", CONV_CASES)
def test_conv2d_gradcheck(cin, cout, k, s, p, H):
    layer = Conv2d(cin, cout, k, stride=s, padding=p)
    layer.b[:] = np.random.randn(cout)            # non-zero bias so it is really checked
    check_layer(layer, np.random.randn(2, cin, H, H))


@pytest.mark.parametrize("cin,cout,k,s,p,H", CONV_CASES)
def test_im2col_matches_loops(cin, cout, k, s, p, H):
    layer = Conv2d(cin, cout, k, stride=s, padding=p)
    x = np.random.randn(2, cin, H, H)
    out = layer.forward(x)
    assert rel_error(out, conv2d_forward_loops(x, layer.W, layer.b, s, p)) < 1e-12
    g = np.random.randn(*out.shape)
    dx = layer.backward(g)
    dx_ref, dW_ref, db_ref = conv2d_backward_loops(g, x, layer.W, s, p)
    assert rel_error(dx, dx_ref) < 1e-12
    assert rel_error(layer.dW, dW_ref) < 1e-12
    assert rel_error(layer.db, db_ref) < 1e-12


@pytest.mark.parametrize("k,stride", [(2, None), (3, 2), (2, 1)])
def test_maxpool_gradcheck(k, stride):
    shape = (2, 3, 7, 7)
    n = int(np.prod(shape))
    # no ties: the max is well defined and finite differences do not cross cells
    x = np.random.permutation(n).reshape(shape).astype(np.float64) / 10 + 1e-3 * np.random.randn(*shape)
    check_layer(MaxPool2d(k, stride), x)


def test_maxpool_by_hand():
    x = np.array([[[[1., 2., 5., 0.],
                    [3., 4., 1., 1.],
                    [0., 0., 9., 8.],
                    [7., 1., 6., 2.]]]])
    pool = MaxPool2d(2)
    out = pool.forward(x)
    assert out.tolist() == [[[[4., 5.], [7., 9.]]]]
    dx = pool.backward(np.array([[[[10., 20.], [30., 40.]]]]))
    expected = np.zeros_like(x)
    expected[0, 0, 1, 1], expected[0, 0, 0, 2] = 10., 20.
    expected[0, 0, 3, 0], expected[0, 0, 2, 2] = 30., 40.
    assert np.array_equal(dx, expected)


def test_flatten_roundtrip():
    f = Flatten()
    x = np.random.randn(3, 2, 4, 5)
    out = f.forward(x)
    assert out.shape == (3, 40)
    assert f.backward(out).shape == x.shape and np.array_equal(f.backward(out), x)


def test_shapes():
    x = np.random.randn(5, 1, 28, 28)
    y = Conv2d(1, 8, 3, padding=1).forward(x)
    assert y.shape == (5, 8, 28, 28)
    assert MaxPool2d(2).forward(y).shape == (5, 8, 14, 14)


def _small_cnn():
    return Sequential([Conv2d(1, 2, 3, padding=1), ReLU(), MaxPool2d(2), Flatten(), Linear(18, 3)])


def test_end_to_end_gradcheck():
    model = _small_cnn()
    x = np.random.randn(2, 1, 6, 6)
    y = np.random.randint(0, 3, 2)
    loss = CrossEntropy()
    loss(model.forward(x), y)
    dx = model.backward(loss.backward())
    analytic = [g.copy() for _, g in model.parameters()]
    f = lambda: CrossEntropy()(model.forward(x), y)
    for (p, _), g in zip(model.parameters(), analytic):
        assert rel_error(g, numerical_grad(f, p)) < 1e-5
    assert rel_error(dx, numerical_grad(f, x)) < 1e-5


def test_overfit_small_batch():
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    X = (X[:16] / 16.0).reshape(16, 1, 8, 8)
    model = Sequential([Conv2d(1, 4, 3, padding=1), ReLU(), MaxPool2d(2), Flatten(), Linear(64, 10)])
    hist = model.fit(X, y[:16], epochs=200, batch_size=16, optimizer=Adam(lr=1e-2), verbose=False)
    assert hist["loss"][-1] < 0.05


def test_convlayer_facade():
    layer = ConvLayer(in_channels=3, out_channels=4)
    out = Sequential([layer]).forward(np.random.randn(2, 3, 8, 8))
    assert out.shape == (2, 4, 8, 8)
    assert [type(l).__name__ for l in layer.unroll()] == ["Conv2d", "ReLU"]
    assert np.all(out >= 0)


def test_conv_dtype_float32():
    set_default_dtype(np.float32)
    layer = Conv2d(1, 2, 3, padding=1)
    assert layer.forward(np.random.randn(2, 1, 5, 5).astype(np.float32)).dtype == np.float32
