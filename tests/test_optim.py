import numpy as np

from cnn.optim import SGD, Adam


def test_sgd_plain_step():
    p, g = np.random.randn(5), np.random.randn(5)
    expected = p - 0.1 * g
    SGD(lr=0.1).step([(p, g)])
    assert np.allclose(p, expected)


def test_update_is_in_place():
    p = np.random.randn(4)
    view, pid = p[:], id(p)
    SGD(lr=0.5).step([(p, np.ones(4))])
    assert id(p) == pid and np.allclose(view, p)
    p2 = np.random.randn(4)
    view2 = p2[:]
    Adam(lr=0.1).step([(p2, np.ones(4))])
    assert np.allclose(view2, p2)


def test_sgd_momentum_two_steps():
    p0 = np.random.randn(3)
    p, g = p0.copy(), np.random.randn(3)
    opt = SGD(lr=0.1, momentum=0.9)
    opt.step([(p, g)])
    opt.step([(p, g)])
    assert np.allclose(p, p0 - 0.1 * g - 0.1 * (0.9 * g + g))


def test_adam_minimizes_quadratic():
    p = np.zeros(3)
    opt = Adam(lr=0.05)
    for _ in range(2000):
        opt.step([(p, 2 * (p - 3))])
    assert np.all(np.abs(p - 3) < 1e-2)


def test_adam_first_step_magnitude():
    p = np.zeros(4)
    Adam(lr=0.01).step([(p, np.array([1.0, -5.0, 0.3, 100.0]))])
    assert np.allclose(np.abs(p), 0.01, rtol=1e-3)
