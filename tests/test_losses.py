import numpy as np

from cnn.losses import CrossEntropy, MSE
from gradcheck import numerical_grad, rel_error


def test_cross_entropy_gradient_int_and_onehot():
    logits = np.random.randn(5, 4)
    y = np.random.randint(0, 4, 5)
    for target in (y, np.eye(4)[y]):
        ce = CrossEntropy()
        ce(logits, target)
        analytic = ce.backward()
        numeric = numerical_grad(lambda: CrossEntropy()(logits, target), logits)
        assert rel_error(analytic, numeric) < 1e-5


def test_cross_entropy_stable():
    ce = CrossEntropy()
    loss = ce(np.random.randn(5, 4) * 1e3, np.random.randint(0, 4, 5))
    assert np.isfinite(loss) and np.all(np.isfinite(ce.backward()))


def test_cross_entropy_value_uniform():
    assert np.isclose(CrossEntropy()(np.zeros((3, 10)), np.array([0, 5, 9])), np.log(10))


def test_mse_gradient():
    pred, target = np.random.randn(6, 3), np.random.randn(6, 3)
    mse = MSE()
    mse(pred, target)
    analytic = mse.backward()
    assert rel_error(analytic, numerical_grad(lambda: MSE()(pred, target), pred)) < 1e-5
