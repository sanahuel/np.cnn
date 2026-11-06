import numpy as np
from sklearn.datasets import load_digits

from cnn.model import NN, CNN, FCLayer, Flatten, ConvLayer


def _digits_legacy(n=400):
    X, y = load_digits(return_X_y=True)
    Y = np.eye(10)[y].reshape(len(y), 10, 1)            # (N, 10, 1)
    return X[:n], Y[:n]


def _cnn():
    return CNN(layers=[
        ConvLayer(k_dim=3, padding=1, stride=1),
        ConvLayer(k_dim=3, padding=1, stride=1),
        Flatten(),
        FCLayer(n_input=64, n_output=128, activation='relu'),
        FCLayer(n_input=128, n_output=64, activation='relu'),
        FCLayer(n_input=64, n_output=32, activation='relu'),
        FCLayer(n_input=32, n_output=10, activation='softmax'),
    ])


def test_nn_legacy_shapes():
    X, Y = _digits_legacy()
    X = X.reshape(len(X), 64, 1)                        # (N, 64, 1)
    nn = NN(layers=[64, 128, 64, 32, 10], function='relu')
    nn.train(X, Y, epochs=3, lr=0.0001)
    nn.test(X, Y)


def test_cnn_legacy_notebook(capsys):
    X, Y = _digits_legacy()
    X = X.reshape(len(X), 1, 8, 8)                      # (N, 1, 8, 8)
    cnn = _cnn()
    cnn.train(X, Y, epochs=3, lr=0.001)
    cnn.test(X, Y)
    assert "accuracy" in capsys.readouterr().out.lower()


def test_cnn_forward_single_sample():
    cnn = CNN([ConvLayer(), Flatten(), FCLayer(64, 10, 'softmax')])
    out = cnn.forward(np.random.randn(1, 8, 8))
    assert out.shape == (10, 1)
    assert np.isclose(out.sum(), 1.0)


def test_cnn_legacy_min_accuracy(capsys):
    # umbral = 0.9 x la peor accuracy observada en 3 semillas (0.7375)
    X, Y = _digits_legacy()
    X = X.reshape(len(X), 1, 8, 8)
    cnn = _cnn()
    cnn.train(X, Y, epochs=5, lr=0.001)
    capsys.readouterr()
    cnn.test(X, Y)
    out = capsys.readouterr().out
    acc = float(out.split(":")[1].strip().rstrip("%")) / 100
    assert acc >= 0.66
