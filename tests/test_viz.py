import numpy as np

from cnn import NN, Sequential, Linear, ReLU, Adam
from cnn.viz import (plot_history, plot_decision_boundary, DecisionBoundaryRecorder,
                    animate_decision_boundary, confusion_matrix, plot_confusion_matrix)


def _toy():
    X = np.random.randn(60, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = Sequential([Linear(2, 8), ReLU(), Linear(8, 2)])
    return model, X, y


def test_plot_history(tmp_path):
    model, X, y = _toy()
    h = model.fit(X, y, epochs=3, validation_data=(X, y), verbose=False)
    plot_history(h, path=tmp_path / "h.png")
    assert (tmp_path / "h.png").stat().st_size > 0


def test_decision_boundary_and_facade(tmp_path):
    model, X, y = _toy()
    plot_decision_boundary(model, X, y, resolution=30, path=tmp_path / "b.png")
    assert (tmp_path / "b.png").stat().st_size > 0
    nn = NN([2, 4, 2])  # facade accepted too
    plot_decision_boundary(nn, X, y, resolution=20)


def test_gif_has_frames(tmp_path):
    model, X, y = _toy()
    rec = DecisionBoundaryRecorder(X, resolution=20)
    model.fit(X, y, epochs=3, optimizer=Adam(1e-2), verbose=False, callbacks=[rec])
    assert len(rec.frames) == 3
    out = tmp_path / "b.gif"
    animate_decision_boundary(rec, X, y, out)
    assert out.stat().st_size > 0


def test_confusion_matrix_by_hand_and_plot(tmp_path):
    cm = confusion_matrix([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0])
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    plot_confusion_matrix(cm, normalize=True, path=tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 0
