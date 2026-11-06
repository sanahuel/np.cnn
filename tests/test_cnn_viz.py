import numpy as np

from cnn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Adam
from cnn.viz import (get_activations, plot_filters, plot_feature_maps, plot_misclassified,
                     saliency_map, plot_saliency, FilterRecorder, animate_filters)


def _toy():
    X = np.random.randn(24, 1, 8, 8)
    y = np.random.randint(0, 3, 24)
    model = Sequential([Conv2d(1, 4, 3, padding=1), ReLU(), MaxPool2d(2), Flatten(), Linear(64, 3)])
    return model, X, y


def test_all_plots_run(tmp_path):
    model, X, y = _toy()
    conv = model.layers[0]
    assert len(get_activations(model, X[0])) == len(model.layers)
    plot_filters(conv, path=tmp_path / "f.png")
    plot_feature_maps(model, X[0], path=tmp_path / "fm.png")
    plot_misclassified(X, y, (y + 1) % 3, n=6, path=tmp_path / "m.png")
    plot_saliency(model, X, [0, 1], path=tmp_path / "s.png")
    for name in ("f", "fm", "m", "s"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0


def test_saliency_shape_nonneg_and_weights_unchanged():
    model, X, _ = _toy()
    before = [p.copy() for p, _ in model.parameters()]
    sal = saliency_map(model, X[0])
    assert sal.shape == (8, 8) and np.all(sal >= 0)
    assert all(np.array_equal(b, p) for b, (p, _) in zip(before, model.parameters()))


def test_animate_filters_gif(tmp_path):
    model, X, y = _toy()
    rec = FilterRecorder(model.layers[0])
    model.fit(X, y, epochs=3, batch_size=8, optimizer=Adam(1e-2), verbose=False, callbacks=[rec])
    assert len(rec.frames) == 4  # initial + 3 epochs
    out = tmp_path / "filters.gif"
    animate_filters(rec, out)
    assert out.stat().st_size > 0
