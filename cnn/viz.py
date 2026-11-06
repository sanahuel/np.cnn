import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def _as_sequential(model):
    return getattr(model, "model", model)


def _finish(fig, path):
    if path is not None:
        fig.savefig(path, dpi=100, bbox_inches="tight")
    return fig


def plot_history(history, path=None):
    """Loss and accuracy curves (train, and validation if present)."""
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(10, 3.5))
    for key, label in (("loss", "train"), ("val_loss", "validation")):
        if key in history:
            ax_l.plot(range(1, len(history[key]) + 1), history[key], label=label)
    ax_l.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_l.legend()
    for key, label in (("acc", "train"), ("val_acc", "validation")):
        if key in history and not np.all(np.isnan(history[key])):
            ax_a.plot(range(1, len(history[key]) + 1), history[key], label=label)
    ax_a.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    ax_a.legend()
    fig.tight_layout()
    return _finish(fig, path)


def _grid(X, resolution, margin=0.5):
    x0, x1 = X[:, 0].min() - margin, X[:, 0].max() + margin
    y0, y1 = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x0, x1, resolution), np.linspace(y0, y1, resolution))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def plot_decision_boundary(model, X, y, ax=None, resolution=200, path=None):
    """Probability map of class 1 (binary) or predicted class (multiclass) with the data on top."""
    seq = _as_sequential(model)
    X, y = np.asarray(X), np.asarray(y)
    if y.ndim == 2:
        y = y.argmax(1)
    xx, yy, grid = _grid(X, resolution)
    if len(np.unique(y)) <= 2:
        z = seq.predict_proba(grid)[:, 1].reshape(xx.shape)
        levels = np.linspace(0, 1, 21)
    else:
        z = seq.predict(grid).reshape(xx.shape)
        levels = np.arange(z.max() + 2) - 0.5
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    ax.contourf(xx, yy, z, levels=levels, cmap="RdBu_r", alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolors="k", s=18)
    ax.set(xlabel="x1", ylabel="x2", title="Decision boundary")
    _finish(fig, path)
    return ax


class DecisionBoundaryRecorder:
    """fit() callback: stores the probability grid every `every` epochs (not the model)."""

    def __init__(self, X, every=1, resolution=100, margin=0.5):
        X = np.asarray(X)
        self.xx, self.yy, self.grid = _grid(X, resolution, margin)
        self.every, self.frames = every, []

    def __call__(self, model, epoch, logs):
        if epoch % self.every == 0:
            probs = _as_sequential(model).predict_proba(self.grid)[:, 1].reshape(self.xx.shape)
            self.frames.append((epoch, probs, logs.get("acc")))


def animate_decision_boundary(recorder, X, y, path, fps=8, max_frames=60):
    """GIF of the decision boundary evolving during training."""
    X, y = np.asarray(X), np.asarray(y)
    if y.ndim == 2:
        y = y.argmax(1)
    frames = recorder.frames
    if len(frames) > max_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).astype(int)]
    fig, ax = plt.subplots(figsize=(4, 4))

    def draw(k):
        epoch, probs, acc = frames[k]
        ax.clear()
        ax.contourf(recorder.xx, recorder.yy, probs, levels=np.linspace(0, 1, 21),
                    cmap="RdBu_r", alpha=0.7)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolors="k", s=12)
        title = f"epoch {epoch + 1}"
        if acc is not None and not np.isnan(acc):
            title += f" - acc {acc * 100:.1f}%"
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    anim = FuncAnimation(fig, draw, frames=len(frames))
    anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return path


def confusion_matrix(y_true, y_pred, n_classes=None):
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    n = n_classes or int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def plot_confusion_matrix(cm, class_names=None, normalize=False, path=None, ax=None):
    cm = np.asarray(cm)
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    im = ax.imshow(shown, cmap="Blues")
    fig.colorbar(im, ax=ax)
    n = len(cm)
    names = class_names if class_names is not None else [str(i) for i in range(n)]
    ax.set(xticks=range(n), yticks=range(n), xticklabels=names, yticklabels=names,
           xlabel="predicted", ylabel="true", title="Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = shown.max() / 2 if shown.size else 0
    for i in range(n):
        for j in range(n):
            text = f"{shown[i, j]:.2f}" if normalize else str(shown[i, j])
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if shown[i, j] > thresh else "black")
    fig.tight_layout()
    _finish(fig, path)
    return ax


# =============================================================================
# Solo np.cnn: todo lo que hay por encima de esta línea es idéntico a np.nn
# =============================================================================
from .layers import Conv2d  # noqa: E402


def _grid_axes(n, ncols):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    return fig, axes


def get_activations(model, x):
    """Forward of ONE image (C, H, W) in eval mode, layer by layer: [(layer, output), ...]."""
    seq = _as_sequential(model)
    seq.eval()
    out = np.asarray(x, dtype=seq.dtype)[None]
    acts = []
    for layer in seq.layers:
        out = layer.forward(out)
        acts.append((layer, out))
    return acts


def plot_filters(conv, path=None, ncols=8):
    """Grid of the learned filters (channel 0 if the conv has several input channels)."""
    W = conv.W[:, 0] if conv.W.ndim == 4 else conv.W
    vmax = np.abs(W).max() or 1.0
    fig, axes = _grid_axes(len(W), min(ncols, len(W)))
    for f, ax in zip(range(len(W)), axes.ravel()):
        ax.imshow(W[f], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.suptitle("Learned filters")
    return _finish(fig, path)


def plot_feature_maps(model, x, path=None, max_maps=8):
    """One row per Conv2d: the activation (output of the following layer) of up to max_maps channels."""
    acts = get_activations(model, x)
    rows = []
    for i, (layer, out) in enumerate(acts):
        if isinstance(layer, Conv2d):
            rows.append(acts[i + 1][1][0] if i + 1 < len(acts) else out[0])
    fig, axes = plt.subplots(len(rows), max_maps, figsize=(1.6 * max_maps, 1.7 * len(rows)),
                             squeeze=False)
    for r, maps in enumerate(rows):
        for c in range(max_maps):
            axes[r, c].axis("off")
            if c < len(maps):
                axes[r, c].imshow(maps[c], cmap="viridis")
        axes[r, 0].set_title(f"conv {r + 1}", fontsize=8, loc="left")
    fig.tight_layout()
    return _finish(fig, path)


def plot_misclassified(X, y_true, y_pred, class_names=None, n=16, path=None, ncols=8):
    """Grid of wrongly classified images, titled 'pred / true'."""
    X, y_true, y_pred = np.asarray(X), np.asarray(y_true), np.asarray(y_pred)
    wrong = np.flatnonzero(y_true != y_pred)[:n]
    names = class_names if class_names is not None else [str(i) for i in range(int(y_true.max()) + 1)]
    fig, axes = _grid_axes(max(len(wrong), 1), ncols)
    for idx, ax in zip(wrong, axes.ravel()):
        img = X[idx]
        ax.imshow(img[0] if img.ndim == 3 else img.reshape(int(np.sqrt(img.size)), -1), cmap="gray")
        ax.set_title(f"{names[y_pred[idx]]} / {names[y_true[idx]]}", fontsize=7)
    fig.suptitle("Misclassified (predicted / true)")
    return _finish(fig, path)


def saliency_map(model, x, target=None):
    """|d logit_target / d x|, max over channels, for one image (C, H, W). Returns (H, W)."""
    seq = _as_sequential(model)
    seq.eval()
    xb = np.asarray(x, dtype=seq.dtype)[None]
    logits = seq.forward(xb)
    target = int(logits.argmax()) if target is None else int(target)
    g = np.zeros_like(logits)
    g[0, target] = 1.0
    dx = seq.backward(g)   # overwrites stored grads, but without optimizer.step the weights are unchanged
    return np.abs(dx[0]).max(axis=0)


def plot_saliency(model, X, indices, class_names=None, path=None):
    """One row per image: the original and its saliency map."""
    seq = _as_sequential(model)
    fig, axes = plt.subplots(len(indices), 2, figsize=(3.2, 1.7 * len(indices)), squeeze=False)
    for r, idx in enumerate(indices):
        x = np.asarray(X[idx])
        pred = int(seq.predict(x[None])[0])
        sal = saliency_map(seq, x, pred)
        axes[r, 0].imshow(x[0], cmap="gray")
        axes[r, 1].imshow(sal, cmap="hot")
        label = class_names[pred] if class_names is not None else pred
        axes[r, 0].set_title(f"pred: {label}", fontsize=8)
        axes[r, 1].set_title("saliency", fontsize=8)
        axes[r, 0].axis("off")
        axes[r, 1].axis("off")
    fig.tight_layout()
    return _finish(fig, path)


class FilterRecorder:
    """fit() callback: stores a copy of conv.W every `every` epochs."""

    def __init__(self, conv, every=1):
        self.conv, self.every = conv, every
        self.frames = [(-1, conv.W.copy())]   # initial filters

    def __call__(self, model, epoch, logs):
        if epoch % self.every == 0:
            self.frames.append((epoch, self.conv.W.copy()))


def animate_filters(recorder, path, fps=4, ncols=4):
    """GIF of how the filters of the first input channel evolve during training."""
    frames = recorder.frames
    vmax = max(np.abs(W).max() for _, W in frames) or 1.0
    n = frames[0][1].shape[0]
    ncols = min(ncols, n)
    fig, axes = _grid_axes(n, ncols)
    images = [ax.imshow(np.zeros(frames[0][1].shape[-2:]), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
              for ax in axes.ravel()[:n]]

    def draw(k):
        epoch, W = frames[k]
        for f, im in enumerate(images):
            im.set_data(W[f, 0])
        fig.suptitle("initial filters" if epoch < 0 else f"epoch {epoch + 1}")
        return images

    anim = FuncAnimation(fig, draw, frames=len(frames))
    anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return path
