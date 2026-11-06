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
