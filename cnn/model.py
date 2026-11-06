import numpy as np

from .layers import (Layer, Linear, ReLU, Sigmoid, Tanh, get_default_dtype, get_rng)
from .losses import CrossEntropy
from .optim import SGD, Adam


def softmax(z):
    """Row-wise, numerically stable softmax."""
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _labels(y):
    """One-hot (N, C) or integer (N,) -> integer labels (N,)."""
    y = np.asarray(y)
    return y.argmax(axis=1) if y.ndim == 2 else y


def _from_legacy(a):
    """(N, F, 1) -> (N, F). Any other shape is left unchanged."""
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[-1] == 1:
        return a[..., 0]
    return a


def _format_epoch(epoch, epochs, logs):
    msg = f"Epoch {epoch + 1}/{epochs} - loss: {logs['loss']:.4f}"
    if not np.isnan(logs["acc"]):
        msg += f" - accuracy: {logs['acc'] * 100:.3f}%"
    if "val_loss" in logs:
        msg += f" - val_loss: {logs['val_loss']:.4f}"
        if not np.isnan(logs["val_acc"]):
            msg += f" - val_accuracy: {logs['val_acc'] * 100:.3f}%"
    return msg


class Sequential:
    """Chain of layers. forward() returns logits; the loss (CrossEntropy) includes the softmax."""

    def __init__(self, layers):
        self.layers = []
        for layer in layers:
            if hasattr(layer, "unroll"):
                self.layers.extend(layer.unroll())
            else:
                self.layers.append(layer)
        self.history = None

    # --- core -----------------------------------------------------------
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    __call__ = forward

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        """[(param, grad), ...] in a stable order."""
        out = []
        for layer in self.layers:
            p, g = layer.params(), layer.grads()
            out.extend((p[k], g[k]) for k in p)
        return out

    def train(self):
        for layer in self.layers:
            layer.train()
        return self

    def eval(self):
        for layer in self.layers:
            layer.eval()
        return self

    @property
    def dtype(self):
        for layer in self.layers:
            for p in layer.params().values():
                return p.dtype
        return np.dtype(get_default_dtype())

    # --- training -------------------------------------------------------
    def fit(self, X, y, epochs=10, batch_size=32, optimizer=None, loss=None,
            validation_data=None, shuffle=True, verbose=True, callbacks=(), seed=None):
        optimizer = optimizer if optimizer is not None else Adam(lr=1e-3)
        loss = loss if loss is not None else CrossEntropy()
        is_clf = isinstance(loss, CrossEntropy)
        X = np.asarray(X, dtype=self.dtype)
        y = np.asarray(y)
        rng = np.random.default_rng(seed) if seed is not None else get_rng()
        history = {"loss": [], "acc": []}
        if validation_data is not None:
            history.update(val_loss=[], val_acc=[])

        n = len(X)
        for epoch in range(epochs):
            self.train()
            order = rng.permutation(n) if shuffle else np.arange(n)
            total_loss, correct = 0.0, 0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                xb, yb = X[idx], y[idx]
                out = self.forward(xb)
                total_loss += loss(out, yb) * len(idx)
                self.backward(loss.backward())
                optimizer.step(self.parameters())
                if is_clf:
                    correct += int((out.argmax(1) == _labels(yb)).sum())

            logs = {"loss": total_loss / n, "acc": correct / n if is_clf else float("nan")}
            if validation_data is not None:
                logs["val_loss"], logs["val_acc"] = self.evaluate(*validation_data, loss=loss)
            for k, v in logs.items():
                history[k].append(v)
            if verbose:
                print(_format_epoch(epoch, epochs, logs))
            for cb in callbacks:
                cb(self, epoch, logs)

        self.history = history
        return history

    # --- inference ------------------------------------------------------
    def _logits(self, X, batch_size):
        self.eval()
        X = np.asarray(X, dtype=self.dtype)
        return np.concatenate([self.forward(X[i:i + batch_size])
                               for i in range(0, len(X), batch_size)])

    def predict_proba(self, X, batch_size=256):
        return softmax(self._logits(X, batch_size))

    def predict(self, X, batch_size=256):
        return self._logits(X, batch_size).argmax(axis=1)

    def evaluate(self, X, y, loss=None, batch_size=256):
        """Returns (loss, accuracy); accuracy is nan if the loss is not a classification loss."""
        loss = loss if loss is not None else CrossEntropy()
        logits = self._logits(X, batch_size)
        y = np.asarray(y)
        value = loss(logits, y)
        if isinstance(loss, CrossEntropy):
            acc = float((logits.argmax(1) == _labels(y)).mean())
        else:
            acc = float("nan")
        return value, acc

    # --- persistence ----------------------------------------------------
    def save(self, path):
        arrays = {f"{i}.{k}": p for i, layer in enumerate(self.layers)
                  for k, p in layer.params().items()}
        path = str(path)
        np.savez(path if path.endswith(".npz") else path + ".npz", **arrays)

    def load(self, path):
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        with np.load(path) as data:
            for i, layer in enumerate(self.layers):
                for k, p in layer.params().items():
                    key = f"{i}.{k}"
                    if key not in data.files:
                        raise ValueError(f"missing parameter {key!r} in {path}")
                    if data[key].shape != p.shape:
                        raise ValueError(f"shape mismatch for {key!r}: file {data[key].shape}, model {p.shape}")
                    p[...] = data[key]

    def __repr__(self):
        n_params = sum(p.size for p, _ in self.parameters())
        lines = [f"  ({i}) {layer!r}" for i, layer in enumerate(self.layers)]
        return "Sequential(\n" + "\n".join(lines) + f"\n)  # {n_params} parameters"


# --- composite layers (old API) -----------------------------------------
class _Composite(Layer):
    """A layer made of sub-layers. Sequential unrolls it; standalone it delegates."""

    def unroll(self):
        raise NotImplementedError

    def forward(self, x):
        for layer in self.unroll():
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.unroll()):
            grad = layer.backward(grad)
        return grad

    def params(self):
        return {f"{i}.{k}": p for i, l in enumerate(self.unroll()) for k, p in l.params().items()}

    def grads(self):
        return {f"{i}.{k}": g for i, l in enumerate(self.unroll()) for k, g in l.grads().items()}

    def train(self):
        super().train()
        for l in self.unroll():
            l.train()
        return self

    def eval(self):
        super().eval()
        for l in self.unroll():
            l.eval()
        return self


_ACTIVATIONS = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}


class FCLayer(_Composite):
    """Linear + activation. 'softmax' outputs logits: the softmax lives in CrossEntropy."""

    def __init__(self, n_input, n_output, activation="relu"):
        super().__init__()
        self.activation = activation.lower()
        if self.activation != "softmax" and self.activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        init = "he" if self.activation == "relu" else "xavier"
        self.linear = Linear(n_input, n_output, init=init)
        self.act = None if self.activation == "softmax" else _ACTIVATIONS[self.activation]()

    def unroll(self):
        return [self.linear] + ([self.act] if self.act is not None else [])

    def __repr__(self):
        return f"FCLayer({self.linear.W.shape[0]}, {self.linear.W.shape[1]}, activation={self.activation!r})"


class _LegacyModel:
    """Facade compatible with the old API on top of a Sequential."""

    def __init__(self, layers):
        self.layers = list(layers)
        self.model = Sequential(self.layers)

    def forward(self, x):
        return softmax(self.model.forward(np.asarray(x, dtype=self.model.dtype)))

    def train(self, X, Y, epochs, lr=0.01, batch_size=1, optimizer=None,
              validation_data=None, verbose=True, callbacks=()):
        X, Y = self._prep_X(X), _from_legacy(Y)
        if validation_data is not None:
            validation_data = (self._prep_X(validation_data[0]), _from_legacy(validation_data[1]))
        if verbose:
            print(f"Training for {epochs} epochs...")
        return self.model.fit(X, Y, epochs=epochs, batch_size=batch_size,
                              optimizer=optimizer or SGD(lr=lr), loss=CrossEntropy(),
                              validation_data=validation_data, verbose=verbose,
                              callbacks=callbacks)

    def test(self, X, Y, verbose=True):
        _, acc = self.model.evaluate(self._prep_X(X), _from_legacy(Y))
        if verbose:
            print(f"Test accuracy: {acc * 100: .3f}%")
        return acc

    def predict(self, X):
        return self.model.predict(self._prep_X(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep_X(X))

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)

    def _prep_X(self, X):
        return _from_legacy(X)


class NN(_LegacyModel):
    def __init__(self, layers, function="relu"):
        self.structure = list(layers)
        self.input_size = layers[0]
        self.function = function.lower()
        if self.function not in _ACTIVATIONS:
            raise ValueError(f"function must be one of {sorted(_ACTIVATIONS)}, got {function!r}")
        fcs = [FCLayer(layers[i], layers[i + 1], self.function) for i in range(len(layers) - 2)]
        fcs.append(FCLayer(layers[-2], layers[-1], "softmax"))
        super().__init__(fcs)

    def __repr__(self):
        return f"NN structure: {self.structure}\n{self.model!r}"
