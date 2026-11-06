import numpy as np

_DEFAULT_DTYPE = np.float32
_RNG = np.random.default_rng()


def set_default_dtype(dtype):
    """Set the dtype used to create new parameters."""
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = np.dtype(dtype).type


def get_default_dtype():
    return _DEFAULT_DTYPE


def set_seed(seed):
    """Seed weight initialisation, Dropout and shuffling."""
    global _RNG
    _RNG = np.random.default_rng(seed)


def get_rng():
    return _RNG  # always read through get_rng(): set_seed replaces the object


class Layer:
    """Base layer. forward(x) -> y, backward(grad_out) -> grad_in; shapes are (batch, features)."""

    def __init__(self):
        self.training = True

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def params(self):
        return {}

    def grads(self):
        return {}

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def __repr__(self):
        return f"{type(self).__name__}()"


class Linear(Layer):
    """y = x @ W + b. x: (N, in) -> (N, out)."""

    def __init__(self, in_features, out_features, init="he"):
        super().__init__()
        if init == "he":
            std = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
        else:
            raise ValueError(f"unknown init {init!r}")
        dt = get_default_dtype()
        self.W = (get_rng().standard_normal((in_features, out_features)) * std).astype(dt)
        self.b = np.zeros(out_features, dtype=dt)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}

    def __repr__(self):
        return f"Linear({self.W.shape[0]}, {self.W.shape[1]})"


class ReLU(Layer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class Sigmoid(Layer):
    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1.0 - self.out)


class Tanh(Layer):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1.0 - self.out ** 2)


class Softmax(Layer):
    """Row-wise softmax. Meant for inference: models train on logits with CrossEntropy."""

    def forward(self, x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        self.out = e / e.sum(axis=1, keepdims=True)
        return self.out

    def backward(self, grad):
        s = self.out
        return s * (grad - (grad * s).sum(axis=1, keepdims=True))


class Dropout(Layer):
    """Inverted dropout: active only in train mode."""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            self.mask = None
            return x
        self.mask = ((get_rng().random(x.shape) >= self.p) / (1.0 - self.p)).astype(x.dtype)
        return x * self.mask

    def backward(self, grad):
        return grad if self.mask is None else grad * self.mask

    def __repr__(self):
        return f"Dropout(p={self.p})"
