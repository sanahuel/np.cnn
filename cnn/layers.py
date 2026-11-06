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


# =============================================================================
# Solo np.cnn: todo lo que hay por encima de esta línea es idéntico a np.nn
# =============================================================================


def _im2col(xp, k, s, out_h, out_w):
    """xp: (N, C, Hp, Wp), already padded -> (N*out_h*out_w, C*k*k)."""
    N, C = xp.shape[:2]
    cols = np.empty((N, C, k, k, out_h, out_w), dtype=xp.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j] = xp[:, :, i:i + s * out_h:s, j:j + s * out_w:s]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, C * k * k)


def _col2im(dcols, xp_shape, k, s, out_h, out_w):
    """Inverse of _im2col (accumulating overlaps): (N*out_h*out_w, C*k*k) -> (N, C, Hp, Wp)."""
    N, C = xp_shape[:2]
    dcols = dcols.reshape(N, out_h, out_w, C, k, k).transpose(0, 3, 4, 5, 1, 2)
    dxp = np.zeros(xp_shape, dtype=dcols.dtype)
    for i in range(k):
        for j in range(k):
            dxp[:, :, i:i + s * out_h:s, j:j + s * out_w:s] += dcols[:, :, i, j]
    return dxp


class Conv2d(Layer):
    """2D convolution via im2col. x: (N, C_in, H, W) -> (N, C_out, H_out, W_out)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.k, self.stride, self.padding = kernel_size, stride, padding
        fan_in = in_channels * kernel_size * kernel_size
        dt = get_default_dtype()
        shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.W = (get_rng().standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(dt)   # He
        self.b = np.zeros(out_channels, dtype=dt)
        self.dW, self.db = np.zeros_like(self.W), np.zeros_like(self.b)

    def output_shape(self, h, w):
        k, s, p = self.k, self.stride, self.padding
        return (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1

    def forward(self, x):
        N, C, H, W = x.shape
        p = self.padding
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p else x
        out_h, out_w = self.output_shape(H, W)
        cols = _im2col(xp, self.k, self.stride, out_h, out_w)
        out = cols @ self.W.reshape(self.out_channels, -1).T + self.b      # (N*oh*ow, F)
        self._cache = (x.shape, xp.shape, cols, out_h, out_w)
        return out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

    def backward(self, grad):
        x_shape, xp_shape, cols, out_h, out_w = self._cache
        F = self.out_channels
        g = grad.transpose(0, 2, 3, 1).reshape(-1, F)                       # (N*oh*ow, F)
        self.dW = (g.T @ cols).reshape(self.W.shape)
        self.db = g.sum(axis=0)
        dcols = g @ self.W.reshape(F, -1)
        dxp = _col2im(dcols, xp_shape, self.k, self.stride, out_h, out_w)
        p = self.padding
        H, W = x_shape[2:]
        return dxp[:, :, p:p + H, p:p + W] if p else dxp                     # remove the padding

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}

    def __repr__(self):
        return (f"Conv2d({self.in_channels}, {self.out_channels}, kernel_size={self.k}, "
                f"stride={self.stride}, padding={self.padding})")


class MaxPool2d(Layer):
    """Max pooling. The gradient of each window goes to the position of its maximum."""

    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.k = kernel_size
        self.stride = stride or kernel_size

    def forward(self, x):
        N, C, H, W = x.shape
        k, s = self.k, self.stride
        out_h, out_w = (H - k) // s + 1, (W - k) // s + 1
        win = np.empty((N, C, out_h, out_w, k * k), dtype=x.dtype)
        for i in range(k):
            for j in range(k):
                win[..., i * k + j] = x[:, :, i:i + s * out_h:s, j:j + s * out_w:s]
        self._argmax = win.argmax(axis=-1)
        self._cache = (x.shape, out_h, out_w)
        return win.max(axis=-1)

    def backward(self, grad):
        x_shape, out_h, out_w = self._cache
        k, s = self.k, self.stride
        dx = np.zeros(x_shape, dtype=grad.dtype)
        for i in range(k):
            for j in range(k):
                mask = self._argmax == (i * k + j)
                dx[:, :, i:i + s * out_h:s, j:j + s * out_w:s] += grad * mask
        return dx

    def __repr__(self):
        return f"MaxPool2d({self.k}, stride={self.stride})"


class Flatten(Layer):
    """(N, C, H, W) -> (N, C*H*W), respecting the batch dimension."""

    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self._shape)
