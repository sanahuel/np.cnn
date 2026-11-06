import numpy as np


class Loss:
    def __call__(self, pred, target):
        return self.forward(pred, target)


class CrossEntropy(Loss):
    """Softmax + cross-entropy on logits. target: integer labels (N,) or one-hot (N, C)."""

    def forward(self, logits, target):
        z = logits - logits.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        if target.ndim == 1:
            target = np.eye(logits.shape[1], dtype=logits.dtype)[target]
        self.p = np.exp(log_p)
        self.y = target.astype(logits.dtype, copy=False)
        return float(-(self.y * log_p).sum() / logits.shape[0])

    def backward(self):
        return (self.p - self.y) / self.p.shape[0]


class MSE(Loss):
    def forward(self, pred, target):
        self.diff = pred - target.reshape(pred.shape)
        return float(np.mean(self.diff ** 2))

    def backward(self):
        return 2.0 * self.diff / self.diff.size
