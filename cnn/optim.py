import numpy as np


class Optimizer:
    """step(params) takes a list of (param, grad) pairs and updates params in place."""

    def __init__(self, lr):
        self.lr = lr

    def step(self, params):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(lr)
        self.momentum, self.weight_decay = momentum, weight_decay
        self.v = None

    def step(self, params):
        if self.v is None:
            self.v = [np.zeros_like(p) for p, _ in params]
        for i, (p, g) in enumerate(params):
            if self.weight_decay:
                g = g + self.weight_decay * p
            if self.momentum:
                self.v[i] = self.momentum * self.v[i] + g
                g = self.v[i]
            p -= self.lr * g


class Adam(Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(lr)
        self.b1, self.b2 = betas
        self.eps, self.weight_decay = eps, weight_decay
        self.m = self.v = None
        self.t = 0

    def step(self, params):
        if self.m is None:
            self.m = [np.zeros_like(p) for p, _ in params]
            self.v = [np.zeros_like(p) for p, _ in params]
        self.t += 1
        for i, (p, g) in enumerate(params):
            if self.weight_decay:
                g = g + self.weight_decay * p
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
