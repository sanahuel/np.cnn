"""Convolution with Python loops over the output positions (vectorized over batch and channels).

Reference implementation for the equivalence tests and the benchmark baseline.
"""
import numpy as np


def conv2d_forward_loops(x, W, b, stride=1, padding=0):
    N, C, H, Wd = x.shape
    F, _, k, _ = W.shape
    s, p = stride, padding
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    out_h, out_w = (H + 2 * p - k) // s + 1, (Wd + 2 * p - k) // s + 1
    out = np.empty((N, F, out_h, out_w), dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            patch = xp[:, :, i * s:i * s + k, j * s:j * s + k]                  # (N, C, k, k)
            out[:, :, i, j] = np.tensordot(patch, W, axes=([1, 2, 3], [1, 2, 3])) + b
    return out


def conv2d_backward_loops(grad, x, W, stride=1, padding=0):
    N, C, H, Wd = x.shape
    F, _, k, _ = W.shape
    s, p = stride, padding
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    out_h, out_w = grad.shape[2:]
    dxp, dW = np.zeros_like(xp), np.zeros_like(W)
    db = grad.sum(axis=(0, 2, 3))
    for i in range(out_h):
        for j in range(out_w):
            patch = xp[:, :, i * s:i * s + k, j * s:j * s + k]
            g = grad[:, :, i, j]                                                  # (N, F)
            dW += np.tensordot(g, patch, axes=([0], [0]))                         # (F, C, k, k)
            dxp[:, :, i * s:i * s + k, j * s:j * s + k] += np.tensordot(g, W, axes=([1], [0]))
    return dxp[:, :, p:p + H, p:p + Wd], dW, db
