import numpy as np


def numerical_grad(f, arr, eps=1e-6):
    """Gradiente numérico de f() (escalar) respecto a `arr`, que se perturba en el sitio."""
    grad = np.zeros_like(arr, dtype=np.float64)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = arr[idx]
        arr[idx] = old + eps
        f_plus = f()
        arr[idx] = old - eps
        f_minus = f()
        arr[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def rel_error(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12)


def check_layer(layer, x, seed=0, tol=1e-5):
    rng = np.random.default_rng(seed)
    out = layer.forward(x)
    R = rng.standard_normal(out.shape)            # L = sum(out * R)  =>  dL/dout = R
    dx = layer.backward(R)
    analytic = {"x": dx, **{k: g.copy() for k, g in layer.grads().items()}}

    f = lambda: float(np.sum(layer.forward(x) * R))
    numeric = {"x": numerical_grad(f, x)}
    for k, p in layer.params().items():
        numeric[k] = numerical_grad(f, p)

    for k in analytic:
        err = rel_error(analytic[k], numeric[k])
        assert err < tol, f"{type(layer).__name__}.{k}: rel_error={err:.2e}"
