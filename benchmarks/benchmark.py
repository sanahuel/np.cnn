"""np.cnn benchmarks.

A) Python loops vs im2col convolution (forward + backward).
B) The same small CNN in np.cnn and PyTorch (CPU) on MNIST and Fashion-MNIST.

    python benchmarks/benchmark.py [--only conv|mnist|fashion] [--epochs 5] [--quick]
"""
import argparse
import csv
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BATCH = 64


def _write(name, rows, fields, table):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (RESULTS / f"{name}.md").write_text(table + "\n", encoding="utf-8")


# --- A) loops vs im2col ----------------------------------------------------
def bench_conv(quick):
    from cnn import Conv2d, set_seed
    from cnn.reference import conv2d_forward_loops, conv2d_backward_loops
    set_seed(0)
    configs = [(32, 1, 8, 28, 3, 1), (32, 8, 16, 14, 3, 1), (32, 16, 32, 7, 3, 1)]  # N, Cin, Cout, H, k, p
    reps = 2 if quick else 5
    rows = []
    for N, cin, cout, H, k, p in configs:
        x = np.random.randn(N, cin, H, H).astype(np.float32)
        conv = Conv2d(cin, cout, k, stride=1, padding=p)

        def run_loops():
            out = conv2d_forward_loops(x, conv.W, conv.b, 1, p)
            conv2d_backward_loops(out, x, conv.W, 1, p)

        def run_im2col():
            conv.backward(conv.forward(x))

        times = {}
        for name, fn in (("loops", run_loops), ("im2col", run_im2col)):
            fn()  # warm-up
            best = float("inf")
            for _ in range(reps):
                t0 = time.perf_counter()
                fn()
                best = min(best, time.perf_counter() - t0)
            times[name] = best * 1000
        rows.append({"config": f"N={N}, {cin}->{cout}, {H}x{H}, k={k}, p={p}",
                     "loops_ms": times["loops"], "im2col_ms": times["im2col"],
                     "speedup": times["loops"] / times["im2col"]})
    table = ["| Config | loops (ms) | im2col (ms) | speedup |", "|---|---|---|---|"]
    table += [f"| {r['config']} | {r['loops_ms']:.1f} | {r['im2col_ms']:.1f} | {r['speedup']:.1f}x |" for r in rows]
    table = "\n".join(table)
    print(table)
    if not quick:
        _write("benchmark_conv", rows, ["config", "loops_ms", "im2col_ms", "speedup"], table)


# --- B) CNN on MNIST / Fashion-MNIST ------------------------------------------
def run_npcnn(data, epochs):
    from cnn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU, Dropout, Adam, set_seed
    (Xtr, ytr), (Xte, yte) = data
    set_seed(0)
    model = Sequential([
        Conv2d(1, 8, 3, padding=1), ReLU(), MaxPool2d(2),
        Conv2d(8, 16, 3, padding=1), ReLU(), MaxPool2d(2),
        Flatten(), Linear(16 * 7 * 7, 128), ReLU(), Dropout(0.25), Linear(128, 10)])
    opt, times = Adam(lr=1e-3), []
    for _ in range(epochs):
        t0 = time.perf_counter()
        model.fit(Xtr, ytr, epochs=1, batch_size=BATCH, optimizer=opt, verbose=False)
        times.append(time.perf_counter() - t0)
    _, acc = model.evaluate(Xte, yte)
    return {"framework": "np.cnn", "s_per_epoch": float(np.mean(times)), "test_acc": acc,
            "torch_version": "", "torch_threads": ""}


def run_torch(data, epochs):
    try:
        import torch
        from torch import nn as tnn
    except ImportError:
        print("PyTorch is not installed: skipped (pip install -e '.[bench]')")
        return None
    (Xtr, ytr), (Xte, yte) = data
    torch.manual_seed(0)
    model = tnn.Sequential(
        tnn.Conv2d(1, 8, 3, padding=1), tnn.ReLU(), tnn.MaxPool2d(2),
        tnn.Conv2d(8, 16, 3, padding=1), tnn.ReLU(), tnn.MaxPool2d(2),
        tnn.Flatten(), tnn.Linear(16 * 7 * 7, 128), tnn.ReLU(), tnn.Dropout(0.25), tnn.Linear(128, 10))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = tnn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)
    times = []
    for _ in range(epochs):
        t0 = time.perf_counter()
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        times.append(time.perf_counter() - t0)
    model.eval()
    with torch.no_grad():
        acc = (model(torch.from_numpy(Xte)).argmax(1).numpy() == yte).mean()
    return {"framework": "PyTorch", "s_per_epoch": float(np.mean(times)), "test_acc": float(acc),
            "torch_version": torch.__version__, "torch_threads": torch.get_num_threads()}


def bench_cnn(datasets, epochs, quick):
    from cnn import set_default_dtype
    from cnn.utils import load_mnist
    set_default_dtype(np.float32)
    cpu = platform.processor() or platform.machine()
    rows = []
    for name in datasets:
        (Xtr, ytr), (Xte, yte) = load_mnist(name, flatten=False)
        ep = epochs
        if quick:
            Xtr, ytr, ep = Xtr[:5000], ytr[:5000], 1
        data = ((Xtr, ytr), (Xte, yte))
        for r in (run_npcnn(data, ep), run_torch(data, ep)):
            if r:
                r.update(dataset=name, epochs=ep, batch_size=BATCH, cpu=cpu, numpy_version=np.__version__)
                rows.append(r)
    table = ["| Dataset | Framework | s / epoch | Test accuracy |", "|---|---|---|---|"]
    table += [f"| {r['dataset']} | {r['framework']} | {r['s_per_epoch']:.1f} | {r['test_acc'] * 100:.2f}% |"
              for r in rows]
    table = "\n".join(table)
    print(table)
    if not quick:
        fields = ["dataset", "framework", "epochs", "batch_size", "s_per_epoch", "test_acc", "cpu",
                  "numpy_version", "torch_version", "torch_threads"]
        _write("benchmark_cnn", rows, fields, table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["conv", "mnist", "fashion"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="small subsets and 1 epoch; writes no files")
    args = ap.parse_args()
    if args.only in (None, "conv"):
        bench_conv(args.quick)
    datasets = [d for d in ("mnist", "fashion") if args.only in (None, d)]
    if datasets:
        bench_cnn(datasets, args.epochs, args.quick)


if __name__ == "__main__":
    main()
