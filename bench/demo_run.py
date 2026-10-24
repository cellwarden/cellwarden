"""Live demo: the whole scfast workflow on one 10x .h5, with a timer per step and a UMAP image.

    python bench/demo_run.py --input matrix.h5 --out umap.png

Steps as bench/run_py.py's scfast engines (Seurat's vst HVGs, clipped PCA, k = 20 on 10 PCs,
Leiden at resolution 0.5, UMAP), on the CPU. The input should already be QC'd (bench/prep.py
writes such files).
"""

from __future__ import annotations

import time

T_PROCESS = time.perf_counter()

import argparse  # noqa: E402

import numpy as np  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="umap.png")
    a = p.parse_args()

    import scfast

    times: dict[str, float] = {}

    def step(name, fn):
        t = time.perf_counter()
        out = fn()
        times[name] = time.perf_counter() - t
        print(f"  {name:<10} {times[name]:7.2f} s", flush=True)
        return out

    print(f"scfast on {a.input}")
    X, _, _ = step("read", lambda: scfast.read_10x_h5(a.input))
    pre = step("normalize", lambda: scfast.preprocess(X, n_hvg=2000))
    pcs, _, _ = step("pca", lambda: scfast.pca(pre["X_hvg"], n_comps=30, device="cpu"))
    idx, dist = step("neighbors", lambda: scfast.knn(np.ascontiguousarray(pcs[:, :10]), k=20, method="auto", device="cpu"))
    labels = step("cluster", lambda: scfast.leiden(scfast.snn(idx), resolution=0.5, seed=0, n_iterations=2))
    emb = step("umap", lambda: scfast.umap(idx, dist, pcs[:, :2]))
    n = pcs.shape[0]
    total = time.perf_counter() - T_PROCESS
    print(f"{n:,} cells, {len(np.unique(labels))} clusters: {sum(times.values()):.2f} s of steps, "
          f"{total:.2f} s from process start")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(emb[:, 0], emb[:, 1], c=labels, s=max(0.05, min(2.0, 2e5 / n)), cmap="tab20", linewidths=0)
    ax.set_title(f"{n:,} cells, {len(np.unique(labels))} clusters, {total:.1f} s end to end")
    lo, hi = np.quantile(emb, [0.0005, 0.9995], axis=0)  # a few isolated cells would stretch the view
    pad = 0.03 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
