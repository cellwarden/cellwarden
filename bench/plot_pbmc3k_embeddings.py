#!/usr/bin/env python3
"""PCA + UMAP panels for PBMC3k bench outputs (parity pcs.f32 / labels.txt).

UMAP is recomputed from each engine's saved top-10 PCs (umap-learn, seed=0) for a
comparable 2D view — not the uwot/rapids UMAP times from the timed bench steps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import read_outputs  # noqa: E402


def umap2d(pcs: np.ndarray, seed: int = 0) -> np.ndarray:
    import umap

    x = np.ascontiguousarray(pcs[:, :10], dtype=np.float32)
    return umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.5).fit_transform(x)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=HERE / "results/pbmc3k")
    p.add_argument("--dataset", default="n2638")
    p.add_argument(
        "--engines",
        default="seurat,scanpy-gpu,scfast-gpu",
        help="comma-separated folder names under outputs/",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run = args.run.resolve()
    out_png = args.out or (run / "plots" / f"pca_umap_{args.dataset}.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    fig, axes = plt.subplots(len(engines), 2, figsize=(10, 3.2 * len(engines)), squeeze=False)

    for row, eng in enumerate(engines):
        odir = run / "outputs" / eng / args.dataset
        if not (odir / "pcs.f32").is_file():
            raise SystemExit(f"missing outputs: {odir}")
        _bc, pcs, _knn, labels = read_outputs(odir)
        lab = np.array(labels, dtype=str)
        u = umap2d(pcs, seed=args.seed)

        for col, (xy, xlab, ylab) in enumerate(
            [
                (pcs[:, :2], "PC1", "PC2"),
                (u, "UMAP1", "UMAP2"),
            ]
        ):
            ax = axes[row, col]
            for cat in np.unique(lab):
                m = lab == cat
                ax.scatter(xy[m, 0], xy[m, 1], s=4, alpha=0.75, label=cat, rasterized=True)
            ax.set_title(f"{eng} — {xlab} vs {ylab}")
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            ax.set_aspect("equal", adjustable="datalim")
        axes[row, 1].legend(
            markerscale=3, fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False
        )

    fig.suptitle(
        f"PBMC 3k ({args.dataset}): cluster colours from each engine's labels\n"
        "UMAP: umap-learn on that engine's saved PCs (dims 1–10), seed=0",
        fontsize=10,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
