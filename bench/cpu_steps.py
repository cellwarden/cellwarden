"""Time scfast's CPU PCA, kNN, SNN and Leiden on one prepared dataset; report kNN overlap and ARI.

    python bench/cpu_steps.py --data /tmp/scscale/prep500k/n500000 --methods exact,hnsw --reps 3

The HVG-scaled input (CSR, 2000 genes) is cached next to the data as hvg.npz, so only the first
run pays for read + normalise + HVG. kNN overlap is the mean fraction of each cell's k - 1
neighbours shared with the reference method (the first in --methods); ARI is against the
reference clustering and against the planted labels (cells.tsv, when it has cell_type).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def hvg_matrix(data: Path, n_hvg: int) -> sp.csr_matrix:
    cache = data / f"hvg{n_hvg}.npz"
    if cache.exists():
        return sp.load_npz(cache).tocsr()
    import scanpy as sc

    a = sc.read_10x_h5(str(data / "matrix.h5"))
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor="seurat_v3", layer="counts")
    x = sp.csr_matrix(a[:, a.var.highly_variable].X, dtype=np.float32)
    sp.save_npz(cache, x, compressed=False)
    return x


def timed(fn, reps):
    ts, out = [], None
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return out, float(np.median(ts)), [round(v, 3) for v in ts]


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.sort(a[:, 1:], axis=1), np.sort(b[:, 1:], axis=1)
    hits = sum(np.isin(a[i], b[i], assume_unique=True).sum() for i in range(0, len(a), max(1, len(a) // 20000)))
    rows = len(range(0, len(a), max(1, len(a) // 20000)))
    return hits / (rows * a.shape[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--methods", default="exact,hnsw")
    p.add_argument("--steps", default="pca,knn,snn,leiden")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--dims", type=int, default=10)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--resolution", type=float, default=0.5)
    p.add_argument("--save-labels", help="folder for each method's Leiden labels (labels_<method>.npy)")
    a = p.parse_args()

    import scfast
    from sklearn.metrics import adjusted_rand_score as ari

    data = Path(a.data)
    x = hvg_matrix(data, a.n_hvg)
    steps = a.steps.split(",")
    res = {"cells": x.shape[0], "data": str(data)}
    pcs_cache = data / f"pcs{a.n_pcs}.npy"
    if "pca" in steps or not pcs_cache.exists():
        (emb, _, _), res["pca_s"], res["pca_all"] = timed(lambda: scfast.pca(x, n_comps=a.n_pcs, seed=0), a.reps)
        np.save(pcs_cache, emb)
    pcs = np.ascontiguousarray(np.load(pcs_cache)[:, : a.dims], dtype=np.float32)

    truth = None
    cells = data / "cells.tsv"
    if cells.exists():
        import pandas as pd

        t = pd.read_csv(cells, sep="\t")
        if "cell_type" in t:
            truth = t["cell_type"].to_numpy()

    ref_idx = ref_lab = None
    for m in a.methods.split(","):
        r = {}
        method, _, opt = m.partition(":")
        kw = dict(kv.split("=") for kv in opt.split(";") if kv)
        kw = {k: int(v) for k, v in kw.items()}
        (idx, _), r["knn_s"], r["knn_all"] = timed(lambda: scfast.knn(pcs, k=a.k, method=method, device="cpu", **kw), a.reps)
        if ref_idx is None:
            ref_idx = idx
        else:
            r["overlap_vs_ref"] = round(overlap(idx, ref_idx), 4)
        if "snn" in steps or "leiden" in steps:
            g, r["snn_s"], r["snn_all"] = timed(lambda: scfast.snn(idx), a.reps)
        if "leiden" in steps:
            lab, r["leiden_s"], r["leiden_all"] = timed(
                lambda: scfast.leiden(g, resolution=a.resolution, seed=0, n_iterations=2, device="cpu"), a.reps)
            r["n_clusters"] = int(lab.max()) + 1
            if a.save_labels:
                Path(a.save_labels).mkdir(parents=True, exist_ok=True)
                np.save(Path(a.save_labels) / f"labels_{m.replace(':', '_').replace(';', '_')}.npy", lab)
            if ref_lab is None:
                ref_lab = lab
            else:
                r["ari_vs_ref"] = round(ari(ref_lab, lab), 4)
            if truth is not None:
                r["ari_vs_truth"] = round(ari(truth, lab), 4)
        res[m] = r
        print(m, json.dumps(r), flush=True)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
