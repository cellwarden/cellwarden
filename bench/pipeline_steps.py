"""Run the Seurat-style workflow on one dataset and save every step's output for inspection.

    python bench/pipeline_steps.py --input sample_filtered_feature_bc_matrix.h5 --out results/run

Works on a 10x folder, 10x .h5 (only Gene Expression features are used, so hashing / CITE-seq
files are fine) or .h5ad with raw counts. Each step writes its own folder of tables (CSV) and
plots (PNG), plus a timing; `summary.md` links them. Engines: `scfast-cpu` (scfast PCA, kNN, SNN
and Leiden) or `scanpy` (reference).

Steps: 01_qc, 02_normalize, 03_hvg, 04_pca, 05_neighbors, 06_cluster, 07_umap.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from h5io import read_counts  # noqa: E402
from prep import make_unique  # noqa: E402

MARKERS = {"T": ["CD3E", "IL7R"], "CD8 T": ["CD8A"], "NK": ["NKG7", "GNLY"], "B": ["MS4A1", "CD79A"],
           "CD14 mono": ["CD14", "LYZ"], "CD16 mono": ["FCGR3A"], "DC": ["FCER1A"], "Platelet": ["PPBP"]}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--engine", choices=["scfast-cpu", "scanpy"], default="scfast-cpu")
    p.add_argument("--min-cells", type=int, default=3)
    p.add_argument("--min-genes", type=int, default=200)
    p.add_argument("--max-genes", type=float, default=float("inf"), help="upper n_genes cut-off (off by default: GEM-X cells detect far more than the PBMC 3k limit of 2,500)")
    p.add_argument("--max-mt", type=float, default=20.0, help="percent mitochondrial cut-off")
    p.add_argument("--mt-prefix", default="MT-")
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--dims", type=int, default=10, help="leading PCs for neighbours and UMAP")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--resolution", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--umap-epochs", type=int, default=None, help="default: scanpy's choice (200 for large data)")
    p.add_argument("--umap-init", default="spectral", choices=["spectral", "random", "pca"],
                   help="scanpy engine only; scfast-cpu starts UMAP from the first two PCs")
    p.add_argument("--skip-umap", action="store_true")
    return p.parse_args(argv)


class Steps:
    def __init__(self, out: Path):
        self.out, self.times = out, {}
        out.mkdir(parents=True, exist_ok=True)

    def folder(self, name):
        d = self.out / name
        d.mkdir(exist_ok=True)
        return d

    def run(self, name, fn):
        t0 = time.perf_counter()
        res = fn(self.folder(name))
        self.times[name] = round(time.perf_counter() - t0, 3)
        print(f"{name}: {self.times[name]:.2f} s", flush=True)
        return res


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    import scanpy as sc

    a = parse(argv)
    S = Steps(Path(a.out))
    scfast = None
    if a.engine == "scfast-cpu":
        import scfast

    # ------------------------------------------------------------------ read + 01 QC
    t0 = time.perf_counter()
    counts, barcodes, ids, names = read_counts(a.input)
    read_s = round(time.perf_counter() - t0, 3)
    print(f"read: {counts.shape[0]} barcodes x {counts.shape[1]} genes in {read_s} s", flush=True)

    def qc(d):
        n0, g0 = counts.shape
        keep_g = np.flatnonzero(np.diff((counts > 0).tocsc().indptr) >= a.min_cells)
        c = counts[:, keep_g]
        nm = make_unique([names[i] for i in keep_g])
        gid = [ids[i] for i in keep_g]
        n_genes = np.diff(c.indptr)
        total = np.asarray(c.sum(axis=1)).ravel()
        mt = np.array([n.upper().startswith(a.mt_prefix.upper()) for n in nm])
        mtc = np.asarray(c[:, np.flatnonzero(mt)].sum(axis=1)).ravel() if mt.any() else np.zeros(len(total))
        pct = np.divide(100 * mtc, total, out=np.zeros(len(total)), where=total > 0)
        keep = (n_genes > a.min_genes) & (n_genes < a.max_genes) & (pct < a.max_mt)
        df = pd.DataFrame({"barcode": barcodes, "n_genes": n_genes, "total_counts": total, "pct_mt": pct,
                           "passed_qc": keep})
        df.to_csv(d / "qc_metrics_per_cell.csv", index=False)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for axi, col in zip(ax, ["n_genes", "total_counts", "pct_mt"]):
            axi.hist(df[col], bins=80, color="#4c78a8")
            axi.set_title(col)
        for v, col in [(a.min_genes, 0), (a.max_genes, 0)]:
            ax[0].axvline(v, color="crimson", ls="--")
        ax[2].axvline(a.max_mt, color="crimson", ls="--")
        savefig(fig, d / "qc_distributions.png")
        summary = {"barcodes_in": int(n0), "genes_in": int(g0), "genes_kept": len(gid),
                   "mito_genes": int(mt.sum()), "cells_passed": int(keep.sum()),
                   "cells_removed": int((~keep).sum()),
                   "thresholds": {"min_genes": a.min_genes, "max_genes": a.max_genes, "max_mt": a.max_mt}}
        (d / "qc_summary.json").write_text(json.dumps(summary, indent=2))
        print("  ", summary)
        return c[np.flatnonzero(keep)], [barcodes[i] for i in np.flatnonzero(keep)], gid, nm

    c, bcs, gid, nm = S.run("01_qc", qc)
    import anndata as ad

    adata = ad.AnnData(c.astype(np.float32), obs=pd.DataFrame(index=bcs), var=pd.DataFrame(index=nm))
    adata.var["gene_id"] = gid
    adata.layers["counts"] = adata.X.copy()
    rng_note = {"cells": adata.n_obs, "genes": adata.n_vars}

    # ------------------------------------------------------------------ 02 normalise
    def normalize(d):
        lib0 = np.asarray(adata.X.sum(axis=1)).ravel()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        pd.DataFrame({"barcode": adata.obs_names, "library_size_raw": lib0}).to_csv(d / "library_sizes.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        ax[0].hist(lib0, bins=80, color="#4c78a8")
        ax[0].set_title("raw library size")
        vals = adata.X.data if sp.issparse(adata.X) else np.ravel(adata.X)
        ax[1].hist(vals, bins=80, color="#f58518")
        ax[1].set_title("log-normalised non-zero values")
        savefig(fig, d / "normalization.png")
        (d / "summary.json").write_text(json.dumps({"target_sum": 1e4, "median_library_size": float(np.median(lib0))}, indent=2))

    S.run("02_normalize", normalize)

    # ------------------------------------------------------------------ 03 HVG
    def hvg(d):
        sc.pp.highly_variable_genes(adata, n_top_genes=a.n_hvg, flavor="seurat_v3", layer="counts")
        v = adata.var.copy()
        v.index.name = "gene"
        v.sort_values("highly_variable_rank").to_csv(d / "hvg_table.csv")
        v[v.highly_variable].sort_values("highly_variable_rank")[[]].to_csv(d / "hvg_selected.csv")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        h = v.highly_variable
        ax.scatter(v.means[~h], v.variances_norm[~h], s=2, c="#9aa5b1", label="other")
        ax.scatter(v.means[h], v.variances_norm[h], s=3, c="crimson", label=f"top {a.n_hvg} HVGs")
        ax.set_xscale("log")
        ax.set_xlabel("mean")
        ax.set_ylabel("standardised variance")
        ax.legend(markerscale=3)
        savefig(fig, d / "hvg_scatter.png")

    S.run("03_hvg", hvg)
    hv = adata[:, adata.var.highly_variable].copy()

    # ------------------------------------------------------------------ 04 PCA
    def pca(d):
        if a.engine == "scfast-cpu":
            emb, load, sdev = scfast.pca(sp.csr_matrix(hv.X), n_comps=a.n_pcs, seed=a.seed)
            hv.obsm["X_pca"], hv.varm["PCs"] = emb, load
            var = sdev.astype("float64") ** 2
        else:
            sc.pp.scale(hv, max_value=10)
            sc.pp.pca(hv, n_comps=a.n_pcs, random_state=a.seed)
            emb, load = hv.obsm["X_pca"], hv.varm["PCs"]
            var = hv.uns["pca"]["variance"]
        cols = [f"PC{i + 1}" for i in range(a.n_pcs)]
        pd.DataFrame(emb, index=hv.obs_names, columns=cols).to_csv(d / "pca_embeddings.csv")
        pd.DataFrame(load, index=hv.var_names, columns=cols).to_csv(d / "pca_loadings.csv")
        pd.DataFrame({"pc": cols, "variance": var, "sdev": np.sqrt(var)}).to_csv(d / "pca_variance.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(range(1, a.n_pcs + 1), np.sqrt(var), "o-")
        ax[0].axvline(a.dims + 0.5, color="crimson", ls="--")
        ax[0].set_xlabel("PC")
        ax[0].set_ylabel("standard deviation")
        ax[0].set_title(f"elbow (dashed: {a.dims} PCs used downstream)")
        ax[1].scatter(emb[:, 0], emb[:, 1], s=0.5, alpha=0.4, c="#4c78a8", rasterized=True)
        ax[1].set_xlabel("PC1")
        ax[1].set_ylabel("PC2")
        savefig(fig, d / "pca_plots.png")

    S.run("04_pca", pca)

    # ------------------------------------------------------------------ 05 neighbours
    def neighbors(d):
        x = np.ascontiguousarray(hv.obsm["X_pca"][:, : a.dims])
        if a.engine == "scfast-cpu":
            idx, dist = scfast.knn(x, k=a.k, method="exact")
            snn = scfast.snn(idx)
            hv.obsp["snn"] = snn
            hv.obsm["knn_indices"], hv.obsm["knn_distances"] = idx, dist
        else:
            sc.pp.neighbors(hv, n_neighbors=a.k, n_pcs=a.dims, random_state=a.seed)
            idx = None
            dist = None
            snn = hv.obsp["connectivities"]
        if idx is not None:
            np.savez_compressed(d / "knn.npz", indices=idx.astype(np.int32), distances=dist.astype(np.float32))
        deg = np.diff(snn.tocsr().indptr)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        ax[0].hist(deg, bins=60, color="#4c78a8")
        ax[0].set_title("SNN graph degree")
        if dist is not None:
            ax[1].hist(dist[:, 1:].ravel(), bins=80, color="#f58518")
            ax[1].set_title("kNN distances")
        savefig(fig, d / "graph.png")
        (d / "summary.json").write_text(json.dumps({"k": a.k, "dims": a.dims, "graph_edges": int(snn.nnz),
                                                     "mean_degree": float(deg.mean())}, indent=2))

    S.run("05_neighbors", neighbors)

    # ------------------------------------------------------------------ 06 clusters
    def cluster(d):
        if a.engine == "scfast-cpu":
            lab = scfast.leiden(hv.obsp["snn"], resolution=a.resolution, seed=a.seed, n_iterations=2)
            hv.obs["cluster"] = pd.Categorical(lab.astype(str))
        else:
            sc.tl.leiden(hv, resolution=a.resolution, random_state=a.seed, flavor="igraph", n_iterations=2,
                         directed=False, key_added="cluster")
        sizes = hv.obs["cluster"].value_counts().sort_index(key=lambda s: s.astype(int))
        hv.obs[["cluster"]].rename_axis("barcode").to_csv(d / "clusters.csv")
        sizes.rename_axis("cluster").rename("cells").to_csv(d / "cluster_sizes.csv")
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.bar(sizes.index.astype(str), sizes.values, color="#4c78a8")
        ax.set_xlabel("cluster")
        ax.set_ylabel("cells")
        savefig(fig, d / "cluster_sizes.png")
        print("  clusters:", len(sizes))

    S.run("06_cluster", cluster)

    # ------------------------------------------------------------------ 07 UMAP
    if not a.skip_umap:
        def umap(d):
            if a.engine == "scfast-cpu":
                hv.obsm["X_umap"] = scfast.umap(hv.obsm["knn_indices"], hv.obsm["knn_distances"], hv.obsm["X_pca"][:, :2],
                                                n_epochs=a.umap_epochs, seed=a.seed)
            else:
                kw = {"init_pos": a.umap_init}
                if a.umap_epochs:
                    kw["maxiter"] = a.umap_epochs
                sc.tl.umap(hv, random_state=a.seed, **kw)
            u = hv.obsm["X_umap"]
            pd.DataFrame(u, index=hv.obs_names, columns=["UMAP1", "UMAP2"]).rename_axis("barcode").to_csv(d / "umap_coordinates.csv")
            cl = hv.obs["cluster"].astype(int).values
            fig, ax = plt.subplots(figsize=(7, 6))
            sc_ = ax.scatter(u[:, 0], u[:, 1], c=cl, s=0.6, cmap="tab20", rasterized=True)
            for k in np.unique(cl):
                m = np.median(u[cl == k], axis=0)
                ax.text(m[0], m[1], str(k), fontsize=9, weight="bold")
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")
            ax.set_title("clusters")
            savefig(fig, d / "umap_clusters.png")
            present = [(t, g) for t, gs in MARKERS.items() for g in gs if g in adata.var_names]
            if present:
                n = len(present)
                cols = min(4, n)
                rows = int(np.ceil(n / cols))
                fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows), squeeze=False)
                for axi in axs.ravel():
                    axi.axis("off")
                for axi, (t, g) in zip(axs.ravel(), present):
                    e = adata[hv.obs_names, g].X
                    e = e.toarray().ravel() if sp.issparse(e) else np.ravel(e)
                    o = np.argsort(e)
                    axi.axis("on")
                    axi.scatter(u[o, 0], u[o, 1], c=e[o], s=0.4, cmap="viridis", rasterized=True)
                    axi.set_title(f"{g} ({t})", fontsize=9)
                    axi.set_xticks([])
                    axi.set_yticks([])
                savefig(fig, d / "umap_markers.png")

        S.run("07_umap", umap)

    # ------------------------------------------------------------------ cluster x marker table + summary
    markers = [g for gs in MARKERS.values() for g in gs if g in adata.var_names]
    if markers:
        e = adata[hv.obs_names, markers].X
        e = e.toarray() if sp.issparse(e) else np.asarray(e)
        tbl = pd.DataFrame(e, columns=markers).groupby(hv.obs["cluster"].values).mean()
        tbl.index.name = "cluster"
        tbl.to_csv(Path(a.out) / "cluster_marker_means.csv")
    hv.write_h5ad(Path(a.out) / "final.h5ad")
    times = {"read": read_s, **S.times, "total_compute": round(sum(S.times.values()), 3)}
    (Path(a.out) / "timings.json").write_text(json.dumps({"engine": a.engine, **rng_note, "seconds": times}, indent=2))
    lines = [f"# Pipeline outputs ({a.engine})", "", f"Input: `{a.input}`", "",
             "| Step | Seconds | Output folder |", "|---|---:|---|"]
    lines += [f"| {k} | {v} | `{k}/` |" for k, v in S.times.items()]
    lines += ["", f"Read: {read_s} s. Total compute: {times['total_compute']} s.", "",
              "Also: `cluster_marker_means.csv` (mean log-normalised marker expression per cluster, for annotation), "
              "`final.h5ad` (all results in one AnnData), `timings.json`."]
    (Path(a.out) / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
