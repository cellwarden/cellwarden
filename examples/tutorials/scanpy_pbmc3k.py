# Scanpy starter: PBMC 3k, mirrors seurat_pbmc3k.R
import os, scanpy as sc, pandas as pd
os.chdir(os.path.expanduser("~/seurat-workspace"))
os.makedirs("plots_scanpy", exist_ok=True)
sc.settings.figdir = "plots_scanpy"; sc.settings.verbosity = 1
sc.set_figure_params(dpi=100, dpi_save=150, frameon=False)

# 1. Data (reuses the 10x download from the Seurat run)
adata = sc.read_10x_mtx("filtered_gene_bc_matrices/hg19/", var_names="gene_symbols", cache=True)
adata.var_names_make_unique()
sc.pp.filter_cells(adata, min_genes=200); sc.pp.filter_genes(adata, min_cells=3)

# 2. QC (same thresholds as Seurat)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
sc.pl.violin(adata, ["n_genes_by_counts","total_counts","pct_counts_mt"], jitter=0.4, multi_panel=True, save="_01_qc.png", show=False)
adata = adata[(adata.obs.n_genes_by_counts < 2500) & (adata.obs.pct_counts_mt < 5)].copy()

# 3. Normalise, HVGs, scale, PCA
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata); adata.raw = adata
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
adata = adata[:, adata.var.highly_variable].copy()   # full genes stay in adata.raw
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)

# 4. Neighbours, Leiden, UMAP
sc.pp.neighbors(adata, n_pcs=10, n_neighbors=10)
sc.tl.leiden(adata, resolution=0.5, flavor="igraph", n_iterations=2, key_added="leiden")
sc.tl.umap(adata)

# 5. Markers
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
sc.get.rank_genes_groups_df(adata, group=None).to_csv("scanpy_cluster_markers.csv", index=False)

# 6. Annotation by z-scored canonical marker means
canon = {"Naive CD4 T":["CCR7","LEF1"], "Memory CD4 T":["S100A4","IL7R","CD40LG"], "CD14 Mono":["CD14","S100A8"],
         "B":["MS4A1","CD79A"], "CD8 T":["CD8A","CD8B"], "FCGR3A Mono":["FCGR3A","MS4A7"],
         "NK":["GNLY","GZMB"], "DC":["FCER1A","CLEC10A"], "Platelet":["PPBP","PF4"]}
genes = sorted({g for v in canon.values() for g in v})
expr = sc.get.obs_df(adata, keys=genes, use_raw=True); expr["leiden"] = adata.obs["leiden"].values
avg = expr.groupby("leiden", observed=True).mean()
z = (avg - avg.mean()) / avg.std()
score = pd.DataFrame({ct: z[g].mean(axis=1) for ct, g in canon.items()})
adata.obs["cell_type"] = adata.obs["leiden"].map(score.idxmax(axis=1)).astype("category")

# 7. Plots
sc.pl.umap(adata, color="leiden", legend_loc="on data", save="_02_clusters.png", show=False)
sc.pl.umap(adata, color="cell_type", legend_loc="on data", legend_fontsize=8, save="_03_celltypes.png", show=False)
sc.pl.umap(adata, color=["MS4A1","GNLY","CD3E","CD14","FCER1A","FCGR3A","LYZ","PPBP","CD8A"], ncols=3, save="_04_features.png", show=False)
sc.pl.violin(adata, ["MS4A1","CD79A","NKG7","PPBP"], groupby="cell_type", rotation=60, save="_05_violin.png", show=False)
sc.pl.dotplot(adata, canon, groupby="cell_type", standard_scale="var", save="_06_dotplot.png", show=False)
sc.pl.rank_genes_groups_heatmap(adata, n_genes=3, groupby="leiden", show_gene_labels=True, save="_07_heatmap.png", show=False)

adata.write_h5ad("pbmc3k_scanpy.h5ad")
print("CELLS", adata.n_obs, "CLUSTERS", adata.obs.leiden.nunique())
print(pd.crosstab(adata.obs.leiden, adata.obs.cell_type))
