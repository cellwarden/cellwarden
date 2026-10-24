# Seurat v5 starter: PBMC 3k (10x Genomics) end-to-end
suppressPackageStartupMessages({ library(Seurat); library(ggplot2); library(patchwork); library(dplyr) })


if (!dir.exists("filtered_gene_bc_matrices")) {
  dir.create("~/seurat-workspace")
}

setwd("~/seurat-workspace"); dir.create("plots", showWarnings = FALSE)

# 1. Data
if (!dir.exists("filtered_gene_bc_matrices")) {
  download.file("https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz", "pbmc3k.tar.gz", mode = "wb")
  untar("pbmc3k.tar.gz")
}
counts <- Read10X("filtered_gene_bc_matrices/hg19/")
pbmc <- CreateSeuratObject(counts, project = "pbmc3k", min.cells = 3, min.features = 200)

# 2. QC
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
ggsave("plots/01_qc_violin.png", VlnPlot(pbmc, c("nFeature_RNA","nCount_RNA","percent.mt"), ncol = 3, pt.size = 0.1), width = 10, height = 4)
pbmc <- subset(pbmc, nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)

# 3. Normalise, variable features, scale, PCA
pbmc <- NormalizeData(pbmc) |> FindVariableFeatures(nfeatures = 2000) |> ScaleData() |> RunPCA(verbose = FALSE)
ggsave("plots/02_elbow.png", ElbowPlot(pbmc), width = 5, height = 4)

# 4. Cluster and UMAP
pbmc <- FindNeighbors(pbmc, dims = 1:10) |> FindClusters(resolution = 0.5) |> RunUMAP(dims = 1:10)

# 5. Markers (uses presto automatically if installed)
markers <- FindAllMarkers(pbmc, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
write.csv(markers, "pbmc3k_cluster_markers.csv", row.names = FALSE)

# 6. Simple annotation by canonical markers
canon <- list(`Naive CD4 T`=c("CCR7","LEF1"), `Memory CD4 T`=c("S100A4","IL7R","CD40LG"), `CD14 Mono`=c("CD14","S100A8"),
              B=c("MS4A1","CD79A"), `CD8 T`=c("CD8A","CD8B"), `FCGR3A Mono`=c("FCGR3A","MS4A7"),
              NK=c("GNLY","GZMB"), DC=c("FCER1A","CLEC10A"), Platelet=c("PPBP","PF4"))
avg <- AverageExpression(pbmc, features = unique(unlist(canon)), layer = "data")$RNA
z <- t(scale(t(as.matrix(log1p(avg)))))   # z-score each gene across clusters
score <- sapply(canon, function(g) colMeans(z[g, , drop = FALSE]))
labels <- setNames(colnames(score)[apply(score, 1, which.max)], levels(pbmc))
pbmc$cell_type <- unname(labels[as.character(Idents(pbmc))])

# 7. Plots
ggsave("plots/03_dimplot_clusters.png", DimPlot(pbmc, label = TRUE, pt.size = 0.5) + NoLegend(), width = 6, height = 5)
ggsave("plots/04_dimplot_celltypes.png", DimPlot(pbmc, group.by = "cell_type", label = TRUE, repel = TRUE, pt.size = 0.5) + NoLegend(), width = 6, height = 5)
feat <- c("MS4A1","GNLY","CD3E","CD14","FCER1A","FCGR3A","LYZ","PPBP","CD8A")
ggsave("plots/05_featureplot.png", FeaturePlot(pbmc, feat, ncol = 3, order = TRUE), width = 11, height = 10)
ggsave("plots/06_vlnplot.png", VlnPlot(pbmc, c("MS4A1","CD79A","NKG7","PPBP"), group.by = "cell_type", pt.size = 0, ncol = 2), width = 11, height = 7)
top <- markers |> group_by(cluster) |> slice_max(avg_log2FC, n = 3) |> pull(gene) |> unique()
ggsave("plots/07_dotplot.png", DotPlot(pbmc, features = top) + RotatedAxis(), width = 12, height = 5)
pbmc <- ScaleData(pbmc, features = rownames(pbmc), verbose = FALSE)
ggsave("plots/08_heatmap.png", DoHeatmap(pbmc, features = top) + NoLegend(), width = 10, height = 8)

saveRDS(pbmc, "pbmc3k_seurat.rds")
cat("CELLS", ncol(pbmc), "CLUSTERS", nlevels(pbmc), "\n"); print(table(pbmc$seurat_clusters, pbmc$cell_type))
