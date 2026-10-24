# The objects scfastR writes are ones Seurat accepts downstream. Each Fast step is also
# tested on Seurat's own upstream result, so a kernel can be checked before the others land.

toy_seurat_pca <- function() {
  obj <- toy_object()
  cached("toy_seurat_pca", Seurat::RunPCA(Seurat::ScaleData(obj, verbose = FALSE), npcs = 20, verbose = FALSE))
}

toy_seurat_snn <- function() {
  cached("toy_seurat_snn", Seurat::FindNeighbors(toy_seurat_pca(), dims = 1:10, verbose = FALSE))
}

builds <- function(p) {
  expect_s3_class(p, "ggplot")
  expect_no_error(ggplot2::ggplot_build(p))
}

test_that("RunPCAFast stores a valid DimReduc without ScaleData", {
  obj <- toy_object()
  o <- skip_if_stub(RunPCAFast(obj, npcs = 20, verbose = FALSE))
  red <- o[["pca"]]
  expect_s4_class(red, "DimReduc")
  expect_true(methods::validObject(red))
  emb <- SeuratObject::Embeddings(red)
  load <- SeuratObject::Loadings(red)
  hvg <- SeuratObject::VariableFeatures(obj)
  expect_equal(dim(emb), c(ncol(obj), 20))
  expect_identical(rownames(emb), colnames(obj))
  expect_identical(colnames(emb), paste0("PC_", 1:20))
  expect_identical(rownames(load), hvg)
  expect_identical(SeuratObject::Key(red), "PC_")
  expect_identical(SeuratObject::DefaultAssay(red), "RNA")
  sdev <- SeuratObject::Stdev(red)
  expect_length(sdev, 20)
  expect_true(all(sdev > 0))
  expect_true(all(diff(sdev) <= 1e-4 * sdev[1]))
  expect_false("scale.data" %in% SeuratObject::Layers(o[["RNA"]]))
  expect_true(any(startsWith(names(o@commands), "RunPCAFast")))

  # Seurat's conventions: stdev = sd of the embedding columns, orthonormal loadings, and
  # embeddings = scaled data %*% loadings (checked against an explicit ScaleData here). The
  # last needs an accurate SVD, so this run raises the randomised-SVD settings.
  expect_equal(unname(apply(emb, 2, stats::sd)), sdev, tolerance = 1e-3)
  expect_equal(unname(crossprod(load)), diag(20), tolerance = 1e-3)
  acc <- RunPCAFast(obj, npcs = 20, oversample = 40, power.iters = 12, verbose = FALSE)[["pca"]]
  expect_identical(acc@misc$scfast$power.iters, 12)
  z <- t(scale(t(as.matrix(SeuratObject::LayerData(obj, layer = "data")[hvg, ]))))
  expect_equal(unname(t(z) %*% SeuratObject::Loadings(acc)), unname(SeuratObject::Embeddings(acc)), tolerance = 1e-3)

  builds(Seurat::ElbowPlot(o))
  builds(Seurat::DimPlot(o, reduction = "pca"))
  o2 <- Seurat::FindNeighbors(o, dims = 1:10, verbose = FALSE)
  expect_true(all(c("RNA_nn", "RNA_snn") %in% SeuratObject::Graphs(o2)))
})

test_that("RunPCAFast respects reduction.name, reduction.key, features and the seed alias", {
  obj <- toy_object()
  feats <- SeuratObject::VariableFeatures(obj)[1:40]
  o <- skip_if_stub(RunPCAFast(obj, features = feats, npcs = 5, seed = 3, reduction.name = "fastpca",
                               reduction.key = "fPC_", verbose = FALSE))
  red <- o[["fastpca"]]
  expect_identical(rownames(SeuratObject::Loadings(red)), feats)
  expect_identical(colnames(SeuratObject::Embeddings(red)), paste0("fPC_", 1:5))
  expect_identical(red@misc$scfast$seed, 3)
  expect_false("pca" %in% SeuratObject::Reductions(o))
})

test_that("RunPCAFast works on a Seurat v3 Assay", {
  skip_if_no_seurat()
  obj <- toy_object()
  counts <- SeuratObject::LayerData(obj, layer = "counts")
  o3 <- withr::with_options(list(Seurat.object.assay.version = "v3"), SeuratObject::CreateSeuratObject(counts))
  expect_false(inherits(o3[["RNA"]], "Assay5"))
  o3 <- Seurat::NormalizeData(o3, verbose = FALSE)
  SeuratObject::VariableFeatures(o3) <- SeuratObject::VariableFeatures(obj)
  i3 <- scfastR:::.pca_input(o3, "RNA")
  i5 <- scfastR:::.pca_input(obj, "RNA")
  expect_identical(i3$features, i5$features)
  expect_equal(unname(as.matrix(i3$m)), unname(as.matrix(i5$m)))
  o3 <- skip_if_stub(RunPCAFast(o3, npcs = 5, verbose = FALSE))
  expect_s4_class(o3[["pca"]], "DimReduc")
})

test_that("FindNeighborsFast stores Seurat-style kNN and SNN graphs", {
  o <- toy_seurat_pca()
  o2 <- skip_if_stub(FindNeighborsFast(o, dims = 1:10, k.param = 20, verbose = FALSE))
  nn <- o2[["RNA_nn"]]
  snn <- o2[["RNA_snn"]]
  for (g in list(nn, snn)) {
    expect_s4_class(g, "Graph")
    expect_identical(SeuratObject::DefaultAssay(g), "RNA")
    expect_identical(dimnames(g), list(colnames(o), colnames(o)))
  }
  expect_true(all(Matrix::rowSums(nn) == 20))
  expect_true(all(Matrix::diag(nn) == 1))
  expect_true(all(nn@x == 1))
  expect_true(Matrix::isSymmetric(snn))
  expect_true(all(Matrix::diag(snn) == 1))
  expect_true(all(snn@x >= 1 / 15 - 1e-6 & snn@x <= 1 + 1e-6))
  expect_true(any(startsWith(names(o2@commands), "FindNeighborsFast")))
  # Seurat clusters on it (Louvain: no Python needed)
  o3 <- Seurat::FindClusters(o2, resolution = 0.5, verbose = FALSE)
  expect_gt(nlevels(o3$seurat_clusters), 1)
})

test_that("FindNeighborsFast graph.name handling matches Seurat", {
  o <- toy_seurat_pca()
  o2 <- skip_if_stub(FindNeighborsFast(o, dims = 1:5, graph.name = c("my_nn", "my_snn"), verbose = FALSE))
  expect_true(all(c("my_nn", "my_snn") %in% SeuratObject::Graphs(o2)))
  expect_message(o3 <- FindNeighborsFast(o, dims = 1:5, graph.name = "only_nn", verbose = FALSE),
                 "storing nearest-neighbor graph only")
  expect_identical(setdiff(SeuratObject::Graphs(o3), SeuratObject::Graphs(o)), "only_nn")
})

test_that("FindClustersFast writes seurat_clusters, Idents and the res column", {
  o <- toy_seurat_snn()
  o2 <- skip_if_stub(FindClustersFast(o, resolution = 0.5, verbose = FALSE))
  cl <- o2$seurat_clusters
  expect_s3_class(cl, "factor")
  expect_identical(levels(cl), as.character(seq_len(nlevels(cl)) - 1L))
  expect_true("RNA_snn_res.0.5" %in% colnames(o2[[]]))
  expect_identical(as.character(o2$RNA_snn_res.0.5), as.character(cl))
  expect_identical(unname(SeuratObject::Idents(o2)), unname(cl))
  expect_identical(levels(SeuratObject::Idents(o2)), levels(cl))
  expect_true(any(startsWith(names(o2@commands), "FindClustersFast")))
  expect_gt(ari(cl, o2$group), 0.9)
  builds(Seurat::DimPlot(o2, reduction = "pca"))
  mk <- Seurat::FindMarkers(o2, ident.1 = "0", verbose = FALSE)
  expect_s3_class(mk, "data.frame")
  expect_gt(nrow(mk), 0)
})

test_that("FindClustersFast: several resolutions, seed alias, determinism", {
  o <- toy_seurat_snn()
  a <- skip_if_stub(FindClustersFast(o, resolution = c(0.3, 1), seed = 7, verbose = FALSE))
  expect_true(all(c("RNA_snn_res.0.3", "RNA_snn_res.1") %in% colnames(a[[]])))
  expect_identical(as.character(a$seurat_clusters), as.character(a$RNA_snn_res.1))
  b <- FindClustersFast(o, resolution = c(0.3, 1), random.seed = 7, algorithm = 4, method = "matrix", verbose = FALSE)
  expect_identical(a$RNA_snn_res.0.3, b$RNA_snn_res.0.3)
  expect_identical(a$RNA_snn_res.1, b$RNA_snn_res.1)
})

test_that("singletons are merged into their most connected cluster", {
  # two 4-cliques plus one cell attached to the first clique only
  n <- 9
  g <- matrix(0, n, n)
  g[1:4, 1:4] <- 1
  g[5:8, 5:8] <- 1
  g[9, 1:2] <- g[1:2, 9] <- 0.2
  diag(g) <- 1
  g <- methods::as(methods::as(g, "CsparseMatrix"), "generalMatrix")
  labels <- c(0L, 0L, 0L, 0L, 1L, 1L, 1L, 1L, 2L)
  out <- scfastR:::.group_singletons(labels, g, verbose = FALSE)
  expect_identical(out, c(rep(0L, 4), rep(1L, 4), 0L))
  # an isolated singleton picks a cluster without disturbing the caller's RNG stream
  g2 <- g
  g2[9, 1:2] <- g2[1:2, 9] <- 0
  g2 <- Matrix::drop0(g2)
  set.seed(42)
  r1 <- stats::runif(1)
  set.seed(42)
  out2 <- scfastR:::.group_singletons(labels, g2, verbose = FALSE)
  expect_identical(stats::runif(1), r1)
  expect_true(out2[9] %in% c(0L, 1L))
})

test_that("the full fast chain feeds RunUMAP, DimPlot and FindAllMarkers", {
  obj <- toy_object()
  o <- skip_if_stub(RunPCAFast(obj, npcs = 20, verbose = FALSE))
  o <- skip_if_stub(FindNeighborsFast(o, dims = 1:10, verbose = FALSE))
  o <- skip_if_stub(FindClustersFast(o, verbose = FALSE))
  o <- suppressWarnings(Seurat::RunUMAP(o, dims = 1:10, verbose = FALSE))
  expect_equal(dim(SeuratObject::Embeddings(o, "umap")), c(ncol(o), 2))
  builds(Seurat::DimPlot(o, label = TRUE))
  mk <- Seurat::FindAllMarkers(o, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25, verbose = FALSE)
  expect_gt(nrow(mk), 0)
  expect_true(all(as.character(mk$cluster) %in% levels(o$seurat_clusters)))
  expect_gt(ari(o$seurat_clusters, o$group), 0.9)
})

test_that("RunUMAPFast stores a finite umap DimReduc that separates the planted cell types", {
  testthat::skip_if_not_installed("cluster")
  o <- skip_if_stub(RunUMAPFast(fixture_seurat_ref(), dims = 1:10, verbose = FALSE))
  red <- o[["umap"]]
  expect_true(methods::validObject(red))
  emb <- SeuratObject::Embeddings(red)
  expect_equal(dim(emb), c(ncol(o), 2))
  expect_identical(colnames(emb), c("umap_1", "umap_2"))
  expect_true(all(is.finite(emb)))
  sil <- cluster::silhouette(as.integer(factor(o$cell_type)), stats::dist(emb))[, 3]
  expect_gt(mean(sil), 0.7)
})
