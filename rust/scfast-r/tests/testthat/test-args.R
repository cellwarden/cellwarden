# Argument validation. These checks run before any kernel, so they pass with stub kernels.

test_that("scfast_pca validates its arguments", {
  m <- Matrix::rsparsematrix(30, 20, 0.3)
  expect_error(scfast_pca("a"), "dgCMatrix")
  expect_error(scfast_pca(list(1)), "dgCMatrix")
  expect_error(scfast_pca(m, npcs = 0), "npcs")
  expect_error(scfast_pca(m, npcs = 2.5), "npcs")
  expect_error(scfast_pca(m, npcs = NA), "npcs")
  expect_error(scfast_pca(m, npcs = c(2, 3)), "npcs")
  expect_error(scfast_pca(m, npcs = 20), "less than")
  expect_error(scfast_pca(m, seed = -1), "seed")
  expect_error(scfast_pca(m, seed = 1.5), "seed")
  expect_error(scfast_pca(m, oversample = -1), "oversample")
  bad <- m
  bad@x[1] <- NA
  expect_error(scfast_pca(bad, npcs = 2), "non-finite", class = "scfast_error")
})

test_that("scfast_knn validates its arguments", {
  x <- matrix(rnorm(60), 20, 3)
  expect_error(scfast_knn("a"), "numeric matrix")
  expect_error(scfast_knn(x, k = 0), "'k'")
  expect_error(scfast_knn(x, k = 1), "'k'")
  expect_error(scfast_knn(x, k = 21), "larger than the number of cells")
  expect_error(scfast_knn(x, method = "annoy"), "'method'")
  expect_error(scfast_knn(x, seed = -2), "seed")
  expect_error(scfast_knn(x, hnsw.m = 0), "hnsw.m")
  y <- x
  y[1, 1] <- NA
  expect_error(scfast_knn(y, k = 5), "non-finite", class = "scfast_error")
})

test_that("scfast_snn validates its arguments", {
  idx <- matrix(c(1:4, 2L, 1L, 4L, 3L), 4, 2)
  expect_error(scfast_snn(1:4), "matrix")
  expect_error(scfast_snn(idx + 0.5), "whole numbers")
  expect_error(scfast_snn(idx, prune = -0.1), "prune")
  expect_error(scfast_snn(idx, prune = 2), "prune")
  bad <- idx
  bad[1, 2] <- 0L
  expect_error(scfast_snn(bad), "outside 1..4", class = "scfast_error")
})

test_that("scfast_leiden validates its arguments", {
  g <- Matrix::sparseMatrix(i = c(1, 2, 2, 3), j = c(2, 1, 3, 2), x = 1, dims = c(3, 3))
  expect_error(scfast_leiden(g[1:2, ]), "square")
  expect_error(scfast_leiden(g, resolution = -0.1), "resolution")
  expect_error(scfast_leiden(g, resolution = c(0.5, 1)), "single number")
  expect_error(scfast_leiden(g, n.iter = 0), "n.iter")
  expect_error(scfast_leiden(g, seed = -1), "seed")
  asym <- Matrix::sparseMatrix(i = c(1, 2), j = c(2, 3), x = 1, dims = c(3, 3))
  expect_warning(tryCatch(scfast_leiden(asym), scfast_not_implemented = function(e) NULL), "not symmetric")
})

test_that("RunPCAFast validates its arguments", {
  obj <- toy_object()
  expect_error(RunPCAFast(list()), "Seurat object")
  expect_error(RunPCAFast(obj, assay = "ADT"), "assay 'ADT' not found")
  expect_error(RunPCAFast(obj, npcs = 0), "npcs")
  expect_error(RunPCAFast(obj, seed.use = -1), "seed.use")
  expect_error(RunPCAFast(obj, reduction.name = ""), "reduction.name")
  expect_error(RunPCAFast(obj, device = "tpu"), "device")
  expect_error(RunPCAFast(obj, device = "cuda:x"), "device")
  expect_error(RunPCAFast(obj, device = "cuda"), "without CUDA")
  expect_error(RunPCAFast(obj, verbose = NA), "verbose")
  expect_error(RunPCAFast(obj, features = 1:10), "character")
  expect_error(suppressWarnings(RunPCAFast(obj, features = c("nope1", "nope2"))), "fewer than 2")
  # unknown features are dropped with a warning (the kernel may still be a stub)
  expect_warning(tryCatch(RunPCAFast(obj, features = c(rownames(obj)[1:20], "nope"), npcs = 5, verbose = FALSE),
                          scfast_not_implemented = function(e) NULL), "not in assay")
  # not normalised
  raw <- SeuratObject::CreateSeuratObject(SeuratObject::LayerData(obj, layer = "counts"))
  expect_error(RunPCAFast(raw, features = rownames(raw)), "NormalizeData")
  # no variable features
  o2 <- Seurat::NormalizeData(raw, verbose = FALSE)
  expect_error(RunPCAFast(o2), "FindVariableFeatures")
  expect_error(RunPCAFast(obj, power.iters = -1), "power.iters")
})

test_that("RunPCAFast drops zero-variance features and caps npcs like Seurat", {
  obj <- toy_object()
  counts <- SeuratObject::LayerData(obj, layer = "counts")
  counts[1:2, ] <- 0
  o <- SeuratObject::CreateSeuratObject(counts)
  o <- Seurat::NormalizeData(o, verbose = FALSE)
  expect_warning(inp <- scfastR:::.pca_input(o, "RNA", features = rownames(o)[1:30]), "zero variance")
  expect_identical(inp$features, rownames(o)[3:30])
  expect_identical(nrow(inp$m), 28L)
  expect_warning(tryCatch(RunPCAFast(obj, features = rownames(obj)[1:10], npcs = 30, verbose = FALSE),
                          scfast_not_implemented = function(e) NULL), "reduced to 9")
})

test_that("RunPCAFast rejects split layers", {
  obj <- toy_object()
  o <- obj
  o[["RNA"]] <- split(o[["RNA"]], f = rep(c("a", "b"), length.out = ncol(o)))
  expect_error(RunPCAFast(o), "JoinLayers")
})

test_that("FindNeighborsFast validates its arguments", {
  obj <- toy_object()
  expect_error(FindNeighborsFast(list()), "Seurat object")
  expect_error(FindNeighborsFast(obj), "reduction 'pca' not found")
  o <- Seurat::ScaleData(obj, verbose = FALSE)
  o <- Seurat::RunPCA(o, npcs = 10, verbose = FALSE)
  expect_error(FindNeighborsFast(o, dims = 1:11), "More dimensions")
  expect_error(FindNeighborsFast(o, dims = c(1, 1)), "duplicates")
  expect_error(FindNeighborsFast(o, dims = 0:3), "positive")
  expect_error(FindNeighborsFast(o, k.param = 1), "k.param")
  expect_error(FindNeighborsFast(o, prune.SNN = 2), "prune.SNN")
  expect_error(FindNeighborsFast(o, method = "annoy"), "'method'")
  expect_error(FindNeighborsFast(o, graph.name = c("a", "b", "c")), "graph.name")
  expect_error(FindNeighborsFast(o, device = "cuda"), "without CUDA")
})

test_that("FindClustersFast validates its arguments", {
  obj <- toy_object()
  o <- Seurat::ScaleData(obj, verbose = FALSE)
  o <- Seurat::RunPCA(o, npcs = 10, verbose = FALSE)
  expect_error(FindClustersFast(o), "graph.name not present")
  o <- Seurat::FindNeighbors(o, dims = 1:10, verbose = FALSE)
  expect_error(FindClustersFast(list()), "Seurat object")
  expect_error(FindClustersFast(o, graph.name = "pca"), "not present|graph object")
  expect_error(FindClustersFast(o, resolution = -1), "resolution")
  expect_error(FindClustersFast(o, n.iter = 0), "n.iter")
  expect_error(FindClustersFast(o, random.seed = 0.5), "random.seed")
  expect_error(FindClustersFast(o, algorithm = 1), "Leiden only")
  expect_error(FindClustersFast(o, n.start = 10), "unused argument")
  expect_error(FindClustersFast(o, seed = 1, random.seed = 2), "not both")
  expect_error(FindClustersFast(o, group.singletons = "yes"), "group.singletons")
})
