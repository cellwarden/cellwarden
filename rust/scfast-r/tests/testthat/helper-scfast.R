# Shared helpers for the scfastR tests.
#
# Kernels are implemented in rust/scfast-core one at a time; until a kernel lands it panics
# with unimplemented!(), which scfastR raises as an R error of class
# "scfast_not_implemented". Functional tests wrap kernel calls in skip_if_stub() so they
# skip ("kernel not implemented yet") instead of failing.

skip_if_stub <- function(expr) {
  tryCatch(expr, scfast_not_implemented = function(e) {
    testthat::skip(paste("kernel not implemented yet:", conditionMessage(e)))
  })
}

skip_if_no_seurat <- function() {
  testthat::skip_if_not_installed("Seurat")
  suppressPackageStartupMessages(requireNamespace("Seurat", quietly = TRUE))
}

# Seurat's FindClusters(algorithm = 4) runs Python leidenalg through reticulate; point
# RETICULATE_PYTHON at an interpreter that has it.
skip_if_no_leidenalg <- function() {
  testthat::skip_if_not_installed("reticulate")
  testthat::skip_if_not_installed("leiden")
  if (!reticulate::py_module_available("leidenalg")) {
    testthat::skip("Python leidenalg not available (set RETICULATE_PYTHON)")
  }
}

.cache <- new.env(parent = emptyenv())

cached <- function(key, expr) {
  if (!exists(key, envir = .cache, inherits = FALSE)) assign(key, expr, envir = .cache)
  get(key, envir = .cache, inherits = FALSE)
}

# ---------------------------------------------------------------- data

# tests/fixtures/pbmc_like in the repository (500 cells x 800 genes, four planted cell types).
fixture_dir <- function() {
  env <- Sys.getenv("SCFAST_FIXTURE_DIR", "")
  if (nzchar(env)) return(env)
  d <- normalizePath(getwd(), mustWork = FALSE)
  for (i in 1:8) {
    cand <- file.path(d, "tests", "fixtures", "pbmc_like")
    if (file.exists(file.path(cand, "matrix.mtx"))) return(cand)
    d <- dirname(d)
  }
  ""
}

# Normalised fixture with 500 variable features, plus the planted cell types.
fixture_object <- function() {
  skip_if_no_seurat()
  fx <- fixture_dir()
  testthat::skip_if(!nzchar(fx), "pbmc_like fixture not found (set SCFAST_FIXTURE_DIR)")
  cached("fixture", {
    m <- methods::as(Matrix::readMM(file.path(fx, "matrix.mtx")), "CsparseMatrix")
    feats <- utils::read.delim(file.path(fx, "features.tsv"), header = FALSE, colClasses = "character", quote = "")
    bcs <- readLines(file.path(fx, "barcodes.tsv"))
    cells <- utils::read.delim(file.path(fx, "cells.tsv"), colClasses = "character", quote = "")
    dimnames(m) <- list(feats[[2]], bcs)
    obj <- suppressWarnings(SeuratObject::CreateSeuratObject(m, min.cells = 0, min.features = 0))
    obj$cell_type <- cells$cell_type[match(colnames(obj), cells$barcode)]
    obj <- Seurat::NormalizeData(obj, verbose = FALSE)
    Seurat::FindVariableFeatures(obj, nfeatures = 500, verbose = FALSE)
  })
}

# The fixture through Seurat's own ScaleData + RunPCA + FindNeighbors (the reference).
fixture_seurat_ref <- function(npcs = 20) {
  obj <- fixture_object()
  cached(paste0("seurat_ref_", npcs), {
    o <- Seurat::ScaleData(obj, verbose = FALSE)
    o <- Seurat::RunPCA(o, npcs = npcs, seed.use = 0, verbose = FALSE)
    Seurat::FindNeighbors(o, dims = 1:10, k.param = 20, verbose = FALSE)
  })
}

# Small self-contained object (no fixture needed): 3 groups with their own marker genes.
toy_object <- function(n_cells = 150, n_genes = 120, seed = 1) {
  skip_if_no_seurat()
  cached(paste("toy", n_cells, n_genes, seed, sep = "_"), withr::with_seed(seed, {
    group <- rep(1:3, length.out = n_cells)
    rate <- matrix(0.3, n_genes, n_cells)
    for (g in 1:3) rate[(g - 1) * 15 + 1:15, group == g] <- 4
    counts <- matrix(stats::rpois(n_genes * n_cells, rate), n_genes, n_cells,
                     dimnames = list(paste0("gene", seq_len(n_genes)), paste0("cell", seq_len(n_cells))))
    obj <- SeuratObject::CreateSeuratObject(methods::as(counts, "CsparseMatrix"))
    obj$group <- factor(group)
    obj <- Seurat::NormalizeData(obj, verbose = FALSE)
    Seurat::FindVariableFeatures(obj, nfeatures = 80, verbose = FALSE)
  }))
}

# ---------------------------------------------------------------- metrics (as bench/parity.py)

# Cosines of the principal angles between the column spaces of a[, 1:n] and b[, 1:n].
subspace_similarity <- function(a, b, n) {
  n <- min(n, ncol(a), ncol(b))
  qa <- qr.Q(qr(scale(a[, seq_len(n), drop = FALSE], center = TRUE, scale = FALSE)))
  qb <- qr.Q(qr(scale(b[, seq_len(n), drop = FALSE], center = TRUE, scale = FALSE)))
  cos <- pmin(pmax(svd(crossprod(qa, qb), nu = 0, nv = 0)$d, 0), 1)
  c(mean = mean(cos), min = min(cos))
}

# Adjusted Rand index.
ari <- function(a, b) {
  tab <- table(a, b)
  c2 <- function(x) sum(choose(x, 2))
  index <- c2(tab)
  ea <- c2(rowSums(tab))
  eb <- c2(colSums(tab))
  expected <- ea * eb / choose(sum(tab), 2)
  maxi <- (ea + eb) / 2
  if (maxi == expected) return(1)
  (index - expected) / (maxi - expected)
}

# Mean fraction of shared neighbours between two kNN graphs (rows = cells, k entries each).
knn_overlap <- function(g1, g2) {
  k <- mean(Matrix::rowSums(g1 > 0))
  mean(Matrix::rowSums((g1 > 0) * (g2 > 0))) / k
}

# n x k neighbour index matrix (self first) from a 0/1 kNN Graph such as Seurat's RNA_nn.
nn_graph_to_idx <- function(g) {
  tg <- Matrix::t(methods::as(g, "CsparseMatrix")) # column i of t(g) = neighbours of cell i
  n <- ncol(tg)
  k <- unique(diff(tg@p))
  stopifnot(length(k) == 1L)
  idx <- matrix(tg@i + 1L, nrow = k)
  # put the cell itself first, keep the rest in index order
  t(vapply(seq_len(n), function(i) {
    col <- idx[, i]
    c(i, col[col != i])[seq_len(k)]
  }, integer(k)))
}
