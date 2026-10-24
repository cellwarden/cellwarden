# Phase 9 (docs/scfast-integration.md): PreprocessFast() / scfast_preprocess() against
# Seurat's NormalizeData() + FindVariableFeatures(selection.method = "vst").
#   data layer            identical to NormalizeData (LogNormalize, scale.factor 1e4)
#   loess                 identical to R's loess(surface = "direct")
#   HVG overlap           >= 95% vs FindVariableFeatures(vst) on pbmc_like and a 2k-cell simulation
#   scaling statistics    mean / sd of the data layer per HVG, as ScaleData uses
# Set SCFAST_TEST_METRICS=/path/file.tsv to record the measured values.

record_pre <- function(metric, value) {
  out <- Sys.getenv("SCFAST_TEST_METRICS", "")
  if (nzchar(out)) cat(sprintf("%s\t%.6f\n", metric, value), file = out, append = TRUE)
  invisible(value)
}

# Raw fixture counts as a Seurat object (no normalisation).
fixture_raw <- function() {
  skip_if_no_seurat()
  fx <- fixture_dir()
  testthat::skip_if(!nzchar(fx), "pbmc_like fixture not found (set SCFAST_FIXTURE_DIR)")
  cached("fixture_raw", {
    m <- methods::as(Matrix::readMM(file.path(fx, "matrix.mtx")), "CsparseMatrix")
    feats <- utils::read.delim(file.path(fx, "features.tsv"), header = FALSE, colClasses = "character", quote = "")
    bcs <- readLines(file.path(fx, "barcodes.tsv"))
    dimnames(m) <- list(feats[[2]], bcs)
    suppressWarnings(SeuratObject::CreateSeuratObject(m, min.cells = 0, min.features = 0))
  })
}

# 2,000 cells x 3,000 genes: five groups with their own up-regulated genes, log-normal
# library sizes, negative binomial counts.
simulated_counts <- function(n_cells = 2000, n_genes = 3000, seed = 7) {
  cached(paste("sim", n_cells, n_genes, seed, sep = "_"), withr::with_seed(seed, {
    groups <- sample(5, n_cells, TRUE)
    base <- stats::rgamma(n_genes, 0.5, 0.5) * 0.2
    rate <- matrix(base, n_genes, n_cells)
    for (g in 1:5) {
      de <- sample(n_genes, 150)
      rate[de, groups == g] <- rate[de, groups == g] * stats::runif(150, 2, 8)
    }
    rate <- sweep(rate, 2, stats::rlnorm(n_cells, 0, 0.4), "*")
    cnt <- matrix(stats::rnbinom(n_genes * n_cells, mu = rate, size = 2), n_genes, n_cells,
                  dimnames = list(paste0("gene", seq_len(n_genes)), paste0("cell", seq_len(n_cells))))
    methods::as(cnt, "CsparseMatrix")
  }))
}

seurat_vst <- function(counts, n) {
  obj <- SeuratObject::CreateSeuratObject(counts)
  obj <- Seurat::NormalizeData(obj, verbose = FALSE)
  Seurat::FindVariableFeatures(obj, selection.method = "vst", nfeatures = n, verbose = FALSE)
}

overlap <- function(a, b) length(intersect(a, b)) / length(a)

test_that("the loess matches R's loess(surface = 'direct') on ties and plain data", {
  withr::with_seed(3, {
    x <- sort(c(stats::rnorm(230), rep(0.5, 7)))
    y <- sin(x) + stats::rnorm(237, sd = 0.3)
  })
  ref <- stats::fitted(stats::loess(y ~ x, span = 0.3, degree = 2, control = stats::loess.control(surface = "direct")))
  f <- skip_if_stub(scfastR:::rs_loess_direct(x, y, 0.3))
  expect_equal(f, unname(ref), tolerance = 1e-10)
  # Unsorted input, span > 1
  o <- sample(237)
  ref2 <- stats::fitted(stats::loess(y[o] ~ x[o], span = 1.5, degree = 2, control = stats::loess.control(surface = "direct")))
  expect_equal(scfastR:::rs_loess_direct(x[o], y[o], 1.5), unname(ref2), tolerance = 1e-10)
})

test_that("scfast_preprocess reproduces NormalizeData and the vst statistics on the toy data", {
  obj <- fixture_raw()
  counts <- SeuratObject::LayerData(obj, layer = "counts")
  res <- skip_if_stub(scfast_preprocess(counts, n_hvg = 500))
  ref <- seurat_vst(counts, 500)
  # data layer: identical values, shared structure
  d <- SeuratObject::LayerData(ref, layer = "data")
  expect_s4_class(res$data, "dgCMatrix")
  expect_identical(dim(res$data), dim(counts))
  expect_identical(dimnames(res$data), dimnames(counts))
  expect_identical(res$data@p, counts@p)
  expect_identical(res$data@i, counts@i)
  expect_equal(res$data@x, d[rownames(res$data), colnames(res$data)]@x, tolerance = 1e-12)
  expect_equal(unname(res$library_size), unname(Matrix::colSums(counts)))
  # vst columns (mean, variance exact; expected variance differs by the loess surface)
  info <- Seurat::HVFInfo(ref, method = "vst")
  expect_equal(res$hvg_stats$mean, unname(info$mean), tolerance = 1e-10)
  expect_equal(res$hvg_stats$variance, unname(info$variance), tolerance = 1e-10)
  nc <- res$hvg_stats$variance > 0
  rel <- abs(log(res$hvg_stats$variance.expected[nc] / info$variance.expected[nc]))
  record_pre("vst_expected_variance_max_log_ratio", max(rel))
  expect_lt(stats::median(rel), 0.02)
  # scaling statistics are the mean / sd of the data layer per HVG
  dm <- as.matrix(d[res$hvg, ])
  expect_equal(unname(res$hvg_mean), unname(rowMeans(dm)), tolerance = 1e-10)
  expect_equal(unname(res$hvg_sd), unname(apply(dm, 1, stats::sd)), tolerance = 1e-10)
  expect_identical(res$clip, sqrt(ncol(counts)))
  expect_length(res$hvg, 500)
  expect_identical(res$hvg, rownames(res$hvg_stats)[order(res$hvg_stats$rank, na.last = NA)])
})

test_that("HVG overlap with FindVariableFeatures(vst) is >= 95% on pbmc_like", {
  obj <- fixture_raw()
  o <- skip_if_stub(PreprocessFast(obj, n_hvg = 500, verbose = FALSE))
  ref <- seurat_vst(SeuratObject::LayerData(obj, layer = "counts"), 500)
  ov <- overlap(SeuratObject::VariableFeatures(o), SeuratObject::VariableFeatures(ref))
  record_pre("hvg_overlap_pbmc_like", ov)
  expect_gte(ov, 0.95)
})

test_that("HVG overlap with FindVariableFeatures(vst) is >= 95% on a simulated 2k-cell dataset", {
  skip_if_no_seurat()
  counts <- simulated_counts()
  res <- skip_if_stub(scfast_preprocess(counts, n_hvg = 2000))
  ref <- seurat_vst(counts, 2000)
  ov <- overlap(res$hvg, SeuratObject::VariableFeatures(ref))
  record_pre("hvg_overlap_sim2k", ov)
  expect_gte(ov, 0.95)
  # The data layer agrees here too.
  d <- SeuratObject::LayerData(ref, layer = "data")
  expect_equal(res$data@x, d@x, tolerance = 1e-12)
})

test_that("PreprocessFast writes a Seurat object that RunPCAFast and Seurat accept", {
  obj <- fixture_raw()
  o <- skip_if_stub(PreprocessFast(obj, n_hvg = 500, verbose = FALSE))
  expect_true("data" %in% SeuratObject::Layers(o[["RNA"]]))
  expect_length(SeuratObject::VariableFeatures(o), 500)
  expect_true(all(SeuratObject::VariableFeatures(o) %in% rownames(o)))
  info <- Seurat::HVFInfo(o, method = "vst")
  expect_identical(rownames(info), rownames(o))
  expect_true(all(c("mean", "variance", "variance.standardized") %in% colnames(info)))
  expect_identical(sum(o[["RNA"]][["vf_vst_counts_variable"]][[1]]), 500L)
  st <- SeuratObject::Misc(o, "scfast")$hvg_stats
  expect_s3_class(st, "data.frame")
  expect_equal(nrow(st), nrow(o))
  expect_identical(sum(st$variable), 500L)
  pre <- SeuratObject::Misc(o, "scfast")$preprocess
  expect_identical(names(pre$hvg_mean), SeuratObject::VariableFeatures(o))
  expect_true(any(startsWith(names(o@commands), "PreprocessFast")))
  # identical to the low-level call
  low <- scfast_preprocess(SeuratObject::LayerData(obj, layer = "counts"), n_hvg = 500)
  expect_identical(SeuratObject::VariableFeatures(o), low$hvg)
  expect_identical(SeuratObject::LayerData(o, layer = "data")@x, low$data@x)
  # downstream
  o2 <- RunPCAFast(o, npcs = 20, verbose = FALSE)
  expect_equal(dim(SeuratObject::Embeddings(o2, "pca")), c(ncol(o), 20L))
  # the implicit PCA's scaling statistics equal the ones PreprocessFast reported
  expect_equal(o2[["pca"]]@misc$scfast$gene_mean, pre$hvg_mean, tolerance = 1e-6)
  expect_equal(o2[["pca"]]@misc$scfast$gene_sd, pre$hvg_sd, tolerance = 1e-6)
  # HVFInfo() / VariableFeaturePlot() need the method, since no FindVariableFeatures command is logged.
  expect_no_error(Seurat::VariableFeaturePlot(o, selection.method = "vst"))
  o3 <- Seurat::ScaleData(o, verbose = FALSE)
  expect_true("scale.data" %in% SeuratObject::Layers(o3[["RNA"]]))
})

test_that("PreprocessFast checks its arguments and the assay", {
  obj <- fixture_raw()
  expect_error(PreprocessFast("x"), "Seurat object")
  expect_error(PreprocessFast(obj, assay = "nope"), "not found")
  expect_error(PreprocessFast(obj, n_hvg = 0), "n_hvg")
  expect_error(PreprocessFast(obj, scale.factor = -1), "scale.factor")
  expect_error(PreprocessFast(obj, loess.span = 0), "loess.span")
  expect_error(PreprocessFast(obj, clip.max = 0), "clip.max")
  expect_warning(o <- skip_if_stub(PreprocessFast(obj, n_hvg = 1e6, verbose = FALSE)), "reduced")
  expect_length(SeuratObject::VariableFeatures(o), nrow(obj))
  expect_error(scfast_preprocess(obj), "dgCMatrix")
  neg <- SeuratObject::LayerData(obj, layer = "counts")
  neg@x[1] <- -1
  expect_error(scfast_preprocess(neg, n_hvg = 10), "non-negative")
})
