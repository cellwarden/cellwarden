# scfast_harmony() and RunHarmonyFast(): batch shifts are removed, cell types stay apart,
# the DimReduc follows RunHarmony()'s conventions, and bad arguments are rejected.

# Gaussian cell types in 10 dims with a shift per batch (rows interleaved).
harmony_blobs <- function(n_per = 100, types = 3, batches = 3, d = 10, seed = 1) {
  withr::with_seed(seed, {
    centres <- matrix(stats::rnorm(types * d, sd = 10), types, d)
    shifts <- matrix(stats::rnorm(batches * d, sd = 4), batches, d)
    n <- n_per * types * batches
    ty <- rep(seq_len(types), length.out = n)
    b <- sample(rep(seq_len(batches), length.out = n))
    emb <- centres[ty, ] + shifts[b, ] + matrix(stats::rnorm(n * d), n, d)
    dimnames(emb) <- list(paste0("cell", seq_len(n)), paste0("PC_", seq_len(d)))
    list(emb = emb, batch = paste0("b", b), type = ty)
  })
}

batch_spread <- function(emb, batch, type) {
  out <- c()
  for (t in unique(type)) {
    cs <- t(sapply(sort(unique(batch)), function(b) colMeans(emb[type == t & batch == b, , drop = FALSE])))
    out <- c(out, as.numeric(stats::dist(cs)))
  }
  mean(out)
}

test_that("scfast_harmony removes batch shifts and keeps cell types apart", {
  x <- harmony_blobs()
  res <- skip_if_stub(scfast_harmony(x$emb, x$batch))
  expect_equal(dim(res$embeddings), dim(x$emb))
  expect_identical(dimnames(res$embeddings), dimnames(x$emb))
  before <- batch_spread(x$emb, x$batch, x$type)
  after <- batch_spread(res$embeddings, x$batch, x$type)
  expect_lt(after, 0.2 * before)
  centres <- t(sapply(1:3, function(t) colMeans(res$embeddings[x$type == t, ])))
  expect_gt(min(stats::dist(centres)), 5 * after)
  expect_identical(res$nclust, 30L) # round(900 / 30)
  expect_length(res$objective, res$iters + 1L)
  expect_identical(res$device, "cpu")
  # same seed, same result; factor and character labels are the same partition
  again <- scfast_harmony(x$emb, factor(x$batch), lambda = NULL)
  expect_identical(again$embeddings, res$embeddings)
  expect_false(identical(scfast_harmony(x$emb, x$batch, seed = 2)$embeddings, res$embeddings))
  fixed <- scfast_harmony(x$emb, x$batch, lambda = 1, nclust = 5, max.iter = 3, early.stop = FALSE,
                          .options = list(max.iter.cluster = 6))
  expect_identical(fixed$iters, 3L)
  expect_identical(fixed$nclust, 5L)
  init <- scfast_harmony(x$emb, x$batch, init.centroids = x$emb[1:4, ])
  expect_identical(init$nclust, 4L)
})

test_that("scfast_harmony rejects bad arguments", {
  x <- harmony_blobs(n_per = 10)
  expect_error(scfast_harmony(x$emb, x$batch[-1]), "labels for")
  expect_error(scfast_harmony(x$emb, replace(x$batch, 2, NA)), "NA")
  expect_error(scfast_harmony(x$emb, x$batch, lambda = 0), "lambda")
  expect_error(scfast_harmony(x$emb, x$batch, theta = -1), "theta", class = "scfast_error")
  expect_error(scfast_harmony(x$emb, x$batch, nclust = 0), "nclust")
  expect_error(scfast_harmony(x$emb, x$batch, .options = list(foo = 1)), "unknown")
  expect_error(scfast_harmony(x$emb, x$batch, init.centroids = x$emb[1:3, 1:2]), "init.centroids")
  expect_error(scfast_harmony(x$emb[1:5, ], x$batch[1:5]), "at least 6 cells", class = "scfast_error")
  bad <- x$emb
  bad[2, 3] <- NaN
  expect_error(scfast_harmony(bad, x$batch), "non-finite", class = "scfast_error")
  expect_error(scfast_harmony(x$emb, x$batch, device = "cuda"))
})

test_that("RunHarmonyFast stores a RunHarmony-style DimReduc", {
  obj <- toy_object()
  o <- skip_if_stub(RunPCAFast(obj, npcs = 10, verbose = FALSE))
  o$batch <- rep(c("a", "b"), length.out = ncol(o))
  h <- RunHarmonyFast(o, "batch", dims.use = 1:8, nclust = 5)
  red <- h[["harmony"]]
  expect_s4_class(red, "DimReduc")
  expect_true(methods::validObject(red))
  emb <- SeuratObject::Embeddings(red)
  expect_equal(dim(emb), c(ncol(o), 8))
  expect_identical(rownames(emb), colnames(o))
  expect_identical(colnames(emb), paste0("harmony_", 1:8))
  expect_identical(SeuratObject::Key(red), "harmony_")
  expect_identical(SeuratObject::DefaultAssay(red), "RNA")
  expect_equal(SeuratObject::Stdev(red), unname(apply(emb, 2, stats::sd)))
  info <- red@misc$scfast
  expect_identical(info$kernel, "harmony")
  expect_identical(info$params$nclust, 5L)
  expect_true(is.logical(info$converged))
  expect_length(info$objective, info$iters + 1L)
  expect_true(any(startsWith(names(h@commands), "RunHarmonyFast")))
  # the same as the low-level call on the same columns
  low <- scfast_harmony(SeuratObject::Embeddings(o, "pca")[, 1:8], o$batch, nclust = 5)
  expect_equal(unname(emb), unname(low$embeddings))
  h2 <- RunHarmonyFast(o, "batch", reduction.save = "hfast")
  expect_identical(SeuratObject::Key(h2[["hfast"]]), "hfast_")
  expect_error(RunHarmonyFast(o, "nope"), "meta data")
  expect_error(RunHarmonyFast(o, "batch", reduction = "umap"), "not found")
  expect_error(RunHarmonyFast(o, "batch", dims.use = 1), "one dimension")
  expect_error(RunHarmonyFast(o, c("batch", "batch")), "one meta data column")
  # downstream Seurat steps accept it
  h3 <- FindNeighborsFast(h, reduction = "harmony", dims = 1:8, verbose = FALSE)
  expect_true("RNA_snn" %in% SeuratObject::Graphs(h3))
})
