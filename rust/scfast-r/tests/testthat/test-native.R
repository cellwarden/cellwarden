# The native layer on its own: loading, panics -> R errors, input validation in Rust.
# None of these need a working kernel.

test_that("the package loads and reports the engine", {
  info <- scfast_info()
  expect_type(info, "list")
  expect_identical(info$package, as.character(utils::packageVersion("scfastR")))
  expect_true(is.character(info$version) && nzchar(info$version))
  expect_true(is.integer(info$threads) && info$threads >= 1L)
  # default build: no CUDA feature, so "auto" resolves to the CPU
  expect_identical(info$cuda, info$cuda_built)
  expect_true(is.logical(info$cuda_built) && is.logical(info$cuda_available))
  if (!info$cuda_built) {
    expect_false(info$cuda_available)
    expect_true(is.character(info$cuda_error))
    expect_null(info$device_name)
  }
  expect_identical(as.character(scfastR:::.check_device("auto")), if (info$cuda_available) "cuda:0" else "cpu")
  expect_identical(as.character(scfastR:::.check_device("cpu")), "cpu")
  expect_identical(attr(scfastR:::.check_device("auto"), "requested"), "auto")
})

test_that("a Rust panic becomes an R error and R keeps running", {
  expect_error(scfastR:::rs_debug_panic("boom"), "scfast debug: boom", fixed = TRUE)
  # many times over: no crash, no corrupted state
  for (i in 1:50) expect_error(scfastR:::rs_debug_panic(paste("panic", i)), paste("panic", i))
  expect_true(scfast_info()$threads >= 1L)
})

test_that("errors are re-signalled with scfast classes", {
  e <- tryCatch(scfastR:::.rust(scfastR:::rs_debug_panic("x")), error = identity)
  expect_s3_class(e, "scfast_error")
  expect_false(inherits(e, "scfast_not_implemented"))
  # what a stub kernel (unimplemented!()) produces
  e <- tryCatch(scfastR:::.rust(scfastR:::rs_debug_panic("not implemented: knn: Phase 4")), error = identity)
  expect_s3_class(e, "scfast_not_implemented")
  expect_s3_class(e, "scfast_error")
})

test_that("stub kernels surface as scfast_not_implemented, real ones return results", {
  m <- Matrix::rsparsematrix(30, 20, 0.3)
  m@x <- abs(m@x)
  r <- tryCatch(scfast_pca(m, npcs = 3), scfast_not_implemented = function(e) "stub")
  expect_true(identical(r, "stub") || is.list(r))
})

test_that("the CSR view rejects malformed sparse arrays before any kernel runs", {
  m <- Matrix::sparseMatrix(i = c(1, 3, 2), j = c(1, 1, 3), x = c(1, 2, 3), dims = c(4, 3))
  f <- function(p = m@p, i = m@i, x = m@x, ng = nrow(m), nc = ncol(m)) {
    scfastR:::rs_csr_view_ptrs(p, i, x, ng, nc)
  }
  expect_type(f(), "list")
  expect_error(f(p = as.numeric(m@p)), "@p must be an integer vector")
  expect_error(f(x = as.integer(m@x)), "@x must be a double vector")
  expect_error(f(p = m@p[-1]), "indptr has")
  expect_error(f(i = c(m@i[-3], 9L)), "out of range")
  expect_error(f(i = c(m@i[-3], -1L)), "negative")
  expect_error(f(x = c(1, NA, 3)), "non-finite")
  expect_error(f(x = c(1, Inf, 3)), "non-finite")
  expect_error(f(x = m@x[-1]), "differ in length")
  expect_error(f(ng = -1L), ">= 0")
  # the same checks guard the PCA entry point
  expect_error(scfastR:::rs_pca_csc(m@p[-1], m@i, m@x, 4L, 3L, 2L, 0, -1L, -1L, "cpu"), "indptr has")
  expect_error(scfastR:::rs_pca_csc(m@p, m@i, m@x, 4L, 3L, 0L, 0, -1L, -1L, "cpu"), "npcs must be >= 1")
  expect_error(scfastR:::rs_pca_csc(m@p, m@i, m@x, 4L, 3L, 2L, -1, -1L, -1L, "cpu"), "seed")
  expect_error(scfastR:::rs_pca_csc(m@p, m@i, m@x, 4L, 3L, 2L, 0.5, -1L, -1L, "cpu"), "seed")
  # device strings are validated in Rust too
  expect_error(scfastR:::rs_pca_csc(m@p, m@i, m@x, 4L, 3L, 2L, 0, -1L, -1L, "tpu"), "device")
  expect_error(scfastR:::rs_device("npu"), "device")
  expect_identical(scfastR:::rs_device("cpu"), "cpu")
  expect_identical(scfastR:::rs_device("CPU"), "cpu")
  if (!isTRUE(scfast_info()$cuda_available)) {
    expect_error(scfastR:::rs_device("cuda"), "GPU")
    expect_identical(scfastR:::rs_device("auto"), "cpu")
    # the kernels refuse an unusable CUDA device before touching the data
    expect_error(scfastR:::rs_pca_csc(m@p, m@i, m@x, 4L, 3L, 2L, 0, -1L, -1L, "cuda"), "GPU")
    expect_error(scfastR:::rs_knn(matrix(rnorm(40), 20, 2), 20L, 2L, 5L, "exact", 0, 16L, 200L, 100L, -1, "cuda:0"), "GPU")
  }
})

test_that("auto falls back to the CPU on a GPU error, an explicit device does not", {
  calls <- character()
  fn <- function(dev) {
    calls <<- c(calls, dev)
    if (dev != "cpu") stop(structure(class = c("scfast_error", "error", "condition"),
                                     list(message = "scfast pca: GPU: out of device memory (alloc)", call = NULL)))
    "ok"
  }
  auto <- structure("cuda:0", requested = "auto")
  expect_warning(r <- scfastR:::.with_cpu_fallback(auto, fn), "falling back to the CPU")
  expect_identical(r, "ok")
  expect_identical(calls, c("cuda:0", "cpu"))
  calls <- character()
  explicit <- structure("cuda:0", requested = "cuda")
  expect_error(scfastR:::.with_cpu_fallback(explicit, fn), "GPU: out of device memory")
  expect_identical(calls, "cuda:0")
  # a non-GPU error is never retried
  fn2 <- function(dev) stop(structure(class = c("scfast_error", "error", "condition"),
                                      list(message = "scfast pca: invalid input: x", call = NULL)))
  expect_error(scfastR:::.with_cpu_fallback(auto, fn2), "invalid input")
})

test_that("kNN, SNN and Leiden validate their inputs in Rust", {
  x <- matrix(rnorm(40), 20, 2)
  expect_error(scfastR:::rs_knn(x, 20L, 3L, 5L, "exact", 0, 16L, 200L, 100L, -1, "cpu"), "expected 20 x 3")
  expect_error(scfastR:::rs_knn(x, 20L, 2L, 21L, "exact", 0, 16L, 200L, 100L, -1, "cpu"), "k must be between")
  expect_error(scfastR:::rs_knn(x, 20L, 2L, 1L, "exact", 0, 16L, 200L, 100L, -1, "cpu"), "k must be between")
  expect_error(scfastR:::rs_knn(x, 20L, 2L, 5L, "kd", 0, 16L, 200L, 100L, -1, "cpu"), "method must be one of")
  expect_error(scfastR:::rs_knn(x, 20L, 2L, 5L, "hnsw", 0, 0L, 200L, 100L, -1, "cpu"), "hnsw parameters")
  y <- x
  y[3, 1] <- NaN
  expect_error(scfastR:::rs_knn(y, 20L, 2L, 5L, "exact", 0, 16L, 200L, 100L, -1, "cpu"), "non-finite")

  idx <- matrix(c(1:4, 2L, 1L, 4L, 3L), 4, 2)
  expect_error(scfastR:::rs_snn(idx, 4L, 3L, 1 / 15), "expected 4 x 3")
  bad <- idx
  bad[2, 2] <- 5L
  expect_error(scfastR:::rs_snn(bad, 4L, 2L, 1 / 15), "outside 1..4")
  expect_error(scfastR:::rs_snn(idx, 4L, 2L, 1.5), "prune must be in")

  g <- Matrix::sparseMatrix(i = c(1, 2), j = c(2, 1), x = c(1, 1), dims = c(3, 3))
  expect_error(scfastR:::rs_leiden(g@p[-1], g@i, g@x, 3L, 0.5, 0, 2L), "inconsistent")
  expect_error(scfastR:::rs_leiden(g@p, g@i, g@x, 3L, -1, 0, 2L), "resolution must be >= 0")
  expect_error(scfastR:::rs_leiden(g@p, g@i, g@x, 3L, 0.5, 0, 0L), "n_iter must be >= 1")
  expect_error(scfastR:::rs_leiden(g@p, g@i, -g@x, 3L, 0.5, 0, 2L), "non-negative")
})

test_that("rs_is_symmetric detects asymmetry", {
  s <- Matrix::sparseMatrix(i = c(1, 2, 2, 3, 1), j = c(2, 1, 3, 2, 1), x = c(0.5, 0.5, 0.2, 0.2, 1), dims = c(3, 3))
  sym <- function(g) scfastR:::rs_is_symmetric(g@p, g@i, g@x, ncol(g), 1e-6)
  expect_true(sym(s))
  a <- s
  a[1, 2] <- 0.7
  expect_false(sym(a))
  a <- Matrix::sparseMatrix(i = c(1, 2), j = c(2, 3), x = c(1, 1), dims = c(3, 3))
  expect_false(sym(a))
  expect_true(sym(Matrix::Diagonal(5, 1) |> methods::as("CsparseMatrix") |> methods::as("generalMatrix")))
})
