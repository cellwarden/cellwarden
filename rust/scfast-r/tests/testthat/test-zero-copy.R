# docs/RUST_ENGINE.md section 3: the dgCMatrix arrays are read by Rust in place.

ptr <- function(x) scfastR:::rs_dataptr(x)

test_that("the CSR view borrows @p, @i and @x in place", {
  m <- Matrix::rsparsematrix(200, 50, 0.1)
  v <- scfastR:::rs_csr_view_ptrs(m@p, m@i, m@x, nrow(m), ncol(m))
  expect_identical(v$indptr, ptr(m@p))
  expect_identical(v$indices, ptr(m@i))
  expect_identical(v$data, ptr(m@x))
  # genes x cells CSC read as cells x genes CSR, no transpose
  expect_identical(c(v$n_rows, v$n_cols, v$nnz), c(ncol(m), nrow(m), length(m@x)))
})

test_that("slot access does not copy (what scfast_pca passes to Rust)", {
  m <- Matrix::rsparsematrix(100, 40, 0.2)
  x1 <- ptr(m@x)
  expect_identical(ptr(m@x), x1)
  expect_identical(ptr(scfastR:::.as_dgc(m, "m")@x), x1)
})

test_that("a Seurat Graph is passed to Leiden without conversion in R", {
  skip_if_no_seurat()
  z <- methods::as(abs(Matrix::rsparsematrix(20, 20, 0.2, symmetric = TRUE)), "generalMatrix")
  dimnames(z) <- list(letters[1:20], letters[1:20])
  g <- SeuratObject::as.Graph(z)
  expect_s4_class(g, "Graph")
  g2 <- scfastR:::.as_dgc(g, "graph")
  expect_identical(ptr(g2@x), ptr(g@x))
  expect_identical(ptr(g2@i), ptr(g@i))
})

test_that("RunPCAFast input: the stored data layer itself when all features are used", {
  obj <- toy_object()
  stored <- methods::slot(obj[["RNA"]], "layers")[["data"]]
  inp <- scfastR:::.pca_input(obj, "RNA", features = rownames(obj))
  expect_identical(ptr(inp$m@x), ptr(stored@x))
  expect_identical(ptr(inp$m@i), ptr(stored@i))
  expect_identical(ptr(inp$m@p), ptr(stored@p))
  expect_identical(inp$features, rownames(obj))
  expect_identical(inp$cells, colnames(obj))
})

test_that("RunPCAFast input: one copy of the variable-feature rows, nothing denser", {
  obj <- toy_object()
  hvg <- SeuratObject::VariableFeatures(obj)
  inp <- scfastR:::.pca_input(obj, "RNA")
  expect_s4_class(inp$m, "dgCMatrix")
  expect_equal(dim(inp$m), c(length(hvg), ncol(obj)))
  expect_identical(inp$features, hvg)
  ref <- SeuratObject::LayerData(obj, layer = "data")[hvg, ]
  expect_equal(unname(as.matrix(inp$m)), unname(as.matrix(ref)))
})
