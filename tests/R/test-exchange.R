# Unit tests for R/exchange/exchange.R (docs/exchange-spec.md v1.0).
#
#   Rscript -e 'testthat::test_file("tests/R/test-exchange.R")'   # from repo root
#
# Python (python3 with anndata, numpy, scipy, h5py) is used to prove that R
# and Python agree; those tests are skipped if python3/anndata is missing.

suppressPackageStartupMessages({
  library(testthat)
  library(Matrix)
  library(Seurat)
  library(hdf5r)
})

.find_repo <- function() {
  d <- normalizePath(getwd())
  for (i in 1:6) {
    if (file.exists(file.path(d, "R", "exchange", "exchange.R"))) return(d)
    d <- dirname(d)
  }
  stop("cannot find R/exchange/exchange.R above ", getwd())
}
REPO <- .find_repo()
source(file.path(REPO, "R", "exchange", "exchange.R"))

PY <- Sys.getenv("PYTHON", "python3")
has_python <- local({
  ok <- tryCatch(system2(PY, c("-c", shQuote("import anndata, numpy, scipy, h5py")),
                         stdout = FALSE, stderr = FALSE) == 0, error = function(e) FALSE)
  isTRUE(ok)
})
skip_if_no_python <- function() skip_if_not(has_python, "python3 with anndata not available")

# Run a Python snippet with UserWarnings as errors; returns stdout lines and
# fails the test on a non-zero exit (stderr is shown).
run_py <- function(code, args = character()) {
  f <- tempfile(fileext = ".py")
  # -I (isolated) ignores PYTHONPATH, so put python/ (for cellwarden) on sys.path here.
  writeLines(c(sprintf("import sys; sys.path.insert(0, %s)", deparse(file.path(REPO, "python"))), code), f)
  out <- tempfile(); err <- tempfile()
  st <- system2(PY, c("-I", "-W", "error::UserWarning", shQuote(f), shQuote(args)),
                stdout = out, stderr = err)
  so <- readLines(out, warn = FALSE); se <- readLines(err, warn = FALSE)
  if (st != 0) fail(paste0("python exited ", st, ":\n", paste(c(so, se), collapse = "\n")))
  so
}

tmp_h5ad <- function(tag) file.path(tempdir(), sprintf("scx-%s-%s.h5ad", tag, basename(tempfile(""))))

# ---------------------------------------------------------------- synthetic data
make_seurat <- function(n_cells = 200, n_genes = 300, seed = 42, with_pca = TRUE) {
  set.seed(seed)
  m <- Matrix::rsparsematrix(n_genes, n_cells, density = 0.15,
                             rand.x = function(n) as.numeric(rpois(n, 3) + 1))
  orig <- sprintf("GENE%03d", seq_len(n_genes))
  orig[c(3, 10, 77)] <- c("LINC_00486", "RP11_34P13.7", "TRBV20_1")
  cells <- sprintf("CELL%04d-1", seq_len(n_cells))
  dimnames(m) <- list(orig, cells)
  m <- as(m, "dgCMatrix")
  meta <- data.frame(
    sample = factor(sample(c("donor_B", "donor_A"), n_cells, TRUE), levels = c("donor_B", "donor_A")),
    grade = factor(sample(c("low", "mid", "high"), n_cells, TRUE), levels = c("low", "mid", "high"), ordered = TRUE),
    label = sample(c("x", "y", "z"), n_cells, TRUE),
    is_qc = sample(c(TRUE, FALSE), n_cells, TRUE),
    n_umi_int = as.integer(colSums(m)),
    score = rnorm(n_cells),
    row.names = cells, stringsAsFactors = FALSE)
  meta$sample[5] <- NA                   # categorical NA -> code -1
  obj <- suppressWarnings(CreateSeuratObject(m, meta.data = meta))
  fm <- data.frame(gene_symbol_original = orig, gene_id = sprintf("ENSG%011d", seq_len(n_genes)),
                   row.names = rownames(obj))
  obj[["RNA"]] <- AddMetaData(obj[["RNA"]], metadata = fm)
  obj <- NormalizeData(obj, verbose = FALSE)
  obj <- FindVariableFeatures(obj, nfeatures = 100, verbose = FALSE)
  if (with_pca) {
    obj <- ScaleData(obj, verbose = FALSE)
    obj <- suppressWarnings(RunPCA(obj, npcs = 10, verbose = FALSE))
    obj <- FindNeighbors(obj, dims = 1:10, verbose = FALSE)
  }
  obj
}

OBJ <- make_seurat()
COUNTS <- LayerData(OBJ, layer = "counts")

# ================================================================ checksum
test_that("int64 LE encoder matches known bytes", {
  expect_identical(.scx_le_int64(c(0, 1, 2^32 + 5, 2^53 - 1)),
                   as.raw(c(0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0, 5,0,0,0,1,0,0,0,
                            0xff,0xff,0xff,0xff,0xff,0xff,0x1f,0)))
  expect_error(.scx_le_int64(-1))
})

test_that("checksum is independent of internal storage", {
  cs <- sc_counts_checksum(COUNTS)
  expect_match(cs, "^[0-9a-f]{64}$")
  expect_identical(sc_counts_checksum(COUNTS), cs)                          # stable
  expect_identical(sc_counts_checksum(as.matrix(COUNTS)), cs)               # dense
  expect_identical(sc_counts_checksum(as(COUNTS, "TsparseMatrix")), cs)     # triplet
  expect_identical(sc_counts_checksum(as(COUNTS, "RsparseMatrix")), cs)     # row-compressed
  # explicit zeros stored
  ez <- COUNTS; ez@x[c(1, 10, 100)] <- 0
  ez_ref <- Matrix::drop0(ez)
  expect_identical(sc_counts_checksum(ez), sc_counts_checksum(ez_ref))
  # unsorted, duplicated triplets summing to the same matrix
  tt <- as(COUNTS, "TsparseMatrix")
  o <- rev(seq_along(tt@x))
  half <- tt@x / 2
  dup <- new("dgTMatrix", i = c(tt@i[o], tt@i), j = c(tt@j[o], tt@j), x = c(half[o], half),
             Dim = tt@Dim, Dimnames = tt@Dimnames)
  expect_identical(sc_counts_checksum(dup), cs)
  # integer-valued storage
  im <- as.matrix(COUNTS); storage.mode(im) <- "integer"
  expect_identical(sc_counts_checksum(im), cs)
  # dimnames do not matter, values and cell order do
  nn <- COUNTS; dimnames(nn) <- NULL
  expect_identical(sc_counts_checksum(nn), cs)
  ch <- COUNTS; ch[1, 1] <- ch[1, 1] + 1
  expect_false(identical(sc_counts_checksum(ch), cs))
  expect_false(identical(sc_counts_checksum(COUNTS[, rev(seq_len(ncol(COUNTS)))]), cs))
})

test_that("checksum of a tiny matrix agrees with an independent numpy+hashlib computation", {
  skip_if_no_python()
  # genes x cells (Seurat). Cells x genes CSR:
  #   cell1: g1=1, g3=2 ; cell2: (empty) ; cell3: g2=5, g3=0.5 ; cell4: g1=7
  m <- Matrix::sparseMatrix(i = c(1, 3, 2, 3, 1), j = c(1, 1, 3, 3, 4),
                            x = c(1, 2, 5, 0.5, 7), dims = c(3, 4))
  py <- run_py(c(
    "import hashlib, struct, numpy as np",
    "indptr = np.array([0, 2, 2, 4, 5]); indices = np.array([0, 2, 1, 2, 0]); data = np.array([1, 2, 5, 0.5, 7.0])",
    "h = hashlib.sha256()",
    "h.update(struct.pack('<qq', 4, 3))",
    "h.update(indptr.astype('<i8').tobytes()); h.update(indices.astype('<i8').tobytes()); h.update(data.astype('<f8').tobytes())",
    "print(h.hexdigest())"))
  expect_identical(sc_counts_checksum(m), py[[length(py)]])
  # also by hand in R from the byte layout of the spec
  raw_ref <- c(.scx_le_int64(c(4, 3)), .scx_le_int64(c(0, 2, 2, 4, 5)), .scx_le_int64(c(0, 2, 1, 2, 0)),
               writeBin(c(1, 2, 5, 0.5, 7), raw(), size = 8, endian = "little"))
  expect_identical(sc_counts_checksum(m), digest::digest(raw_ref, algo = "sha256", serialize = FALSE))
})

test_that("checksum matches cellwarden (Python) on the synthetic counts and the shared fixture", {
  skip_if_no_python()
  skip_if_not(file.exists(file.path(REPO, "python", "cellwarden", "exchange.py")), "python/cellwarden not present")
  f <- tempfile(fileext = ".mtx")
  Matrix::writeMM(COUNTS, f)
  py <- run_py(c(
    "import sys, scipy.io, scipy.sparse as sp",
    "from cellwarden.exchange import counts_checksum",
    "m = sp.csr_matrix(scipy.io.mmread(sys.argv[1]).T)",
    "print(counts_checksum(m))"), f)
  expect_identical(sc_counts_checksum(COUNTS), py[[length(py)]])

  fx <- file.path(REPO, "tests", "fixtures", "pbmc_like")
  skip_if_not(file.exists(file.path(fx, "expected.json")), "fixture not present")
  exp <- jsonlite::fromJSON(file.path(fx, "expected.json"))
  mm <- as(Matrix::readMM(file.path(fx, "matrix.mtx")), "CsparseMatrix")
  expect_identical(sc_counts_checksum(mm), exp$counts_checksum)
})

# ================================================================ full mode
test_that("full write -> anndata reads with no warnings -> R round trip", {
  p <- tmp_h5ad("full")
  expect_warning(write_exchange(OBJ, p, mode = "full", run_id = "t1", stage = "UT_full", config_hash = "abc"),
                 NA)
  expect_identical(validate_exchange(p), character(0))

  # Low-level encodings
  h <- H5File$new(p, "r")
  expect_identical(h$attr_open("encoding-type")$read(), "anndata")
  expect_identical(h[["X"]]$attr_open("encoding-type")$read(), "csr_matrix")
  expect_identical(h[["layers/counts"]]$attr_open("encoding-type")$read(), "csr_matrix")
  expect_identical(as.character(h[["layers/counts/data"]]$get_type()$to_text()), "H5T_STD_I32LE")
  expect_identical(as.character(h[["X/data"]]$get_type()$to_text()), "H5T_IEEE_F32LE")
  expect_identical(h[["obs/label"]]$attr_open("encoding-type")$read(), "string-array")
  expect_identical(h[["obs/_index"]]$attr_open("encoding-type")$read(), "string-array")
  expect_identical(h[["uns/exchange/n_cells"]]$attr_open("encoding-type")$read(), "numeric-scalar")
  expect_identical(h[["uns/exchange/mode"]]$attr_open("encoding-type")$read(), "string")
  h$close_all()

  skip_if_no_python()
  py <- run_py(c(
    "import sys, warnings, json, anndata as ad, numpy as np, pandas as pd",
    "warnings.simplefilter('error')",
    "a = ad.read_h5ad(sys.argv[1])",
    "ex = a.uns['exchange']",
    "out = dict(shape=list(a.shape), sample_levels=list(a.obs['sample'].cat.categories),",
    "  sample_na=int(a.obs['sample'].isna().sum()), grade_ordered=bool(a.obs['grade'].cat.ordered),",
    "  dtypes={c: str(a.obs[c].dtype) for c in a.obs.columns},",
    "  sym=list(a.var['gene_symbol_original'][:12]), hv=int(a.var['highly_variable'].sum()),",
    "  obsm=sorted(a.obsm.keys()), varm={k: list(v.shape) for k, v in a.varm.items()},",
    "  obsp=sorted(a.obsp.keys()), counts_dtype=str(a.layers['counts'].dtype),",
    "  ex={k: (v if isinstance(v, (str, dict)) else int(v)) for k, v in ex.items()})",
    "try:",
    "    from cellwarden.exchange import validate_exchange, counts_checksum",
    "    out['py_problems'] = validate_exchange(sys.argv[1])",
    "    out['py_checksum'] = counts_checksum(a.layers['counts'])",
    "except ImportError:",
    "    pass",
    "print(json.dumps(out))"), p)
  r <- jsonlite::fromJSON(py[[length(py)]], simplifyVector = TRUE)
  expect_equal(r$shape, c(200, 300))
  expect_identical(r$sample_levels, c("donor_B", "donor_A"))
  expect_equal(r$sample_na, 1)
  expect_true(r$grade_ordered)
  expect_identical(r$dtypes$sample, "category")
  expect_identical(r$dtypes$is_qc, "bool")
  expect_identical(r$dtypes$n_umi_int, "int32")
  expect_identical(r$dtypes$score, "float64")
  expect_true(r$dtypes$label %in% c("object", "str", "string"))
  expect_identical(r$sym[c(3, 10)], c("LINC_00486", "RP11_34P13.7"))
  expect_equal(r$hv, 100)
  expect_true(all(c("X_pca") %in% r$obsm))
  expect_equal(r$varm$PCs, c(300, 10))
  expect_identical(r$obsp, c("connectivities", "distances"))
  expect_identical(r$counts_dtype, "int32")
  expect_identical(r$ex$writer, "R")
  expect_identical(r$ex$mode, "full")
  expect_identical(r$ex$run_id, "t1")
  expect_identical(r$ex$config_hash, "abc")
  expect_equal(r$ex$n_cells, 200)
  expect_identical(r$ex$counts_checksum, sc_counts_checksum(COUNTS))
  expect_true(all(c("R", "Seurat", "SeuratObject", "hdf5r") %in% names(r$ex$packages)))
  if (!is.null(r$py_checksum)) {
    expect_identical(r$py_checksum, sc_counts_checksum(COUNTS))
    expect_length(r$py_problems, 0)
  }
})

test_that("read_exchange_seurat restores counts, data, meta.data, features, reductions, graphs", {
  p <- tmp_h5ad("full-rt")
  write_exchange(OBJ, p, mode = "full")
  rd <- read_exchange_seurat(p)
  expect_s4_class(rd, "Seurat")
  expect_identical(colnames(rd), colnames(OBJ))
  expect_identical(rownames(rd), rownames(OBJ))
  expect_identical(sc_counts_checksum(LayerData(rd, layer = "counts")), sc_counts_checksum(COUNTS))
  d0 <- LayerData(OBJ, layer = "data"); d1 <- LayerData(rd, layer = "data")
  expect_lt(max(abs(d0 - d1)), 1e-5)                      # float32 round trip
  md0 <- OBJ[[]]; md1 <- rd[[]]
  for (cn in colnames(md0)) {
    expect_true(cn %in% colnames(md1), info = cn)
    expect_identical(class(md1[[cn]]), class(md0[[cn]]), info = cn)
    if (is.factor(md0[[cn]])) expect_identical(levels(md1[[cn]]), levels(md0[[cn]]), info = cn)
    if (is.double(md0[[cn]])) expect_equal(md1[[cn]], md0[[cn]], info = cn)
    else expect_identical(md1[[cn]], md0[[cn]], info = cn)
  }
  fm <- rd[["RNA"]][[]]
  expect_identical(as.character(fm$gene_symbol_original), as.character(OBJ[["RNA"]][[]]$gene_symbol_original))
  expect_identical(fm$gene_symbol_original[77], "TRBV20_1")
  expect_identical(rownames(rd)[77], "TRBV20-1")
  expect_identical(sort(VariableFeatures(rd)), sort(VariableFeatures(OBJ)))
  expect_true("pca" %in% Reductions(rd))
  expect_identical(Key(rd[["pca"]]), "PC_")
  expect_lt(max(abs(Embeddings(rd, "pca") - Embeddings(OBJ, "pca"))), 1e-4)
  L <- Loadings(rd, "pca")
  expect_identical(dim(L), c(300L, 10L))
  expect_lt(max(abs(L[rownames(Loadings(OBJ, "pca")), ] - Loadings(OBJ, "pca"))), 1e-5)
  expect_true(all(c("RNA_snn", "RNA_nn") %in% Graphs(rd)))
  expect_lt(max(abs(rd[["RNA_snn"]] - OBJ[["RNA_snn"]])), 1e-6)
  expect_identical(rownames(rd[["RNA_snn"]]), colnames(OBJ))
  # plain list reader
  lst <- read_exchange(p)
  expect_identical(dim(lst$X), c(300L, 200L))
  expect_identical(names(lst$layers), "counts")
  expect_identical(dim(lst$obsm$X_pca), c(200L, 10L))
  expect_s4_class(lst$obsp$connectivities, "dgCMatrix")
  expect_identical(lst$uns$exchange$writer, "R")
})

test_that("type mapping: character NA becomes categorical with a warning; empty data layer falls back to counts", {
  o <- make_seurat(60, 80, seed = 3, with_pca = FALSE)
  o$note <- ifelse(seq_len(ncol(o)) %% 7 == 0, NA, "ok")
  p <- tmp_h5ad("na")
  expect_warning(write_exchange(o, p, "full"), "missing values")
  expect_identical(validate_exchange(p), character(0))
  lst <- read_exchange(p)
  expect_true(is.factor(lst$obs$note))
  expect_identical(unname(is.na(lst$obs$note)), unname(is.na(o$note)))
  # no data layer (CreateSeuratObject without NormalizeData has counts only)
  o2 <- suppressWarnings(CreateSeuratObject(LayerData(o, layer = "counts")))
  expect_identical(Layers(o2[["RNA"]]), "counts")
  p2 <- tmp_h5ad("nodata")
  expect_warning(write_exchange(o2, p2, "full"), "no data layer")
  expect_identical(validate_exchange(p2), character(0))
  lst2 <- read_exchange(p2)
  expect_match(lst2$uns$exchange$notes, "raw counts")
  expect_equal(max(abs(lst2$X - LayerData(o2, layer = "counts"))), 0)
})

# ================================================================ slim mode
test_that("slim write holds only HVGs, no counts, and the source checksum", {
  p <- tmp_h5ad("slim")
  write_exchange(OBJ, p, mode = "slim", run_id = "t2", stage = "UT_slim")
  expect_identical(validate_exchange(p), character(0))
  lst <- read_exchange(p)
  hv <- VariableFeatures(OBJ)
  expect_identical(nrow(lst$X), length(hv))
  expect_setequal(rownames(lst$X), hv)
  expect_identical(rownames(lst$X), rownames(OBJ)[rownames(OBJ) %in% hv])   # object order
  expect_length(lst$layers, 0)
  expect_length(lst$varm, 0)
  expect_true(all(lst$var$highly_variable))
  expect_identical(lst$uns$exchange$mode, "slim")
  expect_null(lst$uns$exchange$counts_checksum)
  expect_identical(lst$uns$exchange$source_counts_checksum, sc_counts_checksum(COUNTS))
  expect_lt(max(abs(lst$X - LayerData(OBJ, layer = "data")[rownames(lst$X), ])), 1e-5)

  noh <- NormalizeData(suppressWarnings(CreateSeuratObject(COUNTS)), verbose = FALSE)
  expect_length(VariableFeatures(noh), 0)
  expect_error(write_exchange(noh, tmp_h5ad("slim-nohvg"), mode = "slim"), "VariableFeatures")

  skip_if_no_python()
  py <- run_py(c("import sys, warnings, anndata as ad", "warnings.simplefilter('error')",
                 "a = ad.read_h5ad(sys.argv[1])", "print(a.n_obs, a.n_vars, len([k for k in a.layers.keys() if k is not None]), int(a.var['highly_variable'].sum()))"), p)
  expect_identical(py[[length(py)]], "200 100 0 100")
})

# ================================================================ validation
test_that("validate_exchange catches tampered files", {
  p <- tmp_h5ad("tamper")
  write_exchange(OBJ, p, mode = "full")
  replace_scalar <- function(path, name, value, dtype) {
    h <- H5File$new(path, "r+")
    g <- h[["uns/exchange"]]
    g$link_delete(name)
    d <- g$create_dataset(name, value, dtype = dtype, space = H5S$new("scalar"), chunk_dims = NULL)
    .scx_enc(d, if (is.character(value)) "string" else "numeric-scalar", "0.2.0")
    h$close_all()
  }
  p1 <- tmp_h5ad("t-ncells"); file.copy(p, p1)
  replace_scalar(p1, "n_cells", 199, h5types$H5T_NATIVE_INT64)
  expect_true(any(grepl("n_cells", validate_exchange(p1))))

  p2 <- tmp_h5ad("t-cs"); file.copy(p, p2)
  replace_scalar(p2, "counts_checksum", strrep("0", 64), .scx_str_type())
  expect_true(any(grepl("counts_checksum does not match", validate_exchange(p2))))

  p3 <- tmp_h5ad("t-mode"); file.copy(p, p3)
  replace_scalar(p3, "schema_version", "2.0", .scx_str_type())
  expect_true(any(grepl("unsupported schema_version", validate_exchange(p3))))
  expect_error(read_exchange(p3), "schema_version")

  p4 <- tmp_h5ad("t-key"); file.copy(p, p4)
  h <- H5File$new(p4, "r+"); h[["uns/exchange"]]$link_delete("cells_sha256"); h$close_all()
  expect_true(any(grepl("missing required keys: cells_sha256", validate_exchange(p4))))

  expect_match(validate_exchange(file.path(tempdir(), "nope.h5ad")), "file not found")
  expect_error(read_exchange_seurat(p1), "n_cells")
})

# ================================================================ merge
# Results file built in Python with plain anndata (not via cellwarden), per spec.
make_results <- function(obj, path, source_checksum, drop = integer(), extra = character()) {
  bc <- tempfile(); writeLines(c(colnames(obj)[setdiff(seq_len(ncol(obj)), drop)], extra), bc)
  run_py(c(
    "import sys, hashlib, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad",
    "from datetime import datetime, timezone",
    "bcs = [l.rstrip('\\n') for l in open(sys.argv[1])]",
    "n = len(bcs); rng = np.random.default_rng(0)",
    "obs = pd.DataFrame(index=pd.Index(bcs, dtype=object))",
    "obs['leiden'] = pd.Categorical([str(i % 4) for i in range(n)], categories=['0','1','2','3'])",
    "conn = sp.random(n, n, density=0.05, random_state=1, format='csr', dtype=np.float32)",
    "a = ad.AnnData(X=sp.csr_matrix((n, 0), dtype=np.float32), obs=obs,",
    "               var=pd.DataFrame(index=pd.Index([], dtype=object)),",
    "               obsm={'X_pca': rng.normal(size=(n, 10)).astype(np.float32),",
    "                     'X_umap': np.column_stack([np.arange(n), -np.arange(n)]).astype(np.float32),",
    "                     'X_harmony': rng.normal(size=(n, 10)).astype(np.float32),",
    "                     'X_tsne': rng.normal(size=(n, 2)).astype(np.float32)},",
    "               obsp={'connectivities': conn, 'distances': conn.copy()})",
    "sha = lambda xs: hashlib.sha256('\\n'.join(xs).encode('utf-8')).hexdigest()",
    "a.uns['exchange'] = dict(schema_version='1.0', mode='results', run_id='t3', stage='S2_gpu_core',",
    "  writer='python', created_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),",
    "  config_hash='none', n_cells=np.int64(n), n_genes=np.int64(0), cells_sha256=sha(bcs),",
    "  genes_sha256=sha([]), source_counts_checksum=sys.argv[3], cluster_key='leiden',",
    "  packages=dict(anndata=ad.__version__ if hasattr(ad, '__version__') else 'x'))",
    "with ad.settings.override(allow_write_nullable_strings=False):",
    "    a.write_h5ad(sys.argv[2], convert_strings_to_categoricals=False)",
    "print('ok')"), c(bc, path, source_checksum))
  invisible(path)
}

test_that("merge_exchange_results adds reductions, graphs and clusters by barcode", {
  skip_if_no_python()
  cs <- sc_counts_checksum(COUNTS)
  base <- OBJ
  base[["pca"]] <- NULL
  base@graphs <- list()
  rp <- make_results(base, tmp_h5ad("res"), cs)
  expect_identical(validate_exchange(rp), character(0))

  m <- merge_exchange_results(base, rp)
  expect_identical(colnames(m), colnames(base))
  expect_true(all(c("pca", "umap", "harmony", "tsne") %in% Reductions(m)))
  expect_identical(Key(m[["pca"]]), "PC_")
  expect_identical(Key(m[["umap"]]), "UMAP_")
  expect_identical(Key(m[["harmony"]]), "harmony_")
  expect_identical(Key(m[["tsne"]]), "TSNE_")
  expect_identical(dim(Embeddings(m, "pca")), c(200L, 10L))
  expect_equal(unname(Embeddings(m, "umap")[, 1]), 0:199)
  expect_true(all(c("RNA_snn", "RNA_nn") %in% Graphs(m)))
  expect_identical(dim(m[["RNA_snn"]]), c(200L, 200L))
  expect_identical(DefaultAssay(m[["RNA_snn"]]), "RNA")
  expect_true(is.factor(m$leiden))
  expect_identical(levels(m$leiden), c("0", "1", "2", "3"))
  expect_identical(as.character(m$leiden), as.character((0:199) %% 4))
  expect_identical(as.character(m$seurat_clusters), as.character(m$leiden))
  expect_identical(unname(as.character(Idents(m))), as.character(m$leiden))
  expect_identical(sc_counts_checksum(LayerData(m, layer = "counts")), cs)

  # re-merge needs overwrite
  expect_error(merge_exchange_results(m, rp), "overwrite")
  expect_s4_class(merge_exchange_results(m, rp, overwrite = TRUE), "Seurat")

  # barcode matching, not position: same cells, reversed order in the object
  rev_counts <- COUNTS[, rev(colnames(COUNTS))]
  shuffled <- suppressWarnings(CreateSeuratObject(rev_counts))
  ms <- merge_exchange_results(shuffled, rp)
  expect_identical(colnames(ms), rev(colnames(base)))
  expect_equal(Embeddings(ms, "umap")[colnames(base), ], Embeddings(m, "umap")[colnames(base), ])
  expect_identical(as.character(ms$leiden[colnames(base)]), as.character(m$leiden[colnames(base)]))
  expect_identical(rownames(ms[["RNA_snn"]]), colnames(ms))
  expect_equal(ms[["RNA_snn"]][colnames(base), colnames(base)], m[["RNA_snn"]][colnames(base), colnames(base)])
})

test_that("merge_exchange_results refuses a mismatched checksum and missing/unknown barcodes", {
  skip_if_no_python()
  cs <- sc_counts_checksum(COUNTS)
  bad <- make_results(OBJ, tmp_h5ad("res-bad"), paste0(ifelse(substr(cs, 1, 1) == "0", "1", "0"), substring(cs, 2)))
  expect_error(merge_exchange_results(OBJ, bad), "source_counts_checksum mismatch")
  expect_error(merge_exchange_results(OBJ, bad, overwrite = TRUE, allow_subset = TRUE), "source_counts_checksum mismatch")

  sub <- make_results(OBJ, tmp_h5ad("res-sub"), cs, drop = 8L)
  expect_error(merge_exchange_results(OBJ, sub, overwrite = TRUE), "missing from the results")
  ms <- expect_warning(merge_exchange_results(OBJ, sub, overwrite = TRUE, allow_subset = TRUE),
                       "1 object cell\\(s\\) absent")
  expect_identical(colnames(ms), colnames(OBJ))
  expect_true(is.na(ms$leiden[[8]]))
  expect_false(anyNA(ms$leiden[-8]))
  expect_identical(nrow(Embeddings(ms, "umap")), 199L)

  unk <- make_results(OBJ, tmp_h5ad("res-unk"), cs, extra = "NOT_A_CELL-1")
  expect_error(merge_exchange_results(OBJ, unk, overwrite = TRUE, allow_subset = TRUE), "not in the object")

  # a full-mode file is not a results file
  fp <- tmp_h5ad("not-res"); write_exchange(OBJ, fp, "full")
  expect_error(merge_exchange_results(OBJ, fp), "results-mode")
})
