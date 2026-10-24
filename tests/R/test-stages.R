# Unit tests for the R stages S0 (ingest), S1a (QC metrics) and S2 (core
# processing): R/stages/*.R, contract in docs/stages-spec.md.
#
#   Rscript -e 'testthat::test_file("tests/R/test-stages.R")'   # from repo root
#
# Inputs are built on the fly in temp dirs from tests/fixtures/pbmc_like. The
# GPU-path tests run cellwarden.gpu_core on the CPU (CELLWARDEN_ALLOW_CPU_AS_GPU=1)
# and force the plan with CELLWARDEN_PLAN_JSON; they are skipped when python3 with
# anndata + scanpy is not available.

suppressPackageStartupMessages({
  library(testthat)
  library(Matrix)
  library(Seurat)
  library(hdf5r)
})

.find_repo <- function() {
  d <- normalizePath(getwd())
  for (i in 1:6) {
    if (file.exists(file.path(d, "R", "stages", "run_stage.R"))) return(d)
    d <- dirname(d)
  }
  stop("cannot find R/stages/run_stage.R above ", getwd())
}
REPO <- .find_repo()
source(file.path(REPO, "R", "stages", "run_stage.R"))   # defines helpers; main() only runs under Rscript
scp_source_stages(REPO)
options(Seurat.object.assay.version = "v5")
FIX <- file.path(REPO, "tests", "fixtures", "pbmc_like")

PY <- Sys.getenv("CELLWARDEN_PYTHON", "python3")
has_py_core <- local({
  ok <- tryCatch(system2(PY, c("-c", shQuote("import anndata, scanpy, igraph, leidenalg")),
                         stdout = FALSE, stderr = FALSE) == 0, error = function(e) FALSE)
  isTRUE(ok)
})
has_harmonypy <- has_py_core && isTRUE(tryCatch(
  system2(PY, c("-c", shQuote("import harmonypy")), stdout = FALSE, stderr = FALSE) == 0, error = function(e) FALSE))
skip_if_no_py <- function() skip_if_not(has_py_core, "python3 with anndata/scanpy/leidenalg not available")

# ------------------------------------------------------------------ fixture
load_fixture <- function() {
  m <- as(as(readMM(file.path(FIX, "matrix.mtx")), "CsparseMatrix"), "dgCMatrix")
  feats <- read.delim(file.path(FIX, "features.tsv"), header = FALSE, colClasses = "character",
                      col.names = c("id", "symbol", "type"), quote = "")
  bcs <- readLines(file.path(FIX, "barcodes.tsv"))
  cells <- read.delim(file.path(FIX, "cells.tsv"), colClasses = "character", quote = "")
  dimnames(m) <- list(feats$id, bcs)
  list(m = m, feats = feats, bcs = bcs, cells = cells)
}
FX <- load_fixture()
SUB_G <- 1:200
SUB_C <- 1:60
fx_sub <- function(genes = SUB_G, cells = SUB_C) {
  list(m = FX$m[genes, cells, drop = FALSE], feats = FX$feats[genes, , drop = FALSE], bcs = FX$bcs[cells])
}

# ------------------------------------------------------------------ input writers
write_mtx_dir <- function(dir, x, gz = FALSE, v2 = FALSE) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  mf <- file.path(dir, "matrix.mtx")
  writeMM(x$m, mf)
  ff <- file.path(dir, if (v2) "genes.tsv" else "features.tsv")
  feats <- if (v2) x$feats[, c("id", "symbol")] else x$feats
  write.table(feats, ff, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
  bf <- file.path(dir, "barcodes.tsv")
  writeLines(x$bcs, bf)
  if (gz) for (f in c(mf, ff, bf)) { R.utils::gzip(f, overwrite = TRUE) }
  dir
}

h5_str <- function(grp, name, v) grp$create_dataset(name, as.character(v), dtype = H5T_STRING$new(size = Inf))

# 10x Cell Ranger v3 HDF5 layout; extra = optional antibody features appended.
write_10x_h5 <- function(path, x, n_ab = 0) {
  m <- x$m; ids <- x$feats$id; nm <- x$feats$symbol; ty <- x$feats$type
  if (n_ab > 0) {
    ab <- Matrix::rsparsematrix(n_ab, ncol(m), 0.5, rand.x = function(n) as.numeric(rpois(n, 5) + 1))
    m <- rbind(m, as(ab, "CsparseMatrix"))
    ids <- c(ids, paste0("AB", seq_len(n_ab))); nm <- c(nm, paste0("CD", seq_len(n_ab), "_TotalSeqB"))
    ty <- c(ty, rep("Antibody Capture", n_ab))
  }
  m <- as(m, "dgCMatrix")
  h5 <- H5File$new(path, mode = "w")
  on.exit(h5$close_all())
  g <- h5$create_group("matrix")
  h5_str(g, "barcodes", x$bcs)
  g[["data"]] <- as.integer(m@x)
  g[["indices"]] <- as.integer(m@i)
  g[["indptr"]] <- as.integer(m@p)
  g[["shape"]] <- as.integer(dim(m))
  f <- g$create_group("features")
  h5_str(f, "id", ids); h5_str(f, "name", nm); h5_str(f, "feature_type", ty)
  h5_str(f, "genome", rep("GRCh38", length(ids)))
  path
}

# Plain AnnData (not an exchange file); counts in layers/counts, raw/X or X.
write_plain_h5ad <- function(path, x, where = c("layers", "raw", "X"), obs = NULL) {
  where <- match.arg(where)
  m <- as(x$m, "dgCMatrix")
  lognorm <- m; lognorm@x <- log1p(lognorm@x / rep(Matrix::colSums(m), diff(m@p)) * 1e4)
  h5 <- H5File$new(path, mode = "w")
  on.exit(h5$close_all())
  .scx_enc(h5, "anndata", "0.1.0")
  if (is.null(obs)) obs <- data.frame(row.names = x$bcs)
  .scx_write_dataframe(h5, "obs", obs, x$bcs, "obs")
  var <- data.frame(gene_ids = x$feats$id, feature_types = x$feats$type, stringsAsFactors = FALSE)
  if (where == "raw") {
    hv <- seq_len(50)                        # X holds a gene subset; raw holds all genes
    .scx_write_dataframe(h5, "var", var[hv, , drop = FALSE], x$feats$symbol[hv], "var")
    .scx_write_seurat_sparse(h5, "X", lognorm[hv, , drop = FALSE], h5types$H5T_NATIVE_FLOAT)
    raw <- h5$create_group("raw")
    .scx_write_dataframe(raw, "var", var, x$feats$symbol, "var")
    .scx_write_seurat_sparse(raw, "X", m, h5types$H5T_NATIVE_FLOAT)
  } else {
    .scx_write_dataframe(h5, "var", var, x$feats$symbol, "var")
    if (where == "X") .scx_write_seurat_sparse(h5, "X", m, h5types$H5T_NATIVE_INT32, data = as.integer(m@x))
    else .scx_write_seurat_sparse(h5, "X", lognorm, h5types$H5T_NATIVE_FLOAT)
  }
  lay <- .scx_empty_dict(h5, "layers")
  if (where == "layers") .scx_write_seurat_sparse(lay, "counts", m, h5types$H5T_NATIVE_INT32, data = as.integer(m@x))
  for (k in c("obsm", "varm", "obsp", "varp", "uns")) .scx_empty_dict(h5, k)
  path
}

# Loom: h5py shape (genes, cells) == R matrix cells x genes in hdf5r.
write_loom <- function(path, x) {
  h5 <- H5File$new(path, mode = "w")
  on.exit(h5$close_all())
  h5[["matrix"]] <- t(as.matrix(x$m))
  ra <- h5$create_group("row_attrs"); ca <- h5$create_group("col_attrs")
  h5_str(ra, "Gene", x$feats$symbol); h5_str(ra, "Accession", x$feats$id)
  h5_str(ca, "CellID", x$bcs)
  path
}

write_table <- function(path, x, sep = ",", ids = FALSE) {
  d <- as.data.frame(as.matrix(x$m), check.names = FALSE)
  colnames(d) <- x$bcs
  lead <- if (ids) data.frame(gene_id = x$feats$id, gene = x$feats$symbol, check.names = FALSE)
          else data.frame(gene = x$feats$symbol, check.names = FALSE)
  d <- cbind(lead, d)
  if (grepl("\\.gz$", path)) {
    con <- gzfile(path, "w"); on.exit(close(con))
    write.table(d, con, sep = sep, quote = FALSE, row.names = FALSE)
  } else write.table(d, path, sep = sep, quote = FALSE, row.names = FALSE)
  path
}

# Cells x genes table: header = gene symbols, first column = cell names.
write_table_t <- function(path, x, sep = ",") {
  d <- as.data.frame(t(as.matrix(x$m)), check.names = FALSE)
  colnames(d) <- x$feats$symbol
  d <- cbind(data.frame(cell = x$bcs, check.names = FALSE), d)
  write.table(d, path, sep = sep, quote = FALSE, row.names = FALSE)
  path
}

# Synthetic raw (unfiltered) 10x matrix over the fixture's first 200 genes:
# n_cells cells with 1000-5000 UMIs (first) plus n_empty barcodes with 0-5 UMIs.
raw_10x <- function(n_cells = 300, n_empty = 150000, seed = 1) {
  set.seed(seed)
  g <- FX$feats[SUB_G, , drop = FALSE]; ng <- nrow(g)
  p <- rexp(ng); p <- p / sum(p)
  tc <- round(runif(n_cells, 1000, 5000))
  cells <- vapply(tc, function(t) tabulate(sample.int(ng, t, TRUE, p), ng), integer(ng))
  te <- sample(0:5, n_empty, TRUE)
  j <- rep(seq_len(n_empty), te)
  emp <- sparseMatrix(i = sample.int(ng, length(j), TRUE, p), j = j, x = 1, dims = c(ng, n_empty))
  m <- cbind(as(Matrix(cells * 1, sparse = TRUE), "CsparseMatrix"), emp)
  list(m = as(m, "dgCMatrix"), feats = g, bcs = sprintf("BC%07d-1", seq_len(ncol(m))), n_cells = n_cells)
}

write_seurat_rds <- function(path, x, v3 = FALSE, split = FALSE, with_meta = TRUE) {
  m <- x$m
  rownames(m) <- x$feats$symbol
  old <- options(Seurat.object.assay.version = if (v3) "v3" else "v5")
  on.exit(options(old))
  md <- if (with_meta) data.frame(cell_type = factor(FX$cells$cell_type[match(x$bcs, FX$bcs)]),
                                  row.names = x$bcs) else NULL
  obj <- suppressWarnings(CreateSeuratObject(m, meta.data = md))
  if (split) {
    grp <- rep(c("a", "b"), length.out = ncol(obj))
    obj[["RNA"]] <- split(obj[["RNA"]], f = grp)
  }
  saveRDS(obj, path)
  path
}

# ------------------------------------------------------------------ run dirs
POLICY <- list(
  hard_limits = list(min_genes = 200, max_mito_pct = 25, min_umis = 400),
  protected_populations = list(
    list(name = "Platelet", markers = list("PPBP", "PF4", "GNG11"), min_fraction = 0.01),
    list(name = "Erythrocyte", markers = list("HBB", "HBA1", "HBA2"), min_fraction = 0.05)))

new_run <- function(samples, ..., tag = "run") {
  rd <- file.path(tempdir(), sprintf("cellwarden-%s-%s", tag, basename(tempfile(""))))
  dir.create(file.path(rd, "config"), recursive = TRUE)
  sf <- file.path(rd, "config", "samples.csv")
  write.csv(samples, sf, row.names = FALSE, quote = TRUE)
  pf <- file.path(rd, "config", "qc_policy.yaml")
  yaml::write_yaml(POLICY, pf)
  cfg <- list(project = "test", run_dir = rd, run_id = paste0("test-", tag), config_hash = strrep("a", 64),
              seed = 42L, samples = sf, qc_policy = pf, object_format = "rds",
              resources = list(threads = 1L, max_memory_gb = 4, gpu_mem_cap_gb = 20, nice = 10L),
              doublets = list(method = "none"), qc = list(review_action = "keep"),
              normalization = "lognorm", n_hvg = 500L, device = "cpu", engine = "seurat",
              integration = list(method = "none", batch_key = "sample_id"),
              pcs = "auto", n_neighbors = 20L, clustering = list(resolution = 0.5))
  over <- list(...)
  for (k in names(over)) cfg[[k]] <- over[[k]]
  yaml::write_yaml(cfg, file.path(rd, "config", "config.resolved.yaml"))
  rd
}
quiet <- function(expr) { out <- NULL; utils::capture.output(out <- force(expr)); out }
stage <- function(rd, fun) quiet(fun(scp_read_config(rd), REPO))
s0_obj <- function(rd) scp_load_object(file.path(rd, "S0"))
s0_report <- function(rd) jsonlite::fromJSON(file.path(rd, "S0", "ingest_report.json"), simplifyVector = FALSE)

write_decisions <- function(rd, decisions = NULL) {
  obj <- s0_obj(rd)
  d <- data.frame(cell_id = colnames(obj), sample_id = as.character(obj$sample_id),
                  decision = if (is.null(decisions)) "keep" else decisions,
                  reasons = "", policy_hash = "p", decided_by = "rules")
  dir.create(file.path(rd, "S1"), showWarnings = FALSE)
  scp_write_tsv_gz(d, file.path(rd, "S1", "qc_decisions.tsv.gz"))
  d
}

expect_same_counts <- function(obj, x, sid, seurat_names = FALSE) {
  got <- LayerData(obj, layer = "counts")
  fm <- obj[["RNA"]][[]]
  ref <- x$m
  cols <- paste0(sid, "_", x$bcs)
  expect_true(all(cols %in% colnames(got)))
  sym <- if (seurat_names) gsub("_", "-", x$feats$symbol) else x$feats$symbol
  rows <- match(sym, fm$gene_symbol_original)
  expect_false(anyNA(rows))
  sub <- got[rows, cols]
  expect_equal(unname(as.matrix(sub)), unname(as.matrix(ref)))
}

ari <- function(a, b) {
  tab <- table(a, b); n <- sum(tab)
  s2 <- function(x) sum(choose(x, 2))
  idx <- s2(tab); ai <- s2(rowSums(tab)); bj <- s2(colSums(tab))
  ex <- ai * bj / choose(n, 2)
  (idx - ex) / ((ai + bj) / 2 - ex)
}
purity <- function(cl, truth) { tab <- table(cl, truth); sum(apply(tab, 1, max)) / sum(tab) }

# ================================================================ S0
test_that("format detection follows spec 3.1", {
  td <- file.path(tempdir(), "detect"); dir.create(td, showWarnings = FALSE)
  x <- fx_sub()
  expect_identical(s0_detect_format(write_mtx_dir(file.path(td, "mtx"), x)), "10x_mtx")
  expect_identical(s0_detect_format(write_10x_h5(file.path(td, "a.h5"), x)), "10x_h5")
  expect_identical(s0_detect_format(write_plain_h5ad(file.path(td, "a.h5ad"), x)), "h5ad")
  expect_identical(s0_detect_format(write_loom(file.path(td, "a.loom"), x)), "loom")
  expect_identical(s0_detect_format(write_table(file.path(td, "a.csv"), x)), "table")
  expect_identical(s0_detect_format(write_table(file.path(td, "a.tsv.gz"), x, sep = "\t")), "table")
  expect_identical(s0_detect_format(write_seurat_rds(file.path(td, "a.rds"), x)), "seurat")
  expect_error(s0_detect_format(file.path(td, "nope.h5")), "not found")
})

test_that("S0 reads every format with identical counts, prefixed barcodes and gene metadata", {
  td <- file.path(tempdir(), sprintf("fmts-%s", basename(tempfile(""))))
  dir.create(td)
  x <- fx_sub()
  inputs <- list(
    mtx    = list(path = write_mtx_dir(file.path(td, "mtx"), x), fmt = ""),
    mtxgz  = list(path = write_mtx_dir(file.path(td, "mtxgz"), x, gz = TRUE), fmt = ""),
    mtxv2  = list(path = write_mtx_dir(file.path(td, "mtxv2"), x, v2 = TRUE), fmt = "10x_mtx"),
    h5     = list(path = write_10x_h5(file.path(td, "a.h5"), x, n_ab = 3), fmt = ""),
    h5ad_l = list(path = write_plain_h5ad(file.path(td, "l.h5ad"), x, "layers"), fmt = ""),
    h5ad_r = list(path = write_plain_h5ad(file.path(td, "r.h5ad"), x, "raw"), fmt = ""),
    h5ad_x = list(path = write_plain_h5ad(file.path(td, "x.h5ad"), x, "X"), fmt = "h5ad"),
    loom   = list(path = write_loom(file.path(td, "a.loom"), x), fmt = ""),
    csv    = list(path = write_table(file.path(td, "a.csv"), x, ids = TRUE), fmt = ""),
    tsvgz  = list(path = write_table(file.path(td, "a.tsv.gz"), x, sep = "\t"), fmt = ""),
    rds5   = list(path = write_seurat_rds(file.path(td, "v5.rds"), x), fmt = ""),
    rds3   = list(path = write_seurat_rds(file.path(td, "v3.rds"), x, v3 = TRUE), fmt = "seurat"),
    rdssp  = list(path = write_seurat_rds(file.path(td, "split.rds"), x, split = TRUE), fmt = "seurat"),
    sce    = list(path = local({
      sce <- SingleCellExperiment::SingleCellExperiment(
        assays = list(counts = `dimnames<-`(x$m, list(x$feats$symbol, x$bcs))),
        rowData = data.frame(gene_id = x$feats$id, row.names = x$feats$symbol),
        colData = data.frame(cell_type = factor(FX$cells$cell_type[SUB_C]), row.names = x$bcs))
      f <- file.path(td, "sce.rds"); saveRDS(sce, f); f }), fmt = "seurat"))
  # exchange file (full mode) written by the R exchange layer
  eo <- suppressWarnings(CreateSeuratObject(`rownames<-`(x$m, x$feats$symbol)))
  eo[["RNA"]] <- AddMetaData(eo[["RNA"]], data.frame(gene_symbol_original = x$feats$symbol,
                                                     gene_id = x$feats$id, row.names = rownames(eo)))
  eo$cell_type <- factor(FX$cells$cell_type[SUB_C])
  eo <- NormalizeData(eo, verbose = FALSE)
  inputs$exch <- list(path = write_exchange(eo, file.path(td, "ex.h5ad"), mode = "full"), fmt = "")
  expected_fmt <- c(mtx = "10x_mtx", mtxgz = "10x_mtx", mtxv2 = "10x_mtx", h5 = "10x_h5", h5ad_l = "h5ad",
                    h5ad_r = "h5ad", h5ad_x = "h5ad", loom = "loom", csv = "table", tsvgz = "table",
                    rds5 = "seurat", rds3 = "seurat", rdssp = "seurat", sce = "seurat", exch = "h5ad")
  has_ids <- c(mtx = TRUE, mtxgz = TRUE, mtxv2 = TRUE, h5 = TRUE, h5ad_l = TRUE, h5ad_r = TRUE, h5ad_x = TRUE,
               loom = TRUE, csv = TRUE, tsvgz = FALSE, rds5 = FALSE, rds3 = FALSE, rdssp = FALSE, sce = TRUE, exch = TRUE)
  for (nm in names(inputs)) {
    sid <- paste0("S_", nm)
    rd <- new_run(data.frame(sample_id = sid, path = inputs[[nm]]$path, format = inputs[[nm]]$fmt), tag = nm)
    stage(rd, run_s0)
    obj <- s0_obj(rd)
    rep <- s0_report(rd)
    info <- paste("format case", nm)
    expect_s4_class(obj[["RNA"]], "Assay5")
    expect_identical(Layers(obj[["RNA"]]), "counts", label = info)
    expect_equal(dim(obj), c(200, 60), label = info)
    expect_identical(colnames(obj), paste0(sid, "_", x$bcs), label = info)
    expect_identical(rep$samples[[1]]$format, expected_fmt[[nm]], label = info)
    expect_identical(rep$gene_key, if (has_ids[[nm]]) "gene_id" else "gene_symbol", label = info)
    fm <- obj[["RNA"]][[]]
    expect_true(all(c("gene_symbol_original", "gene_id") %in% colnames(fm)), label = info)
    # originals kept (underscores), Seurat names dash-converted
    # Seurat objects arrive with Seurat's own ('_' -> '-') names; other formats keep the originals
    sym_exp <- if (nm %in% c("rds5", "rds3", "rdssp")) gsub("_", "-", x$feats$symbol) else x$feats$symbol
    expect_setequal(fm$gene_symbol_original, sym_exp)
    if (has_ids[[nm]]) expect_identical(fm$gene_id[match(x$feats$symbol, fm$gene_symbol_original)], x$feats$id, label = info)
    else expect_true(all(fm$gene_id == ""), label = info)
    expect_same_counts(obj, x, sid, seurat_names = nm %in% c("rds5", "rds3", "rdssp"))
    expect_identical(levels(obj$sample_id), sid)
    expect_equal(rep$totals$total_umis, sum(x$m))
    if (nm %in% c("rds5", "rds3", "rdssp", "sce", "exch"))
      expect_identical(as.character(obj$cell_type), FX$cells$cell_type[SUB_C], label = paste(info, "carried meta"))
  }
})

test_that("two samples in two formats: union of genes, zeros where missing, sheet metadata as factors", {
  td <- file.path(tempdir(), sprintf("two-%s", basename(tempfile(""))))
  dir.create(td)
  a <- which(FX$cells$sample == "donor_A"); b <- which(FX$cells$sample == "donor_B")
  xa <- list(m = FX$m[1:700, a], feats = FX$feats[1:700, ], bcs = FX$bcs[a])
  xb <- list(m = FX$m[101:800, b], feats = FX$feats[101:800, ], bcs = FX$bcs[b])
  sheet <- data.frame(sample_id = c("donor_B", "donor_A"),
                      path = c(write_10x_h5(file.path(td, "b.h5"), xb), write_mtx_dir(file.path(td, "a"), xa, gz = TRUE)),
                      format = "", condition = c("treated", "control"), batch = c("B2", "B1"))
  rd <- new_run(sheet, tag = "two")
  stage(rd, run_s0)
  obj <- s0_obj(rd)
  rep <- s0_report(rd)
  expect_equal(dim(obj), c(800, 500))
  expect_identical(rep$gene_key, "gene_id")
  expect_identical(levels(obj$sample_id), c("donor_B", "donor_A"))
  expect_identical(levels(obj$condition), c("treated", "control"))     # sheet order of first appearance
  expect_true(is.factor(obj$batch))
  expect_identical(as.character(obj$condition[paste0("donor_A_", FX$bcs[a[1]])]), "control")
  cnt <- LayerData(obj, layer = "counts")
  fm <- obj[["RNA"]][[]]
  g_only_b <- match(FX$feats$id[750], fm$gene_id)
  expect_identical(sum(cnt[g_only_b, paste0("donor_A_", FX$bcs[a])]), 0)
  expect_equal(sum(cnt[g_only_b, paste0("donor_B_", FX$bcs[b])]), sum(FX$m[750, b]))
  expect_equal(sum(cnt), sum(FX$m[1:700, a]) + sum(FX$m[101:800, b]))
  expect_identical(rep$totals$samples, 2L)
  expect_length(rep$samples, 2)
  expect_identical(vapply(rep$samples, function(s) s$format, ""), c("10x_h5", "10x_mtx"))
  expect_identical(unlist(rep$sample_metadata_columns), c("condition", "batch"))
})

test_that("a sample without gene IDs switches the key to symbols", {
  td <- file.path(tempdir(), sprintf("sym-%s", basename(tempfile(""))))
  dir.create(td)
  x1 <- fx_sub(cells = 1:30); x2 <- fx_sub(cells = 31:60)
  sheet <- data.frame(sample_id = c("s1", "s2"),
                      path = c(write_mtx_dir(file.path(td, "a"), x1), write_table(file.path(td, "b.csv"), x2)),
                      format = "")
  rd <- new_run(sheet, tag = "symkey")
  stage(rd, run_s0)
  rep <- s0_report(rd); obj <- s0_obj(rd)
  expect_identical(rep$gene_key, "gene_symbol")
  expect_match(rep$gene_key_reason, "s2")
  fm <- obj[["RNA"]][[]]
  expect_equal(nrow(obj), 200)
  # IDs known from sample 1 are kept in feature metadata
  expect_identical(fm$gene_id[match(x1$feats$symbol, fm$gene_symbol_original)], x1$feats$id)
})

test_that("duplicate symbols are made unique and reported", {
  td <- file.path(tempdir(), sprintf("dup-%s", basename(tempfile(""))))
  dir.create(td)
  x <- fx_sub()
  x$feats$symbol[c(10, 20, 30)] <- "CD3E"      # CD3E is gene 10 already
  x$feats$symbol[c(40, 41)] <- "DUPGENE"
  rd <- new_run(data.frame(sample_id = "d", path = write_table(file.path(td, "d.tsv"), x, sep = "\t"), format = ""), tag = "dup")
  stage(rd, run_s0)
  rep <- s0_report(rd); obj <- s0_obj(rd)
  expect_identical(rep$samples[[1]]$duplicate_symbols$n, 2L)
  expect_setequal(unlist(rep$samples[[1]]$duplicate_symbols$examples), c("CD3E", "DUPGENE"))
  expect_true(all(c("CD3E", "CD3E.1", "CD3E.2", "DUPGENE", "DUPGENE.1") %in% rownames(obj)))
  expect_false(anyDuplicated(rownames(obj)) > 0)
  fm <- obj[["RNA"]][[]]
  expect_identical(fm["CD3E.2", "gene_symbol_original"], "CD3E")
  expect_equal(unname(LayerData(obj, layer = "counts")["CD3E.1", ]), unname(x$m[20, ]))
})

test_that("non-integer counts stop S0 with a message naming the sample", {
  td <- file.path(tempdir(), sprintf("nonint-%s", basename(tempfile(""))))
  dir.create(td)
  x <- fx_sub(); x$m <- x$m * 0.37
  rd <- new_run(data.frame(sample_id = c("good", "bad_sample"),
                           path = c(write_mtx_dir(file.path(td, "g"), fx_sub()), write_table(file.path(td, "bad.csv"), x)),
                           format = ""), tag = "nonint")
  expect_error(stage(rd, run_s0), "sample bad_sample: counts are not integers.*raw UMI counts")
  # h5ad with only normalised X: rejected with an h5ad hint
  y <- fx_sub()
  p <- write_plain_h5ad(file.path(td, "norm.h5ad"), y, "layers")
  h5 <- H5File$new(p, "r+"); h5$link_delete("layers"); .scx_empty_dict(h5, "layers"); h5$close_all()
  rd2 <- new_run(data.frame(sample_id = "nrm", path = p, format = ""), tag = "nonint2")
  expect_error(stage(rd2, run_s0), "nrm.*X is not integer counts")
  expect_false(file.exists(file.path(rd, "S0", "object.rds")))
})

test_that("sample sheet problems are all reported", {
  td <- file.path(tempdir(), sprintf("sheet-%s", basename(tempfile(""))))
  dir.create(td)
  p <- write_mtx_dir(file.path(td, "a"), fx_sub())
  rd <- new_run(data.frame(sample_id = c("x", "x", ""), path = c(p, p, p), format = c("", "bogus", "")), tag = "sheet")
  err <- tryCatch(stage(rd, run_s0), error = conditionMessage)
  expect_match(err, "duplicated sample_id: x")
  expect_match(err, "empty sample_id")
  expect_match(err, "unknown format")
  rd2 <- new_run(data.frame(sample_id = "m", path = file.path(td, "missing")), tag = "sheet2")
  expect_error(stage(rd2, run_s0), "not found")
  # relative paths resolve against the sample sheet's directory
  rd3 <- new_run(data.frame(sample_id = "rel", path = "../../rel_in"), tag = "sheet3")
  file.copy(p, file.path(dirname(rd3)), recursive = TRUE)
  file.rename(file.path(dirname(rd3), "a"), file.path(dirname(rd3), "rel_in"))
  on.exit(unlink(file.path(dirname(rd3), "rel_in"), recursive = TRUE))
  stage(rd3, run_s0)
  expect_equal(ncol(s0_obj(rd3)), 60)
})

test_that("organism check warns when the top genes do not look human", {
  td <- file.path(tempdir(), sprintf("mouse-%s", basename(tempfile(""))))
  dir.create(td)
  x <- fx_sub()
  x$feats$symbol <- paste0(substr(x$feats$symbol, 1, 1), tolower(substring(x$feats$symbol, 2)))  # Cd3e, Lyz ...
  x$feats$id <- sprintf("ENSMUSG%011d", seq_along(x$feats$id))
  rd <- new_run(data.frame(sample_id = "mm", path = write_mtx_dir(file.path(td, "m"), x)), tag = "mouse")
  stage(rd, run_s0)
  rep <- s0_report(rd)
  expect_lt(rep$human_check$fraction_top_genes, 0.5)
  expect_match(rep$human_check$warning, "may not be human")
  expect_true(any(grepl("may not be human", unlist(rep$warnings))))
  # the real fixture passes
  rd2 <- new_run(data.frame(sample_id = "hs", path = write_mtx_dir(file.path(td, "h"), fx_sub())), tag = "human")
  stage(rd2, run_s0)
  expect_gte(s0_report(rd2)$human_check$fraction_top_genes, 0.5)
  expect_null(s0_report(rd2)$human_check$warning)
})

test_that("a cells x genes table gives the same counts as the genes x cells table", {
  td <- file.path(tempdir(), sprintf("orient-%s", basename(tempfile(""))))
  dir.create(td)
  x <- fx_sub()
  rd <- new_run(data.frame(sample_id = c("gc", "cg"),
                           path = c(write_table(file.path(td, "gc.csv"), x), write_table_t(file.path(td, "cg.csv"), x)),
                           format = ""), tag = "orient")
  stage(rd, run_s0)
  obj <- s0_obj(rd)
  rep <- s0_report(rd)
  expect_equal(dim(obj), c(200, 120))
  expect_same_counts(obj, x, "gc")
  expect_same_counts(obj, x, "cg")
  ori <- vapply(rep$samples, function(s) s$orientation$orientation, "")
  expect_identical(ori, c("genes_x_cells", "cells_x_genes"))
  expect_true(any(grepl("cells x genes and transposed", unlist(rep$samples[[2]]$notes))))
  expect_false(any(grepl("ambiguous", unlist(rep$warnings))))
  # neither side looks like genes: kept as genes x cells, with a warning
  x2 <- x; x2$feats$symbol <- sprintf("feat%03d", seq_along(x$feats$symbol))
  rd2 <- new_run(data.frame(sample_id = "amb", path = write_table(file.path(td, "amb.csv"), x2), format = ""), tag = "amb")
  stage(rd2, run_s0)
  rep2 <- s0_report(rd2)
  expect_equal(dim(s0_obj(rd2)), c(200, 60))
  expect_true(isTRUE(rep2$samples[[1]]$orientation$ambiguous))
  expect_true(any(grepl("amb: table orientation is ambiguous", unlist(rep2$warnings))))
})

test_that("raw 10x matrices: empty droplets removed at the barcode-rank knee (auto/on/off)", {
  td <- file.path(tempdir(), sprintf("edrops-%s", basename(tempfile(""))))
  dir.create(td)
  x <- raw_10x()
  h5 <- write_10x_h5(file.path(td, "raw_feature_bc_matrix.h5"), x)
  rd <- new_run(data.frame(sample_id = "r", path = h5, format = ""), tag = "ed-auto")
  stage(rd, run_s0)
  obj <- s0_obj(rd)
  ed <- s0_report(rd)$samples[[1]]$empty_drops
  expect_true(ed$applied)
  expect_match(ed$reason, "raw 10x")
  expect_equal(ed$kept, 300)
  expect_equal(ed$dropped, 150000)
  expect_gte(ed$threshold_umi, 50)
  expect_lte(ed$threshold_umi, 1000)
  expect_identical(colnames(obj), paste0("r_", x$bcs[1:300]))
  expect_true(any(grepl("empty droplets removed", unlist(s0_report(rd)$samples[[1]]$notes))))
  # off keeps every barcode
  rd2 <- new_run(data.frame(sample_id = "r", path = h5, format = ""), ingest = list(empty_drops = "off"), tag = "ed-off")
  stage(rd2, run_s0)
  expect_equal(ncol(s0_obj(rd2)), 150300)
  expect_false(s0_report(rd2)$samples[[1]]$empty_drops$applied)
  # without a raw_* name, auto still triggers on many barcodes with a low median
  e <- s0_empty_drops(x$m, file.path(td, "sample.h5"), "10x_h5", "auto")
  expect_true(e$info$applied)
  expect_match(e$info$reason, "looks unfiltered")
  # a small mtx: auto leaves it alone, on (YAML bare 'on' reads as TRUE) filters it
  xs <- raw_10x(n_cells = 100, n_empty = 2000, seed = 2)
  dir_s <- write_mtx_dir(file.path(td, "small"), xs)
  rd3 <- new_run(data.frame(sample_id = "s", path = dir_s, format = ""), tag = "ed-small")
  stage(rd3, run_s0)
  expect_equal(ncol(s0_obj(rd3)), 2100)
  expect_match(s0_report(rd3)$samples[[1]]$empty_drops$reason, "looks filtered")
  rd4 <- new_run(data.frame(sample_id = "s", path = dir_s, format = ""), ingest = list(empty_drops = TRUE), tag = "ed-on")
  stage(rd4, run_s0)
  ed4 <- s0_report(rd4)$samples[[1]]$empty_drops
  expect_true(ed4$applied)
  expect_equal(ed4$kept, 100)
  expect_equal(ncol(s0_obj(rd4)), 100)
  # non-10x formats carry no empty_drops field; bad values stop
  expect_null(s0_empty_drops(xs$m, "a.csv", "table", "on"))
  expect_error(.s0_empty_drops_mode(list(ingest = list(empty_drops = "maybe"))), "auto, on, off")
  expect_identical(.s0_empty_drops_mode(list()), "auto")
})

test_that("bundled human symbol list is present and plausible", {
  hs <- .s0_human_symbols(REPO)
  expect_gt(length(hs), 2000)
  expect_true(all(c("CD3E", "MT-CO1", "RPL3", "HLA-DRA", "HBB", "ACTB") %in% hs))
})

# ================================================================ S1a
test_that("S1a metrics match a hand computation, including hb exclusions and protected fractions", {
  genes <- c("MT-CO1", "MT-ND1", "RPL3", "RPS6", "RPLP0", "HBB", "HBA1", "HBZ", "HBQ1", "HBP1", "HBEGF", "HBS1L",
             "PPBP", "PF4", "CD3E", "LYZ", "MRPL13", "RPS6KA1")
  set.seed(1)
  m <- Matrix::Matrix(matrix(rpois(length(genes) * 8, 3), nrow = length(genes)), sparse = TRUE)
  m[, 8] <- 0; m[15, 8] <- 1                       # one UMI: log10 ratio undefined
  m <- as(m, "dgCMatrix")
  feats <- data.frame(id = sprintf("ENSG%011d", seq_along(genes)), symbol = genes, type = "Gene Expression")
  x <- list(m = m, feats = feats, bcs = sprintf("C%02d-1", 1:8))
  td <- file.path(tempdir(), sprintf("s1a-%s", basename(tempfile(""))))
  dir.create(td)
  rd <- new_run(data.frame(sample_id = "q", path = write_mtx_dir(file.path(td, "q"), x)), tag = "s1a")
  stage(rd, run_s0)
  stage(rd, run_s1a)
  f <- file.path(rd, "S1", "qc_metrics.tsv.gz")
  expect_true(file.exists(f))
  expect_identical(readBin(f, "raw", 2), as.raw(c(0x1f, 0x8b)))        # gzip
  met <- read.delim(gzfile(f), check.names = FALSE, colClasses = c(cell_id = "character", sample_id = "character"))
  expect_identical(colnames(met), c("cell_id", "sample_id", "n_genes", "n_umis", "pct_mito", "pct_ribo", "pct_hb",
                                    "log10_genes_per_umi", "protected_Platelet", "protected_Erythrocyte"))
  expect_identical(met$cell_id, paste0("q_", x$bcs))
  expect_true(all(met$sample_id == "q"))
  M <- as.matrix(m)
  for (j in c(1, 3, 7)) {
    tot <- sum(M[, j]); ng <- sum(M[, j] > 0)
    expect_equal(met$n_umis[j], tot)
    expect_equal(met$n_genes[j], ng)
    expect_equal(met$pct_mito[j], 100 * sum(M[1:2, j]) / tot, tolerance = 1e-9)
    # ^RP[SL]: RPL3, RPS6, RPLP0, RPS6KA1 (prefix match, as in the spec); not MRPL13
    expect_equal(met$pct_ribo[j], 100 * sum(M[c(3, 4, 5, 18), j]) / tot, tolerance = 1e-9)
    # haemoglobin: HBB HBA1 HBZ HBQ1; not HBP1 HBEGF HBS1L
    expect_equal(met$pct_hb[j], 100 * sum(M[6:9, j]) / tot, tolerance = 1e-9)
    expect_equal(met$log10_genes_per_umi[j], log10(ng) / log10(tot), tolerance = 1e-9)
    expect_equal(met$protected_Platelet[j], sum(M[13:14, j]) / tot, tolerance = 1e-9)   # GNG11 missing -> 0
    expect_equal(met$protected_Erythrocyte[j], sum(M[6:7, j]) / tot, tolerance = 1e-9)  # HBA2 missing -> 0
  }
  expect_true(is.na(met$log10_genes_per_umi[8]))
  # exchange file: full mode, all cells, counts + log-normalised X
  ex_path <- file.path(rd, "S1", "metrics_counts.h5ad")
  expect_length(validate_exchange(ex_path), 0)
  ex <- read_exchange(ex_path)
  expect_identical(ex$uns$exchange$mode, "full")
  expect_identical(ex$uns$exchange$stage, "S1a")
  expect_identical(ex$uns$exchange$run_id, "test-s1a")
  expect_identical(rownames(ex$obs), met$cell_id)
  expect_identical(ex$uns$exchange$counts_checksum, sc_counts_checksum(LayerData(s0_obj(rd), layer = "counts")))
  j <- 1
  expect_equal(unname(ex$X[, j]), unname(log1p(M[, j] / sum(M[, j]) * 1e4)), tolerance = 1e-6)
})

test_that("S1a logs missing protected-population markers as warnings", {
  x <- fx_sub()
  td <- file.path(tempdir(), sprintf("s1aw-%s", basename(tempfile(""))))
  dir.create(td)
  rd <- new_run(data.frame(sample_id = "w", path = write_mtx_dir(file.path(td, "w"), x)), tag = "s1aw")
  stage(rd, run_s0)
  out <- utils::capture.output(run_s1a(scp_read_config(rd), REPO))
  expect_true(any(grepl("\\[WARN\\] protected population 'Platelet': marker\\(s\\) not in the data, counted as 0: PPBP, PF4, GNG11", out)))
  met <- read.delim(gzfile(file.path(rd, "S1", "qc_metrics.tsv.gz")))
  expect_true(all(met$protected_Platelet == 0))
})

# ================================================================ S2 helpers
test_that("elbow rule picks PCs within 10..50", {
  sd <- c(10, 8, 6, 4, 3, 2.5, 2.2, 2, rep(1.9, 42) - seq(0, 0.4, length.out = 42))
  el <- s2_elbow_pcs(sd)
  expect_gte(el$n, 10); expect_lte(el$n, 50)
  pct <- sd / sum(sd) * 100
  co2 <- max(which(pct[-50] - pct[-1] > 0.1)) + 1
  co1 <- which(cumsum(pct) > 90 & pct < 5)[1]
  expect_identical(el$n, as.integer(max(10, min(50, co1, co2))))
  expect_identical(s2_elbow_pcs(rev(seq(1, 100, length.out = 60)))$n <= 50, TRUE)
  expect_identical(s2_elbow_pcs(c(5, 4, 3))$n, 3L)      # fewer PCs than the minimum available
})

test_that("plan hook is validated", {
  cfg <- list(device = "auto", normalization = "lognorm", resources = list(gpu_mem_cap_gb = 20))
  withr::local_envvar(CELLWARDEN_PLAN_JSON = '{"device":"gpu","plan":"managed_memory","reason":"r"}')
  p <- s2_choose_plan(cfg, REPO, 100, 50)
  expect_identical(p[c("device", "plan", "reason")], list(device = "gpu", plan = "managed_memory", reason = "r"))
  withr::local_envvar(CELLWARDEN_PLAN_JSON = '{"device":"gpu","plan":"cpu"}')
  expect_error(s2_choose_plan(cfg, REPO, 100, 50), "cannot have plan cpu")
  withr::local_envvar(CELLWARDEN_PLAN_JSON = "not json")
  expect_error(s2_choose_plan(cfg, REPO, 100, 50), "not valid JSON")
  f <- tempfile(fileext = ".json"); writeLines('{"device":"cpu","plan":"cpu","reason":"file"}', f)
  withr::local_envvar(CELLWARDEN_PLAN_JSON = f)
  expect_identical(s2_choose_plan(cfg, REPO, 100, 50)$reason, "file")
})

test_that("Leiden fallback via leidenbase is deterministic and size-ordered", {
  set.seed(3)
  blocks <- Matrix::bdiag(lapply(c(30, 20, 10), function(n) Matrix::Matrix(1, n, n)))
  snn <- as(blocks, "dgCMatrix"); diag(snn) <- 0
  dimnames(snn) <- list(paste0("c", 1:60), paste0("c", 1:60))
  a <- s2_leiden_leidenbase(Matrix::drop0(snn), 0.5, 1)
  b <- s2_leiden_leidenbase(Matrix::drop0(snn), 0.5, 1)
  expect_identical(a, b)
  expect_identical(as.integer(table(a)), c(30L, 20L, 10L))
  expect_identical(levels(a), c("0", "1", "2"))
  expect_identical(names(a), rownames(snn))
})

# ================================================================ S2 end to end
TWO <- local({
  td <- file.path(tempdir(), sprintf("s2in-%s", basename(tempfile(""))))
  dir.create(td)
  a <- which(FX$cells$sample == "donor_A"); b <- which(FX$cells$sample == "donor_B")
  xa <- list(m = FX$m[, a], feats = FX$feats, bcs = FX$bcs[a])
  xb <- list(m = FX$m[, b], feats = FX$feats, bcs = FX$bcs[b])
  data.frame(sample_id = c("donor_A", "donor_B"),
             path = c(write_mtx_dir(file.path(td, "A"), xa), write_10x_h5(file.path(td, "B.h5"), xb)),
             format = "")
})
truth_for <- function(obj) {
  bc <- sub("^donor_[AB]_", "", colnames(obj))
  FX$cells$cell_type[match(bc, FX$bcs)]
}
CPU_RUN <- NULL

test_that("S2 CPU path recovers the planted cell types (purity >= 0.9)", {
  rd <- new_run(TWO, tag = "s2cpu")
  stage(rd, run_s0)
  set.seed(5)
  dec <- sample(c("keep", "review", "exclude"), 500, TRUE, prob = c(0.9, 0.05, 0.05))
  write_decisions(rd, dec)
  withr::local_envvar(CELLWARDEN_PLAN_JSON = '{"device":"cpu","plan":"cpu","reason":"test: cpu"}')
  stage(rd, run_s2)
  obj <- scp_load_object(file.path(rd, "S2"))
  sm <- jsonlite::fromJSON(file.path(rd, "S2", "summary.json"), simplifyVector = FALSE)
  expect_equal(ncol(obj), sum(dec != "exclude"))
  expect_identical(sm$cells_in, 500L); expect_identical(sm$cells_out, sum(dec != "exclude"))
  expect_identical(sm$device, "cpu"); expect_identical(sm$plan, "cpu"); expect_identical(sm$reason, "test: cpu")
  expect_identical(sm$path, "cpu_seurat")
  expect_true(all(c("pca", "umap") %in% Reductions(obj)))
  expect_true(all(c("RNA_snn", "RNA_nn") %in% Graphs(obj)))
  expect_gte(sm$pcs, 10); expect_lte(sm$pcs, 50)
  expect_identical(sm$clusters, nlevels(obj$seurat_clusters))
  expect_identical(Idents(obj), obj$seurat_clusters)
  expect_true(all(c("load", "subset", "normalize", "hvg", "scale", "pca", "neighbors", "leiden", "umap", "save") %in% names(sm$timings_s)))
  expect_true(is.numeric(sm$peak_memory$r_peak_rss_gb))
  expect_identical(length(VariableFeatures(obj)), 500L)
  expect_false("scale.data" %in% Layers(obj[["RNA"]]))
  expect_equal(nrow(obj), 800)
  p <- purity(obj$seurat_clusters, truth_for(obj))
  expect_gte(p, 0.9)
  expect_gte(nlevels(obj$seurat_clusters), 4)
  CPU_RUN <<- list(rd = rd, cl = setNames(as.character(obj$seurat_clusters), colnames(obj)))
})

# ---- S2 scfast path (docs/scfast-integration.md); skipped when scfastR is absent
test_that("S2 engine resolution: auto picks scfast only for an installed scfastR and a CPU plan", {
  cpu <- list(device = "cpu"); gpu <- list(device = "gpu")
  expect_identical(s2_resolve_engine(list(engine = "seurat"), cpu), "seurat")
  expect_identical(s2_resolve_engine(list(engine = "auto"), gpu), "seurat")
  expect_error(s2_resolve_engine(list(engine = "bogus"), cpu), "engine must be")
  if (requireNamespace("scfastR", quietly = TRUE)) {
    expect_identical(s2_resolve_engine(list(engine = "auto"), cpu), "scfast")
    expect_identical(s2_resolve_engine(list(), cpu), "scfast")
    expect_error(s2_resolve_engine(list(engine = "scfast"), gpu), "no GPU backend")
  } else {
    expect_identical(s2_resolve_engine(list(engine = "auto"), cpu), "seurat")
    expect_error(s2_resolve_engine(list(engine = "scfast"), cpu), "scfastR")
  }
})

# ---- GPU-plan engine resolution (docs/scfast-integration.md "Pipeline wiring"), scfast_info mocked
test_that("scp_scfast_cuda treats missing cuda fields as FALSE", {
  expect_identical(scp_scfast_cuda(NULL), list(built = FALSE, available = FALSE))
  expect_identical(scp_scfast_cuda(list(version = "0.1.0")), list(built = FALSE, available = FALSE))
  expect_identical(scp_scfast_cuda(list(cuda = FALSE)), list(built = FALSE, available = FALSE))
  expect_identical(scp_scfast_cuda(list(cuda_built = TRUE)), list(built = TRUE, available = FALSE))
  expect_identical(scp_scfast_cuda(list(cuda_built = TRUE, cuda_available = NA)), list(built = TRUE, available = FALSE))
  expect_identical(scp_scfast_cuda(list(cuda_built = FALSE, cuda_available = TRUE)), list(built = FALSE, available = FALSE))
  expect_identical(scp_scfast_cuda(list(cuda_built = TRUE, cuda_available = TRUE)), list(built = TRUE, available = TRUE))
  # the real scfastR on this machine (if any) has no CUDA
  expect_false(scp_scfast_cuda()$available)
})

test_that("scp_resolve_engine on a GPU plan honours engine_gpu and the scfastR CUDA state", {
  # scfast_info() is mocked by rebinding scp_scfast_info where scp_resolve_engine looks it up
  with_info <- function(info, code) {
    env <- environment(scp_resolve_engine)
    old <- get("scp_scfast_info", envir = env)
    assign("scp_scfast_info", function() info, envir = env)
    on.exit(assign("scp_scfast_info", old, envir = env), add = TRUE)
    force(code)
  }
  cuda_ok <- list(version = "0.1.0", cuda_built = TRUE, cuda_available = TRUE)
  cuda_built_only <- list(version = "0.1.0", cuda_built = TRUE, cuda_available = FALSE)
  no_cuda <- list(version = "0.1.0", cuda = FALSE)
  for (info in list(cuda_ok, cuda_built_only, no_cuda, NULL)) {
    with_info(info, {
      expect_identical(scp_resolve_engine(list(engine = "seurat"), "gpu"), "seurat")
      expect_identical(scp_resolve_engine(list(engine_gpu = "rapids"), "gpu"), "seurat")
      expect_error(scp_resolve_engine(list(engine = "scfast", engine_gpu = "rapids"), "gpu"), "no GPU backend")
      expect_error(scp_resolve_engine(list(engine_gpu = "cuda"), "gpu"), "engine_gpu must be")
      expect_error(scp_resolve_engine(list(engine_gpu = "cuda"), "cpu"), "engine_gpu must be")
    })
  }
  with_info(cuda_ok, {
    expect_identical(scp_resolve_engine(list(), "gpu"), "scfast")
    expect_identical(scp_resolve_engine(list(engine = "auto", engine_gpu = "auto"), "gpu"), "scfast")
    expect_identical(scp_resolve_engine(list(engine_gpu = "scfast"), "gpu"), "scfast")
    expect_identical(scp_resolve_engine(list(engine = "scfast"), "gpu"), "scfast")
    expect_identical(s2_resolve_engine(list(), list(device = "gpu")), "scfast")
    expect_identical(s2_resolve_engine(list(), list(device = "gpu"), assay = "SCT"), "seurat")
    expect_error(s2_resolve_engine(list(engine = "scfast"), list(device = "gpu"), assay = "SCT"), "lognorm")
  })
  for (info in list(cuda_built_only, no_cuda, NULL)) {
    with_info(info, {
      expect_identical(scp_resolve_engine(list(), "gpu"), "seurat")
      expect_identical(scp_resolve_engine(list(engine_gpu = "auto"), "gpu"), "seurat")
      expect_identical(s2_resolve_engine(list(), list(device = "gpu")), "seurat")
      expect_error(scp_resolve_engine(list(engine_gpu = "scfast"), "gpu"), "engine_gpu: scfast needs scfastR built with CUDA")
      expect_error(scp_resolve_engine(list(engine = "scfast"), "gpu"), "no GPU backend")
    })
  }
  # the CUDA state never changes a CPU plan
  with_info(cuda_ok, {
    have <- requireNamespace("scfastR", quietly = TRUE)
    expect_identical(scp_resolve_engine(list(engine_gpu = "scfast"), "cpu"), if (have) "scfast" else "seurat")
    expect_identical(scp_resolve_engine(list(engine_gpu = "rapids"), "cpu"), if (have) "scfast" else "seurat")
  })
  expect_identical(.s2_scfast_device("gpu_scfast"), "cuda")
  expect_identical(.s2_scfast_device("scfast"), "cpu")
})

test_that("S2 scfast path recovers the planted cell types and writes no exchange files", {
  skip_if_not_installed("scfastR")
  rd <- new_run(TWO, tag = "s2fast", engine = "scfast")
  stage(rd, run_s0)
  set.seed(5)
  dec <- sample(c("keep", "review", "exclude"), 500, TRUE, prob = c(0.9, 0.05, 0.05))
  write_decisions(rd, dec)
  withr::local_envvar(CELLWARDEN_PLAN_JSON = '{"device":"cpu","plan":"cpu","reason":"test: cpu"}')
  stage(rd, run_s2)
  obj <- scp_load_object(file.path(rd, "S2"))
  sm <- jsonlite::fromJSON(file.path(rd, "S2", "summary.json"), simplifyVector = FALSE)
  expect_equal(ncol(obj), sum(dec != "exclude"))
  expect_identical(sm$device, "cpu"); expect_identical(sm$plan, "cpu")
  expect_identical(sm$path, "cpu_scfast"); expect_identical(sm$engine, "scfast")
  expect_true(is.character(sm$scfast_version) && nzchar(sm$scfast_version))
  expect_false(dir.exists(file.path(rd, "S2", "exchange")))
  expect_length(list.files(rd, pattern = "\\.h5ad$", recursive = TRUE), 0)
  expect_true(all(c("pca", "umap") %in% Reductions(obj)))
  expect_true(all(c("RNA_snn", "RNA_nn") %in% Graphs(obj)))
  expect_gte(sm$pcs, 10); expect_lte(sm$pcs, 50)
  expect_identical(sm$clusters, nlevels(obj$seurat_clusters))
  expect_identical(Idents(obj), obj$seurat_clusters)
  # Phase 9: the fused PreprocessFast() replaces normalize + hvg (+ scale) on the scfast path.
  expect_true(all(c("load", "subset", "preprocess", "pca", "neighbors", "leiden", "umap", "save") %in% names(sm$timings_s)))
  expect_false(any(c("normalize", "hvg", "scale") %in% names(sm$timings_s)))
  expect_match(sm$preprocessing, "PreprocessFast")
  expect_identical(length(VariableFeatures(obj)), 500L)
  expect_false("scale.data" %in% Layers(obj[["RNA"]]))
  expect_equal(nrow(obj), 800)
  expect_identical(ncol(Embeddings(obj, "umap")), 2L)
  expect_match(sm$umap$implementation, "RunUMAPFast")
  expect_gte(purity(obj$seurat_clusters, truth_for(obj)), 0.9)
  expect_gte(nlevels(obj$seurat_clusters), 4)
  if (!is.null(CPU_RUN)) {      # same decisions as the Seurat run: partitions should agree closely
    expect_identical(colnames(obj), names(CPU_RUN$cl))
    expect_gte(ari(CPU_RUN$cl, as.character(obj$seurat_clusters)), 0.8)
  }
})

test_that("S2 scfast path honours a fixed pcs and review_action exclude", {
  skip_if_not_installed("scfastR")
  rd <- new_run(TWO, tag = "s2fastrev", engine = "scfast", qc = list(review_action = "exclude"), pcs = 15L, n_hvg = 300L)
  stage(rd, run_s0)
  dec <- rep(c("keep", "review", "keep", "exclude"), length.out = 500)
  write_decisions(rd, dec)
  withr::local_envvar(CELLWARDEN_PLAN_JSON = '{"device":"cpu","plan":"cpu","reason":"t"}')
  stage(rd, run_s2)
  obj <- scp_load_object(file.path(rd, "S2"))
  sm <- jsonlite::fromJSON(file.path(rd, "S2", "summary.json"))
  expect_equal(ncol(obj), sum(dec == "keep"))
  expect_identical(sm$path, "cpu_scfast")
  expect_identical(sm$pcs, 15L)
  expect_identical(ncol(Embeddings(obj, "pca")), 15L)
  expect_identical(sm$n_hvg, 300L)
})

test_that("S2 engine auto uses scfast on a CPU plan and seurat on a GPU plan", {
  skip_if_not_installed("scfastR")
  rd <- new_run(TWO, tag = "s2fastauto", engine = "auto")
  stage(rd, run_s0); write_decisions(rd)
  withr::local_envvar(CELLWARDEN_PLAN_JSON = '{"device":"cpu","plan":"cpu","reason":"t"}')
  stage(rd, run_s2)
  sm <- jsonlite::fromJSON(file.path(rd, "S2", "summary.json"))
  expect_identical(sm$path, "cpu_scfast"); expect_identical(sm$engine, "scfast")
  rd2 <- new_run(TWO, tag = "s2fastgpu", engine = "scfast", device = "gpu")
  stage(rd2, run_s0); write_decisions(rd2)
  withr::local_envvar(CELLWARDEN_PLAN_JSON = '{"device":"gpu","plan":"in_memory","reason":"t"}')
  expect_error(stage(rd2, run_s2), "no GPU backend")
})

test_that("S2 scfast path with harmony: R harmony runs in process, else falls back to exchange_cpu with a warning", {
  skip_if_not_installed("scfastR")
  rd <- new_run(TWO, tag = "s2fastharm", engine = "scfast", device = "auto",
                integration = list(method = "harmony", batch_key = "sample_id"))
  stage(rd, run_s0); write_decisions(rd)
  withr::local_envvar(CELLWARDEN_PLAN_JSON = '{"device":"cpu","plan":"cpu","reason":"t"}')
  if (scp_has_pkg("harmony")) {
    stage(rd, run_s2)
    sm <- jsonlite::fromJSON(file.path(rd, "S2", "summary.json"))
    expect_identical(sm$path, "cpu_scfast")
    expect_true("harmony" %in% Reductions(scp_load_object(file.path(rd, "S2"))))
  } else {
    skip_if_not(has_harmonypy, "harmonypy not available for the exchange fallback")
    out <- utils::capture.output(run_s2(scp_read_config(rd), REPO))
    expect_true(any(grepl("\\[WARN\\].*falling back to the exchange_cpu path", out)))
    sm <- jsonlite::fromJSON(file.path(rd, "S2", "summary.json"))
    expect_identical(sm$path, "exchange_cpu"); expect_identical(sm$engine, "seurat")
  }
})

test_that("S2 review_action exclude drops review cells; fixed pcs is honoured", {
  rd <- new_run(TWO, tag = "s2rev", qc = list(review_action = "exclude"), pcs = 15L, n_hvg = 300L)
  stage(rd, run_s0)
  dec <- rep(c("keep", "review", "keep", "exclude"), length.out = 500)
  write_decisions(rd, dec)
  withr::local_envvar(CELLWARDEN_PLAN_JSON = '{"device":"cpu","plan":"cpu","reason":"t"}')
  stage(rd, run_s2)
  obj <- scp_load_object(file.path(rd, "S2"))
  sm <- jsonlite::fromJSON(file.path(rd, "S2", "summary.json"))
  expect_equal(ncol(obj), sum(dec == "keep"))
  expect_identical(sm$pcs, 15L)
  expect_identical(ncol(Embeddings(obj, "pca")), 15L)
  expect_identical(sm$n_hvg, 300L)
})

test_that("S2 refuses mismatched QC decisions", {
  rd <- new_run(TWO, tag = "s2bad")
  stage(rd, run_s0)
  d <- write_decisions(rd)
  scp_write_tsv_gz(d[-1, ], file.path(rd, "S1", "qc_decisions.tsv.gz"))
  withr::local_envvar(CELLWARDEN_PLAN_JSON = '{"device":"cpu","plan":"cpu","reason":"t"}')
  expect_error(stage(rd, run_s2), "have no QC decision")
  d$cell_id[1] <- "nope"
  scp_write_tsv_gz(d, file.path(rd, "S1", "qc_decisions.tsv.gz"))
  expect_error(stage(rd, run_s2), "not in the S0 object")
})

test_that("harmony without the R package: clear error on device cpu", {
  skip_if(scp_has_pkg("harmony"), "R harmony is installed")
  rd <- new_run(TWO, tag = "s2harm", integration = list(method = "harmony", batch_key = "sample_id"))
  stage(rd, run_s0); write_decisions(rd)
  withr::local_envvar(CELLWARDEN_PLAN_JSON = '{"device":"cpu","plan":"cpu","reason":"t"}')
  expect_error(stage(rd, run_s2), "R 'harmony' package is not installed")
})

test_that("S2 GPU path (exchange, CPU fallback) gives reductions, graphs, clusters and ARI >= 0.7 vs CPU", {
  skip_if_no_py()
  skip_if(is.null(CPU_RUN), "CPU run did not complete")
  rd <- new_run(TWO, tag = "s2gpu", device = "gpu")
  stage(rd, run_s0)
  # identical decisions to the CPU run
  dir.create(file.path(rd, "S1"), showWarnings = FALSE)
  file.copy(file.path(CPU_RUN$rd, "S1", "qc_decisions.tsv.gz"), file.path(rd, "S1", "qc_decisions.tsv.gz"), overwrite = TRUE)
  withr::local_envvar(CELLWARDEN_ALLOW_CPU_AS_GPU = "1",
                      CELLWARDEN_PLAN_JSON = '{"device":"gpu","plan":"in_memory","reason":"test: forced gpu"}')
  stage(rd, run_s2)
  obj <- scp_load_object(file.path(rd, "S2"))
  sm <- jsonlite::fromJSON(file.path(rd, "S2", "summary.json"), simplifyVector = FALSE)
  expect_identical(sm$device, "gpu"); expect_identical(sm$plan, "in_memory"); expect_identical(sm$path, "gpu_exchange")
  expect_identical(sm$gpu_core_device, "cpu")
  expect_true(all(c("pca", "umap") %in% Reductions(obj)))
  expect_true(all(c("RNA_snn", "RNA_nn") %in% Graphs(obj)))
  expect_true("leiden" %in% colnames(obj[[]]))
  expect_identical(Idents(obj), obj$seurat_clusters)
  expect_identical(ncol(Embeddings(obj, "pca")), sm$pcs)
  expect_true(all(c("write_slim", "gpu_core", "merge") %in% names(sm$timings_s)))
  expect_true(file.exists(file.path(rd, "S2", "exchange", "slim.h5ad")))
  expect_length(validate_exchange(file.path(rd, "S2", "exchange", "results.h5ad")), 0)
  expect_length(obj@misc$cellwarden_exchange_merges, 1)
  expect_identical(colnames(obj), names(CPU_RUN$cl))
  a <- ari(CPU_RUN$cl, as.character(obj$seurat_clusters))
  expect_gte(a, 0.7)
  expect_gte(purity(obj$seurat_clusters, truth_for(obj)), 0.9)
})

test_that("harmony on device auto without R harmony runs through the exchange path with harmonypy", {
  skip_if(scp_has_pkg("harmony"), "R harmony is installed (CPU path is used instead)")
  skip_if_not(has_harmonypy, "harmonypy not available")
  rd <- new_run(TWO, tag = "s2hpy", device = "auto", integration = list(method = "harmony", batch_key = "sample_id"))
  stage(rd, run_s0); write_decisions(rd)
  withr::local_envvar(CELLWARDEN_PLAN_JSON = '{"device":"cpu","plan":"cpu","reason":"no gpu"}')
  stage(rd, run_s2)
  obj <- scp_load_object(file.path(rd, "S2"))
  sm <- jsonlite::fromJSON(file.path(rd, "S2", "summary.json"), simplifyVector = FALSE)
  expect_identical(sm$path, "exchange_cpu")
  expect_true("harmony" %in% Reductions(obj))
  expect_match(sm$integration$implementation, "harmony via cellwarden.gpu_core")
  expect_gte(purity(obj$seurat_clusters, truth_for(obj)), 0.9)
})

# ================================================================ entry point / common
test_that("run_stage.R exits non-zero with a clear message", {
  rs <- file.path(R.home("bin"), "Rscript")
  out <- suppressWarnings(system2(rs, c(file.path(REPO, "R", "stages", "run_stage.R"), "--stage", "S0",
                                        "--run-dir", file.path(tempdir(), "no-such-run")), stdout = TRUE, stderr = TRUE))
  expect_identical(attr(out, "status"), 1L)
  expect_true(any(grepl("\\[ERROR\\] stage S0 failed: run directory not found", out)))
  out2 <- suppressWarnings(system2(rs, c(file.path(REPO, "R", "stages", "run_stage.R"), "--stage", "S9",
                                         "--run-dir", tempdir()), stdout = TRUE, stderr = TRUE))
  expect_identical(attr(out2, "status"), 1L)
  expect_true(any(grepl("unknown stage 'S9'", out2)))
  # a real stage through the entry point
  td <- file.path(tempdir(), sprintf("cli-%s", basename(tempfile(""))))
  dir.create(td)
  rd <- new_run(data.frame(sample_id = "cli", path = write_mtx_dir(file.path(td, "c"), fx_sub())), tag = "cli")
  out3 <- system2(rs, c(file.path(REPO, "R", "stages", "run_stage.R"), "--stage", "S0", "--run-dir", rd),
                  stdout = TRUE, stderr = TRUE)
  expect_null(attr(out3, "status"))
  expect_true(any(grepl("^\\d{4}-\\d\\d-\\d\\dT\\d\\d:\\d\\d:\\d\\dZ \\[INFO\\] S0 done", out3)))
  expect_true(file.exists(file.path(rd, "S0", "object.rds")))
})

test_that("config validation reports every problem", {
  rd <- new_run(data.frame(sample_id = "a", path = "x"), tag = "cfg", device = "tpu", normalization = "bad",
                pcs = "many", n_hvg = 10L)
  err <- tryCatch(scp_read_config(rd), error = conditionMessage)
  for (k in c("device", "normalization", "pcs", "n_hvg")) expect_match(err, k)
})

test_that("object save/load: rds always; qs2 when installed, else falls back to rds", {
  d <- file.path(tempdir(), sprintf("obj-%s", basename(tempfile(""))))
  dir.create(d)
  o <- suppressWarnings(CreateSeuratObject(`rownames<-`(FX$m[1:50, 1:20], FX$feats$symbol[1:50])))
  quiet(scp_save_object(o, d, "rds"))
  expect_identical(colnames(scp_load_object(d)), colnames(o))
  quiet(p <- scp_save_object(o, d, "qs2"))
  if (scp_has_pkg("qs2")) {
    expect_true(file.exists(file.path(d, "object.qs2"))); expect_false(file.exists(file.path(d, "object.rds")))
  } else {
    expect_true(file.exists(file.path(d, "object.rds")))
  }
  expect_identical(colnames(scp_load_object(d)), colnames(o))
})

test_that("thread setup reads CELLWARDEN_THREADS", {
  withr::local_envvar(CELLWARDEN_THREADS = "3")
  expect_identical(scp_setup_threads(list(resources = list(threads = 8))), 3L)
  withr::local_envvar(CELLWARDEN_THREADS = "")
  expect_identical(scp_setup_threads(list(resources = list(threads = 2))), 2L)
})
