#!/usr/bin/env Rscript
# R half of the Seurat <-> AnnData round-trip check (tests/integration).
#
# Driven by run_roundtrip.sh; each subcommand stops with a non-zero exit on
# the first failed batch of assertions.
#
#   Rscript roundtrip.R write        WORKDIR   # step 1: fixture -> Seurat -> full + slim
#   Rscript roundtrip.R merge        WORKDIR   # step 4: merge results + refuse cases
#   Rscript roundtrip.R read-python  WORKDIR   # step 5: read the Python-written full file

suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: roundtrip.R {write|merge|read-python} WORKDIR")
cmd <- args[[1]]
work <- normalizePath(args[[2]], mustWork = TRUE)

script_dir <- local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  dirname(normalizePath(f))
})
repo <- normalizePath(file.path(script_dir, "..", ".."))
fixture <- file.path(repo, "tests", "fixtures", "pbmc_like")
expected <- fromJSON(file.path(fixture, "expected.json"), simplifyVector = TRUE)
source(file.path(repo, "R", "exchange", "exchange.R"))

RUN_ID <- "roundtrip-it"
CONFIG_HASH <- "none"
N_HVG <- 500L
N_PCS <- 30L   # must agree with run_roundtrip.sh / roundtrip.py

failures <- character()
check <- function(cond, msg) {
  ok <- isTRUE(tryCatch(all(cond), error = function(e) FALSE))
  cat(if (ok) "  ok    " else "  FAIL  ", msg, "\n", sep = "")
  if (!ok) failures <<- c(failures, msg)
  invisible(ok)
}
finish <- function(step) {
  if (length(failures)) {
    message(step, ": ", length(failures), " check(s) failed")
    quit(status = 1, save = "no")
  }
  cat(step, ": all checks passed\n", sep = "")
}
# Expect an error; returns the message (or NA if none was raised).
expect_err <- function(expr) {
  tryCatch({ force(expr); NA_character_ }, error = function(e) conditionMessage(e))
}

load_fixture <- function() {
  m <- readMM(file.path(fixture, "matrix.mtx"))
  m <- as(as(m, "CsparseMatrix"), "dgCMatrix")
  feats <- read.delim(file.path(fixture, "features.tsv"), header = FALSE,
                      col.names = c("gene_id", "symbol", "feature_type"),
                      colClasses = "character", quote = "")
  bcs <- readLines(file.path(fixture, "barcodes.tsv"))
  cells <- read.delim(file.path(fixture, "cells.tsv"), colClasses = "character", quote = "")
  dimnames(m) <- list(feats$symbol, bcs)
  list(m = m, feats = feats, bcs = bcs, cells = cells)
}

build_seurat <- function(fx) {
  meta <- data.frame(
    sample = factor(fx$cells$sample, levels = expected$obs_levels$sample),
    cell_type = factor(fx$cells$cell_type, levels = expected$obs_levels$cell_type),
    row.names = fx$bcs
  )
  # Seurat rewrites '_' -> '-' here and warns; that is exactly what we test.
  obj <- suppressWarnings(CreateSeuratObject(fx$m, assay = "RNA", meta.data = meta,
                                             min.cells = 0, min.features = 0))
  fmeta <- data.frame(gene_symbol_original = fx$feats$symbol,
                      gene_id = fx$feats$gene_id,
                      row.names = rownames(obj))
  obj[["RNA"]] <- AddMetaData(obj[["RNA"]], metadata = fmeta)
  obj <- NormalizeData(obj, verbose = FALSE)
  obj <- FindVariableFeatures(obj, nfeatures = N_HVG, verbose = FALSE)
  obj
}

# ---------------------------------------------------------------------- step 1
step_write <- function() {
  fx <- load_fixture()
  obj <- build_seurat(fx)
  seurat_names <- gsub("_", "-", fx$feats$symbol, fixed = TRUE)
  check(identical(rownames(obj), seurat_names), "Seurat renamed '_' to '-' as expected (names in order)")
  check(identical(colnames(obj), fx$bcs), "Seurat cell order == fixture barcodes")
  check(all(dim(obj) == c(expected$n_genes, expected$n_cells)), sprintf("Seurat object is %d x %d", expected$n_genes, expected$n_cells))
  check(length(VariableFeatures(obj)) == N_HVG, sprintf("%d variable features", N_HVG))

  cs <- sc_counts_checksum(LayerData(obj, assay = "RNA", layer = "counts"))
  check(identical(cs, expected$counts_checksum),
        sprintf("sc_counts_checksum(counts) == expected (%s)", cs))
  cs_raw <- sc_counts_checksum(fx$m)
  check(identical(cs_raw, expected$counts_checksum), "sc_counts_checksum(raw readMM matrix) == expected")

  writeLines(VariableFeatures(obj), file.path(work, "hvgs.txt"))
  for (mode in c("full", "slim")) {
    p <- file.path(work, sprintf("r_%s.h5ad", mode))
    if (file.exists(p)) unlink(p)
    err <- expect_err(write_exchange(obj, p, mode = mode, run_id = RUN_ID,
                                     stage = sprintf("IT_R_%s", mode), config_hash = CONFIG_HASH))
    check(is.na(err), sprintf("write_exchange(mode='%s') succeeded %s", mode, ifelse(is.na(err), "", err)))
    probs <- tryCatch(validate_exchange(p), error = function(e) paste("validate_exchange error:", conditionMessage(e)))
    check(is.character(probs) && length(probs) == 0,
          sprintf("R validate_exchange(r_%s.h5ad) is empty %s", mode, paste(probs, collapse = "; ")))
  }
  # R reads back its own full file.
  rd <- tryCatch(read_exchange_seurat(file.path(work, "r_full.h5ad")),
                 error = function(e) { cat("  read_exchange_seurat error: ", conditionMessage(e), "\n"); NULL })
  check(!is.null(rd), "read_exchange_seurat(r_full.h5ad) succeeded")
  if (!is.null(rd)) {
    check(identical(sc_counts_checksum(LayerData(rd, assay = "RNA", layer = "counts")), expected$counts_checksum),
          "R->R: counts checksum preserved")
    check(identical(colnames(rd), fx$bcs) && identical(rownames(rd), seurat_names), "R->R: dimnames preserved")
  }
  saveRDS(obj, file.path(work, "seurat_original.rds"))
  finish("write")
}

# ---------------------------------------------------------------------- step 4
step_merge <- function() {
  obj <- readRDS(file.path(work, "seurat_original.rds"))
  cells_before <- colnames(obj)
  cs_before <- sc_counts_checksum(LayerData(obj, assay = "RNA", layer = "counts"))
  res_path <- file.path(work, "results.h5ad")
  ck <- local({   # read uns/exchange/cluster_key straight from HDF5
    h <- hdf5r::H5File$new(res_path, mode = "r"); on.exit(h$close_all())
    h[["uns/exchange/cluster_key"]]$read()
  })
  py_cl <- read.delim(file.path(work, "results_clusters.tsv"), colClasses = "character")
  py_cl <- setNames(py_cl$cluster, py_cl$barcode)

  merged <- tryCatch(merge_exchange_results(obj, res_path),
                     error = function(e) { cat("  merge error: ", conditionMessage(e), "\n"); NULL })
  check(!is.null(merged), "merge_exchange_results(original, results.h5ad) succeeded")
  if (is.null(merged)) finish("merge")
  n <- expected$n_cells
  check(identical(colnames(merged), cells_before), "cell order unchanged after merge")
  check(identical(sc_counts_checksum(LayerData(merged, assay = "RNA", layer = "counts")), cs_before),
        "counts unchanged after merge")
  check("pca" %in% Reductions(merged), "reduction 'pca' exists")
  check("umap" %in% Reductions(merged), "reduction 'umap' exists")
  if ("pca" %in% Reductions(merged)) {
    e <- Embeddings(merged, "pca")
    check(all(dim(e) == c(n, N_PCS)), sprintf("pca embeddings %d x %d (got %s)", n, N_PCS, paste(dim(e), collapse = "x")))
    check(identical(rownames(e), cells_before), "pca rows are the object's cells in order")
    check(Key(merged[["pca"]]) == "PC_", sprintf("pca key 'PC_' (got '%s')", Key(merged[["pca"]])))
  }
  if ("umap" %in% Reductions(merged)) {
    e <- Embeddings(merged, "umap")
    check(all(dim(e) == c(n, 2L)), sprintf("umap embeddings %d x 2", n))
    check(Key(merged[["umap"]]) == "UMAP_", sprintf("umap key 'UMAP_' (got '%s')", Key(merged[["umap"]])))
  }
  check("harmony" %in% Reductions(merged), "reduction 'harmony' exists (gpu_core ran with --batch-key)")
  if ("harmony" %in% Reductions(merged)) {
    check(all(dim(Embeddings(merged, "harmony")) == c(n, N_PCS)), sprintf("harmony embeddings %d x %d", n, N_PCS))
    check(Key(merged[["harmony"]]) == "harmony_", sprintf("harmony key 'harmony_' (got '%s')", Key(merged[["harmony"]])))
  }
  check(all(c("RNA_snn", "RNA_nn") %in% Graphs(merged)),
        sprintf("graphs RNA_snn and RNA_nn exist (got %s)", paste(Graphs(merged), collapse = ",")))
  for (g in intersect(c("RNA_snn", "RNA_nn"), Graphs(merged))) {
    G <- merged[[g]]
    check(all(dim(G) == c(n, n)) && identical(rownames(G), cells_before) && identical(colnames(G), cells_before),
          sprintf("graph %s is %d x %d with the object's cells", g, n, n))
  }
  md <- merged[[]]
  cl_col <- intersect(c(ck, "seurat_clusters"), colnames(md))
  check(length(cl_col) > 0, sprintf("cluster column ('%s' or 'seurat_clusters') in meta.data", ck))
  if (length(cl_col)) {
    col <- cl_col[[1]]
    cat("        cluster column used: ", col, "\n", sep = "")
    check(is.factor(md[[col]]), sprintf("meta.data$%s is a factor", col))
    check(identical(as.character(md[[col]]), unname(py_cl[rownames(md)])),
          sprintf("meta.data$%s matches results by barcode", col))
    check(identical(as.character(Idents(merged)), as.character(md[[col]])) &&
            identical(names(Idents(merged)), cells_before),
          "Idents() set to the cluster labels, in cell order")
  }
  check(identical(levels(md$sample), expected$obs_levels$sample) &&
          identical(levels(md$cell_type), expected$obs_levels$cell_type),
        "original meta.data factor levels untouched")

  # Merge is barcode-based, not positional: the same results with rows in a
  # different order must give the same object. (Reordering the Seurat object
  # instead would change its counts checksum - spec 6 is order-sensitive - and
  # rule 7.2 would rightly refuse the merge.)
  shuf_path <- file.path(work, "results_shuffled.h5ad")
  m2 <- tryCatch(merge_exchange_results(obj, shuf_path),
                 error = function(e) { cat("  shuffled-results merge error: ", conditionMessage(e), "\n"); NULL })
  check(!is.null(m2), "merge of row-shuffled results succeeded")
  if (!is.null(m2)) {
    check(identical(colnames(m2), cells_before), "cell order unchanged after merging shuffled results")
    for (r in intersect(c("pca", "harmony", "umap"), intersect(Reductions(m2), Reductions(merged)))) {
      check(isTRUE(all.equal(Embeddings(m2, r), Embeddings(merged, r))),
            sprintf("%s embeddings identical whatever the results row order (matched by barcode)", r))
    }
    for (g in intersect(c("RNA_snn", "RNA_nn"), intersect(Graphs(m2), Graphs(merged)))) {
      check(isTRUE(all.equal(as(m2[[g]], "dgCMatrix"), as(merged[[g]], "dgCMatrix"))),
            sprintf("graph %s identical whatever the results row order", g))
    }
    check(identical(as.character(Idents(m2)), as.character(Idents(merged))),
          "Idents identical whatever the results row order")
  }

  # Refuse cases ---------------------------------------------------------------
  err <- expect_err(merge_exchange_results(merged, res_path))
  check(!is.na(err), sprintf("re-merge without overwrite=TRUE errors [%s]", substr(err, 1, 120)))
  err <- expect_err(merge_exchange_results(merged, res_path, overwrite = TRUE))
  check(is.na(err), sprintf("re-merge with overwrite=TRUE succeeds %s", ifelse(is.na(err), "", err)))

  err <- expect_err(merge_exchange_results(obj, file.path(work, "results_badchecksum.h5ad")))
  check(!is.na(err), sprintf("altered source_counts_checksum is refused [%s]", substr(err, 1, 120)))
  err <- expect_err(merge_exchange_results(obj, file.path(work, "results_badchecksum.h5ad"),
                                           overwrite = TRUE, allow_subset = TRUE))
  check(!is.na(err), "altered source_counts_checksum refused even with overwrite/allow_subset")

  sub_path <- file.path(work, "results_subset.h5ad")
  dropped <- readLines(file.path(work, "dropped_barcode.txt"))
  err <- expect_err(merge_exchange_results(obj, sub_path))
  check(!is.na(err), sprintf("results missing barcode %s refused by default [%s]", dropped, substr(err, 1, 120)))
  ms <- tryCatch(merge_exchange_results(obj, sub_path, allow_subset = TRUE),
                 error = function(e) { cat("  allow_subset merge error: ", conditionMessage(e), "\n"); NULL })
  check(!is.null(ms), "results missing one barcode merge with allow_subset=TRUE")
  if (!is.null(ms)) {
    check(identical(colnames(ms), cells_before), "allow_subset merge keeps all object cells in order")
    mdc <- ms[[]]
    col <- intersect(c(ck, "seurat_clusters"), colnames(mdc))
    if (length(col)) {
      v <- as.character(mdc[[col[[1]]]]); names(v) <- rownames(mdc)
      keep <- setdiff(cells_before, dropped)
      check(identical(unname(v[keep]), unname(py_cl[keep])), "allow_subset: kept cells' clusters match by barcode")
      check(is.na(v[[dropped]]), sprintf("allow_subset: dropped cell %s has NA cluster", dropped))
    }
  }
  # A barcode not in the object must always be refused (spec 7.3), even with
  # allow_subset, and even though the source checksum is correct.
  err <- expect_err(merge_exchange_results(obj, file.path(work, "results_extra.h5ad"), allow_subset = TRUE))
  check(!is.na(err) && !grepl("checksum", err, ignore.case = TRUE),
        sprintf("results with a barcode unknown to the object refused (not by checksum) [%s]", substr(err, 1, 120)))
  finish("merge")
}

# ---------------------------------------------------------------------- step 5
step_read_python <- function() {
  fx <- load_fixture()
  p <- file.path(work, "py_full.h5ad")
  probs <- tryCatch(validate_exchange(p), error = function(e) paste("validate_exchange error:", conditionMessage(e)))
  check(is.character(probs) && length(probs) == 0,
        sprintf("R validate_exchange(py_full.h5ad) is empty %s", paste(probs, collapse = "; ")))
  lst <- tryCatch(read_exchange(p), error = function(e) { cat("  read_exchange error: ", conditionMessage(e), "\n"); NULL })
  check(!is.null(lst), "read_exchange(py_full.h5ad) succeeded")
  if (!is.null(lst)) {
    check(identical(lst$uns$exchange$writer, "python"), "uns exchange writer == 'python'")
    check(identical(lst$uns$exchange$counts_checksum, expected$counts_checksum), "uns counts_checksum == expected")
  }
  obj <- tryCatch(read_exchange_seurat(p),
                  error = function(e) { cat("  read_exchange_seurat error: ", conditionMessage(e), "\n"); NULL })
  check(!is.null(obj), "read_exchange_seurat(py_full.h5ad) succeeded")
  if (is.null(obj)) finish("read-python")
  counts <- LayerData(obj, assay = "RNA", layer = "counts")
  cs <- sc_counts_checksum(counts)
  check(identical(cs, expected$counts_checksum), sprintf("Python->R counts checksum == expected (%s)", cs))
  check(identical(colnames(obj), fx$bcs), "cell barcodes in fixture order")
  seurat_names <- gsub("_", "-", fx$feats$symbol, fixed = TRUE)
  check(identical(rownames(obj), seurat_names), "feature names are Seurat-safe ('_' -> '-') in order")
  fm <- obj[["RNA"]][[]]
  check("gene_symbol_original" %in% colnames(fm) &&
          identical(as.character(fm[seurat_names, "gene_symbol_original"]), fx$feats$symbol),
        "gene_symbol_original kept as feature metadata")
  check("gene_id" %in% colnames(fm) && identical(as.character(fm[seurat_names, "gene_id"]), fx$feats$gene_id),
        "gene_id kept as feature metadata")
  dat <- LayerData(obj, assay = "RNA", layer = "data")
  ref <- log1p(t(t(counts) / colSums(counts)) * 1e4)
  check(max(abs(dat - ref)) < 1e-4, "data layer == log1p normalised counts")
  md <- obj[[]]
  for (col in c("sample", "cell_type")) {
    check(is.factor(md[[col]]) && identical(levels(md[[col]]), expected$obs_levels[[col]]),
          sprintf("meta.data$%s factor levels preserved in order (%s)", col,
                  paste(levels(md[[col]]), collapse = ",")))
    check(identical(as.character(md[[col]]), fx$cells[[col]]), sprintf("meta.data$%s values match fixture", col))
  }
  check(is.logical(md$is_donor_A), sprintf("bool obs -> logical (got %s)", class(md$is_donor_A)[1]))
  check(is.integer(md$n_counts) || is.numeric(md$n_counts), "int obs -> integer/numeric")
  check(is.character(md$label), sprintf("string obs -> character (got %s)", class(md$label)[1]))
  hv <- VariableFeatures(obj)
  check(setequal(hv, gsub("_", "-", expected$marker_genes_flat)), "var highly_variable -> VariableFeatures()")
  finish("read-python")
}

switch(cmd,
  "write" = step_write(),
  "merge" = step_merge(),
  "read-python" = step_read_python(),
  stop("unknown command: ", cmd))
