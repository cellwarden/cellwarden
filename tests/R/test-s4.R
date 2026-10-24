# Unit tests for stage S4 pseudobulk DGE (R/stages/s4_dge.R; contract in
# docs/agents-spec.md section 6).
#
#   Rscript -e 'testthat::test_file("tests/R/test-s4.R")'   # from repo root
#
# The stage_inputs fixture has two samples, both `control`, so it cannot
# support a two-condition design with >= 2 replicates per side. The tests
# build a synthetic S2-like object from tests/fixtures/pbmc_like (500 cells,
# 800 genes, four planted cell types): each donor's cells are split at random
# into two pseudo-samples, giving s1, s2 (control) and s3, s4 (treated), and a
# condition effect is planted on 30 genes in the treated samples (15 up:
# counts scaled x5 plus Poisson noise; 15 down: counts thinned to 15 %).
# DESeq2 and limma are installed in this workspace; edgeR is not needed.

suppressPackageStartupMessages({
  library(testthat)
  library(Matrix)
  library(Seurat)
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
source(file.path(REPO, "R", "stages", "run_stage.R"))
scp_source_stages(REPO)
options(Seurat.object.assay.version = "v5")
FIX <- file.path(REPO, "tests", "fixtures", "pbmc_like")

has_deseq2 <- scp_has_pkg("DESeq2")
has_limma <- scp_has_pkg("limma")

# ------------------------------------------------------------------ synthetic object
load_fixture <- function() {
  m <- as(as(readMM(file.path(FIX, "matrix.mtx")), "CsparseMatrix"), "dgCMatrix")
  feats <- read.delim(file.path(FIX, "features.tsv"), header = FALSE, colClasses = "character",
                      col.names = c("id", "symbol", "type"), quote = "")
  bcs <- readLines(file.path(FIX, "barcodes.tsv"))
  cells <- read.delim(file.path(FIX, "cells.tsv"), colClasses = "character", quote = "")
  dimnames(m) <- list(make.unique(feats$symbol), bcs)
  list(m = m, feats = feats, cells = cells)
}
FX <- load_fixture()

# Genes to plant: expressed (not the planted type markers, not MT-), the 30 highest-total of the rest.
pick_planted <- function(m) {
  markers <- unlist(jsonlite::fromJSON(file.path(FIX, "expected.json"))$marker_genes)
  tot <- Matrix::rowSums(m)
  cand <- names(tot)[!(names(tot) %in% markers) & !grepl("^MT-", names(tot)) & tot > 200]
  cand <- cand[order(-tot[cand])][1:30]
  list(up = cand[1:15], down = cand[16:30])
}
PLANTED <- pick_planted(FX$m)

#' Four pseudo-samples with a planted condition effect; returns a Seurat object
#' that looks like an S2 output (counts + data layers, seurat_clusters,
#' sample_id, condition, batch, donor), plus the truth.
make_object <- function(seed = 7, drop_nk_treated = FALSE) {
  set.seed(seed)
  m <- FX$m
  cells <- FX$cells
  sample <- character(ncol(m))
  for (d in unique(cells$sample)) {
    idx <- which(cells$sample == d)
    half <- sample(idx, floor(length(idx) / 2))
    sample[half] <- paste0(d, "_a"); sample[setdiff(idx, half)] <- paste0(d, "_b")
  }
  # donor_A_a, donor_B_a -> control; donor_A_b, donor_B_b -> treated (paired by donor, batch = donor)
  condition <- ifelse(grepl("_a$", sample), "control", "treated")
  treated <- which(condition == "treated")
  m <- as.matrix(m)
  up <- PLANTED$up; down <- PLANTED$down
  m[up, treated] <- m[up, treated] * 5 + matrix(rpois(length(up) * length(treated), 2), nrow = length(up))
  m[down, treated] <- matrix(rbinom(length(down) * length(treated), as.integer(m[down, treated]), 0.15), nrow = length(down))
  keep <- rep(TRUE, ncol(m))
  if (drop_nk_treated) keep <- !(cells$cell_type == "NK" & sample %in% c("donor_A_b", "donor_B_b"))
  m <- as(as(Matrix(m[, keep], sparse = TRUE), "CsparseMatrix"), "dgCMatrix")
  md <- data.frame(row.names = colnames(m),
                   sample_id = factor(sample[keep]), condition = factor(condition[keep], levels = c("control", "treated")),
                   donor = factor(cells$sample[keep]), batch = factor(cells$sample[keep]),
                   cell_type = factor(cells$cell_type[keep]),
                   seurat_clusters = factor(as.integer(factor(cells$cell_type[keep])) - 1L),
                   stringsAsFactors = FALSE)
  obj <- SeuratObject::CreateSeuratObject(SeuratObject::CreateAssay5Object(counts = m), assay = "RNA", meta.data = md)
  obj[["RNA"]] <- SeuratObject::AddMetaData(obj[["RNA"]], data.frame(gene_symbol_original = rownames(m), gene_id = FX$feats$id,
                                                                     row.names = rownames(m)))
  obj <- Seurat::NormalizeData(obj, verbose = FALSE)
  SeuratObject::Idents(obj) <- obj$seurat_clusters
  obj
}

# cluster -> cell type map used by the fake G3 decision (clusters are factor codes of cell_type)
CLUSTER_LABELS <- local({
  ct <- sort(unique(FX$cells$cell_type))
  setNames(ct, as.character(seq_along(ct) - 1L))
})

make_run <- function(obj, dge = list(), with_g3 = TRUE, g3_fine = TRUE) {
  rd <- tempfile("s4run_")
  dir.create(file.path(rd, "S2"), recursive = TRUE)
  dir.create(file.path(rd, "config"), recursive = TRUE)
  scp_save_object(obj, file.path(rd, "S2"), "rds")
  if (with_g3) {
    dir.create(file.path(rd, "gates", "G3"), recursive = TRUE)
    cl <- lapply(names(CLUSTER_LABELS), function(k) list(cluster = k, label_broad = CLUSTER_LABELS[[k]],
                                                          label_fine = if (g3_fine) paste0(CLUSTER_LABELS[[k]], " cell") else NULL,
                                                          cl_id = "CL:0000000", confidence = 0.9, status = "consensus"))
    scp_write_json(list(gate = "G3", mode = "auto", decided_by = "rules", decided_utc = .scp_utc(),
                        decision = list(clusters = cl), changes_vs_proposal = list()),
                   file.path(rd, "gates", "G3", "decision.json"))
  }
  cfg <- list(run_dir = rd, run_id = "test-run", config_hash = "deadbeef", seed = 42L, object_format = "rds",
              resources = list(threads = 1L), dge = dge)
  cfg
}

default_dge <- function(...) {
  d <- list(sample_col = "sample_id", condition_col = "condition", group_by = "cell_type_broad",
            contrasts = list(list(name = "treated_vs_control", numerator = "treated", denominator = "control",
                                  covariates = list("batch"))),
            # 10, not the config default 20: NK has 33 donor_A cells, so its pseudo-samples hold ~16 cells
            min_cells_per_sample = 10L, min_replicates = 2L, method = "deseq2", padj = 0.05, lfc = 0.5)
  over <- list(...)
  for (k in names(over)) d[[k]] <- over[[k]]
  d
}

read_dge <- function(path) {
  df <- read.delim(gzfile(path), stringsAsFactors = FALSE, check.names = FALSE)
  df
}

check_planted <- function(tab, label) {
  expect_equal(colnames(tab), c("gene", "baseMean", "log2fc", "lfcSE", "pvalue", "padj", "direction"), info = label)
  expect_true(all(tab$direction %in% c("up", "down", "ns")), info = label)
  up <- tab[tab$gene %in% PLANTED$up, ]; down <- tab[tab$gene %in% PLANTED$down, ]
  expect_equal(nrow(up), 15, info = label); expect_equal(nrow(down), 15, info = label)
  expect_true(all(up$log2fc > 0), info = paste(label, "up sign"))
  expect_true(all(down$log2fc < 0), info = paste(label, "down sign"))
  expect_true(mean(up$padj < 0.05) >= 0.9, info = paste(label, "up padj"))
  expect_true(mean(down$padj < 0.05) >= 0.9, info = paste(label, "down padj"))
  expect_true(mean(up$direction == "up") >= 0.9, info = paste(label, "up direction"))
  expect_true(mean(down$direction == "down") >= 0.9, info = paste(label, "down direction"))
  # not many false positives among the other genes
  other <- tab[!(tab$gene %in% c(PLANTED$up, PLANTED$down)), ]
  expect_lt(mean(other$direction != "ns"), 0.1, label = paste(label, "false positive rate"))
  # ordered by padj
  pa <- tab$padj[!is.na(tab$padj)]
  expect_true(!is.unsorted(pa), info = paste(label, "sorted by padj"))
}

# ------------------------------------------------------------------ helpers
test_that("s4_config fills defaults and validates", {
  d <- s4_config(list(dge = NULL))
  expect_equal(d$method, "findmarkers"); expect_equal(d$engine, "findmarkers"); expect_equal(d$test, "MAST")
  expect_equal(d$group_by, "cell_type_broad"); expect_equal(length(d$contrasts), 0)
  d <- s4_config(list(dge = list(method = "deseq2")))
  expect_equal(d$engine, "pseudobulk"); expect_equal(d$pseudobulk_method, "deseq2")
  d <- s4_config(list(dge = list(method = "pseudobulk", pseudobulk_method = "limma_voom")))
  expect_equal(d$engine, "pseudobulk"); expect_equal(d$pseudobulk_method, "limma_voom")
  expect_error(s4_config(list(dge = list(test = "magic"))), "dge.test")
  expect_error(s4_config(list(dge = list(contrasts = list(list(name = "a", numerator = "rest", denominator = "x"))))), "rest")
  d <- s4_config(list(dge = list(contrasts = list(list(name = "a", column = "sex", numerator = list("F"), denominator = "rest")))))
  expect_equal(d$contrasts[[1]]$column, "sex"); expect_equal(d$contrasts[[1]]$within, "cell_type_broad")
  expect_equal(s4_sides(c("F", "M", NA, "X"), "F", "rest"), c("num", "den", NA, "den"))
  expect_equal(s4_sides(c("a", "b", "c"), c("a", "b"), "c"), c("num", "num", "den"))
  expect_error(s4_config(list(dge = list(method = "edger"))), "dge.method")
  expect_error(s4_config(list(dge = list(contrasts = list(list(name = "a"))))), "numerator")
  expect_error(s4_config(list(dge = list(contrasts = list(list(name = "a/b", numerator = "x", denominator = "y"))))), "file names")
  d <- s4_config(list(dge = list(contrasts = list(list(name = "a", numerator = "x", denominator = "y", covariates = list("batch", "donor"))))))
  expect_equal(d$contrasts[[1]]$covariates, c("batch", "donor"))
})

test_that("s4_direction applies both thresholds and treats NA as ns", {
  expect_equal(s4_direction(c(1, -1, 0.2, 2, NA), c(0.01, 0.01, 0.01, 0.2, 0.01), 0.05, 0.5), c("up", "down", "ns", "ns", "ns"))
})

test_that("pseudobulk sums equal direct sums and units are group x sample", {
  obj <- make_object()
  pb <- s4_pseudobulk(obj, as.character(obj$cell_type), as.character(obj$sample_id))
  expect_equal(nrow(pb$units), 16)                       # 4 types x 4 samples
  expect_equal(pb$method, "Seurat::AggregateExpression")
  expect_equal(colnames(pb$counts), pb$units$pb_id)
  expect_equal(rownames(pb$counts), rownames(obj))
  m <- scp_counts(obj)
  for (i in c(1, 7, 16)) {
    cells <- colnames(obj)[obj$cell_type == pb$units$group[i] & obj$sample_id == pb$units$sample[i]]
    expect_equal(as.numeric(pb$counts[, i]), as.numeric(Matrix::rowSums(m[, cells, drop = FALSE])))
    expect_equal(pb$units$n_cells[i], length(cells))
  }
  expect_equal(sum(pb$units$n_cells), ncol(obj))
  # a single unit falls back to the direct sum (AggregateExpression refuses one level)
  one <- s4_pseudobulk(obj, rep("all", ncol(obj)), rep("s", ncol(obj)))
  expect_equal(one$method, "direct sum")
  expect_equal(as.numeric(one$counts[, 1]), as.numeric(Matrix::rowSums(m)))
})

test_that("s4_design drops single-level and rank-deficient covariates", {
  cd <- data.frame(condition = factor(c("control", "control", "treated", "treated"), levels = c("control", "treated")),
                   batch = factor(c("B1", "B1", "B1", "B1")), donor = factor(c("D1", "D2", "D1", "D2")),
                   confounded = factor(c("x", "x", "y", "y")))
  d <- s4_design(cd, "condition", c("batch"), 2L)
  expect_true(d$ok); expect_equal(d$covariates, character()); expect_match(d$notes[1], "one level")
  d <- s4_design(cd, "condition", c("confounded"), 2L)
  expect_true(d$ok); expect_equal(d$covariates, character()); expect_match(d$notes[1], "not full rank")
  d <- s4_design(cd, "condition", c("donor"), 2L)          # 4 samples, 3 coefficients: 1 residual df -> fine
  expect_true(d$ok); expect_equal(d$covariates, "donor")
  expect_equal(d$formula, ~donor + condition)
})

# ------------------------------------------------------------------ the stage
test_that("S4 finds the planted genes with DESeq2, writes every output, applies G3 labels", {
  skip_if_not(has_deseq2, "DESeq2 not installed")
  obj <- make_object()
  cfg <- make_run(obj, default_dge())
  rd <- cfg$run_dir
  summ <- run_s4(cfg, REPO)

  # labels from the G3 decision
  ann <- file.path(rd, "S3", "annotation.tsv")
  expect_true(file.exists(ann))
  a <- read.delim(ann, stringsAsFactors = FALSE)
  expect_equal(colnames(a), c("cluster", "label_broad", "label_fine", "cl_id", "confidence", "status"))
  expect_equal(nrow(a), 4)
  expect_equal(summ$labels$source, "gates/G3/decision.json")
  expect_equal(summ$group_by$used, "cell_type_broad"); expect_null(summ$group_by$note)
  expect_equal(summ$method$requested, "deseq2"); expect_equal(unlist(summ$method$used), "deseq2")
  expect_true(file.exists(file.path(rd, "S4", "gene_map.tsv")))
  gm <- read.delim(file.path(rd, "S4", "gene_map.tsv"), stringsAsFactors = FALSE)
  expect_equal(colnames(gm), c("gene", "symbol", "ensembl_id")); expect_equal(nrow(gm), nrow(obj))

  # pseudobulk outputs
  pbd <- file.path(rd, "S4", "pseudobulk", "treated_vs_control")
  expect_true(file.exists(file.path(pbd, "counts.tsv.gz"))); expect_true(file.exists(file.path(pbd, "samples.tsv")))
  st <- read.delim(file.path(pbd, "samples.tsv"), stringsAsFactors = FALSE)
  expect_true(all(c("pb_id", "group", "sample", "n_cells", "condition", "batch", "included", "exclude_reason") %in% colnames(st)))
  expect_equal(nrow(st), 16)
  cts <- read_dge(file.path(pbd, "counts.tsv.gz"))
  expect_equal(colnames(cts), c("gene", st$pb_id[st$included]))
  expect_equal(nrow(cts), nrow(obj))

  # summary structure
  s <- jsonlite::fromJSON(file.path(rd, "S4", "dge_summary.json"), simplifyVector = FALSE)
  expect_equal(s$stage, "S4"); expect_equal(s$run_id, "test-run")
  expect_equal(length(s$contrasts), 1)
  ct <- s$contrasts[[1]]
  expect_equal(ct$name, "treated_vs_control"); expect_equal(ct$numerator, "treated"); expect_equal(ct$denominator, "control")
  expect_equal(sort(unlist(ct$samples$numerator)), c("donor_A_b", "donor_B_b"))
  expect_equal(sort(unlist(ct$samples$denominator)), c("donor_A_a", "donor_B_a"))
  groups <- vapply(ct$groups, function(g) g$group, "")
  expect_equal(sort(groups), sort(unname(CLUSTER_LABELS)))
  for (g in ct$groups) {
    expect_null(g$skipped_reason)
    expect_true(all(c("n_up", "n_down", "n_tested", "top_up", "top_down", "file", "method_used", "n_samples") %in% names(g)))
    expect_equal(g$method_used, "deseq2")
    expect_equal(g$n_samples$numerator, 2); expect_equal(g$n_samples$denominator, 2)
    expect_true(g$n_tested > 100)
    expect_true(file.exists(file.path(rd, g$file)))
    expect_true(length(g$top_up) <= 10 && length(g$top_down) <= 10)
    expect_true(all(c("gene", "log2fc", "padj") %in% names(g$top_up[[1]])))
    # batch == donor here: with 2 donors and 2 conditions the covariate is estimable (paired design)
    expect_equal(unlist(g$covariates_used), "batch")
  }
  # planted genes in every cell type
  files <- list.files(file.path(rd, "S4", "dge"), pattern = "\\.tsv\\.gz$", full.names = TRUE)
  expect_equal(length(files), 4)
  expect_true(all(grepl("^treated_vs_control__", basename(files))))
  for (f in files) check_planted(read_dge(f), basename(f))
  tab <- read_dge(file.path(rd, ct$groups[[1]]$file))
  expect_equal(ct$groups[[1]]$n_up, sum(tab$direction == "up"))
  expect_equal(ct$groups[[1]]$n_down, sum(tab$direction == "down"))
  expect_equal(ct$groups[[1]]$top_up[[1]]$gene, tab$gene[tab$direction == "up"][1])
  unlink(rd, recursive = TRUE)
})

test_that("S4 skips a group without enough replicates per side and records why", {
  skip_if_not(has_deseq2, "DESeq2 not installed")
  obj <- make_object(drop_nk_treated = TRUE)
  cfg <- make_run(obj, default_dge(covariates = NULL))
  cfg$dge$contrasts[[1]]$covariates <- NULL
  rd <- cfg$run_dir
  summ <- run_s4(cfg, REPO)
  ct <- summ$contrasts[[1]]
  nk <- Filter(function(g) g$group == "NK", ct$groups)[[1]]
  expect_match(nk$skipped_reason, "0 treated and 2 control")
  expect_match(nk$skipped_reason, "min_replicates is 2")
  expect_null(nk$file); expect_equal(nk$n_up, 0L); expect_equal(nk$n_tested, 0L)
  expect_equal(length(nk$top_up), 0)
  expect_false(any(grepl("__NK", list.files(file.path(rd, "S4", "dge")))))
  others <- Filter(function(g) g$group != "NK", ct$groups)
  expect_true(all(vapply(others, function(g) is.null(g$skipped_reason), TRUE)))
  # min_cells_per_sample excludes small units: raise it so Mono (fewest cells) drops out
  cfg2 <- make_run(make_object(), default_dge(min_cells_per_sample = 60L))
  s2 <- run_s4(cfg2, REPO)
  st <- read.delim(file.path(cfg2$run_dir, "S4", "pseudobulk", "treated_vs_control", "samples.tsv"), stringsAsFactors = FALSE)
  expect_true(any(!st$included)); expect_true(all(nzchar(st$exclude_reason[!st$included])))
  skipped <- Filter(function(g) !is.null(g$skipped_reason), s2$contrasts[[1]]$groups)
  expect_true(length(skipped) >= 1)
  expect_match(skipped[[1]]$skipped_reason, ">= 60 cells")
  unlink(c(rd, cfg2$run_dir), recursive = TRUE)
})

test_that("method limma_voom works and is the fallback when DESeq2 is unavailable", {
  skip_if_not(has_limma, "limma not installed")
  obj <- make_object()
  cfg <- make_run(obj, default_dge(method = "limma_voom"))
  rd <- cfg$run_dir
  summ <- run_s4(cfg, REPO)
  expect_equal(unlist(summ$method$used), "limma_voom")
  for (g in summ$contrasts[[1]]$groups) expect_equal(g$method_used, "limma_voom")
  files <- list.files(file.path(rd, "S4", "dge"), pattern = "\\.tsv\\.gz$", full.names = TRUE)
  expect_equal(length(files), 4)
  for (f in files) check_planted(read_dge(f), basename(f))
  unlink(rd, recursive = TRUE)

  # fallback: pretend DESeq2 is missing
  pb <- s4_pseudobulk(obj, as.character(obj$cell_type), as.character(obj$sample_id))
  u <- pb$units[pb$units$group == "T", ]
  cd <- data.frame(sample = u$sample, condition = ifelse(grepl("_a$", u$sample), "control", "treated"), row.names = u$pb_id)
  orig <- scp_has_pkg
  scp_has_pkg <<- function(p) if (p == "DESeq2") FALSE else orig(p)
  on.exit(scp_has_pkg <<- orig, add = TRUE)
  r <- s4_dge_group(pb$counts[, u$pb_id], cd, "condition", "treated", "control", character(), "deseq2", 0.05, 0.5, 2L)
  expect_equal(r$method_used, "limma_voom")
  expect_match(r$notes[1], "DESeq2 is not installed")
  expect_equal(colnames(r$table), c("gene", "baseMean", "log2fc", "lfcSE", "pvalue", "padj", "direction"))
  check_planted(r$table, "fallback T")
})

test_that("group_by cluster works, and group_by falls back to seurat_clusters without labels", {
  skip_if_not(has_deseq2, "DESeq2 not installed")
  obj <- make_object()
  cfg <- make_run(obj, default_dge(group_by = "cluster"), with_g3 = FALSE)
  summ <- run_s4(cfg, REPO)
  expect_equal(summ$group_by$requested, "cluster"); expect_equal(summ$group_by$used, "seurat_clusters")
  expect_null(summ$group_by$note)
  expect_equal(summ$labels$source, "none")
  expect_equal(sort(vapply(summ$contrasts[[1]]$groups, function(g) g$group, "")), c("0", "1", "2", "3"))
  files <- list.files(file.path(cfg$run_dir, "S4", "dge"), pattern = "\\.tsv\\.gz$")
  expect_equal(sort(files), paste0("treated_vs_control__", 0:3, ".tsv.gz"))
  unlink(cfg$run_dir, recursive = TRUE)

  # default group_by cell_type_broad without G3 / annotation -> seurat_clusters with a note
  cfg <- make_run(obj, default_dge(), with_g3 = FALSE)
  summ <- run_s4(cfg, REPO)
  expect_equal(summ$group_by$used, "seurat_clusters")
  expect_match(summ$group_by$note, "not a cell metadata column")
  expect_true(any(grepl("labels are not available", unlist(summ$notes))))
  expect_false(file.exists(file.path(cfg$run_dir, "S3", "annotation.tsv")))
  unlink(cfg$run_dir, recursive = TRUE)

  # an existing S3/annotation.tsv is used when there is no G3 decision
  cfg <- make_run(obj, default_dge(), with_g3 = FALSE)
  dir.create(file.path(cfg$run_dir, "S3"))
  write.table(data.frame(cluster = names(CLUSTER_LABELS), label_broad = unname(CLUSTER_LABELS), label_fine = unname(CLUSTER_LABELS)),
              file.path(cfg$run_dir, "S3", "annotation.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  summ <- run_s4(cfg, REPO)
  expect_equal(summ$labels$source, "S3/annotation.tsv"); expect_equal(summ$group_by$used, "cell_type_broad")
  unlink(cfg$run_dir, recursive = TRUE)
})

test_that("S4 validates contrasts against the data and handles an empty contrast list", {
  obj <- make_object()
  cfg <- make_run(obj, default_dge(contrasts = list(list(name = "x", numerator = "disease", denominator = "control"))))
  expect_error(run_s4(cfg, REPO), "not a value of 'condition'")
  unlink(cfg$run_dir, recursive = TRUE)
  cfg <- make_run(obj, default_dge(condition_col = "nope"))
  expect_error(run_s4(cfg, REPO), "not in the cell metadata")
  unlink(cfg$run_dir, recursive = TRUE)
  cfg <- make_run(obj, default_dge(contrasts = list()))
  summ <- run_s4(cfg, REPO)
  expect_equal(length(summ$contrasts), 0)
  expect_true(any(grepl("contrasts is empty", unlist(summ$notes))))
  expect_true(file.exists(file.path(cfg$run_dir, "S4", "dge_summary.json")))
  unlink(cfg$run_dir, recursive = TRUE)
})

test_that("FindMarkers (default engine): markers of each cell type when no contrast is set", {
  obj <- make_object()
  cfg <- make_run(obj, list(test = "wilcox", min_cells_per_side = 5L))
  summ <- run_s4(cfg, REPO)
  expect_equal(summ$method$engine, "findmarkers"); expect_equal(unlist(summ$method$used), "findmarkers:wilcox")
  expect_equal(length(summ$contrasts), 1)
  ct <- summ$contrasts[[1]]
  expect_equal(ct$name, "markers"); expect_equal(ct$within, "none")
  expect_equal(sort(vapply(ct$groups, function(g) g$group, "")), sort(unname(unlist(CLUSTER_LABELS))))
  markers <- jsonlite::fromJSON(file.path(FIX, "expected.json"))$marker_genes
  for (g in ct$groups) {
    expect_equal(g$denominator, "rest")
    tab <- read_dge(file.path(cfg$run_dir, g$file))
    expect_true(all(c("gene", "baseMean", "log2fc", "lfcSE", "pvalue", "padj", "direction", "pct_1", "pct_2") %in% colnames(tab)))
    expect_gt(g$n_up, 0)
    expect_equal(g$n_cells$numerator + g$n_cells$denominator, ncol(obj))
  }
  # the planted marker genes of each type are among its top markers
  for (g in ct$groups) {
    top <- vapply(g$top_up, function(x) x$gene, "")
    expect_true(any(markers[[g$group]] %in% top), info = g$group)
  }
  unlink(cfg$run_dir, recursive = TRUE)
})

test_that("FindMarkers compares two groups of any column, within each cell type or across all cells", {
  obj <- make_object()
  cfg <- make_run(obj, list(test = "wilcox", min_cells_per_side = 5L, contrasts = list(
    list(name = "treated_vs_control", column = "condition", numerator = "treated", denominator = "control"),
    list(name = "donorB_all", column = "donor", numerator = "donor_B", denominator = "rest", within = "none"))))
  summ <- run_s4(cfg, REPO)
  expect_equal(vapply(summ$contrasts, function(c) c$name, ""), c("treated_vs_control", "donorB_all"))  # no markers: contrasts set
  tc <- summ$contrasts[[1]]
  expect_equal(tc$column, "condition"); expect_equal(tc$within, "cell_type_broad")
  found_up <- unique(unlist(lapply(tc$groups, function(g) {
    if (is.null(g$file)) return(character())
    t <- read_dge(file.path(cfg$run_dir, g$file)); t$gene[t$direction == "up"]
  })))
  expect_gt(mean(PLANTED$up %in% found_up), 0.6)
  dn <- summ$contrasts[[2]]
  expect_equal(length(dn$groups), 1); expect_equal(dn$groups[[1]]$group, "all cells")
  expect_equal(unlist(dn$samples$numerator), c("donor_B_a", "donor_B_b"))
  unlink(cfg$run_dir, recursive = TRUE)
  # markers: true adds the per-group markers to the configured contrasts
  cfg <- make_run(obj, list(test = "wilcox", markers = TRUE, min_cells_per_side = 5L, contrasts = list(
    list(name = "tc", numerator = "treated", denominator = "control", within = "none"))))
  expect_equal(vapply(run_s4(cfg, REPO)$contrasts, function(c) c$name, ""), c("tc", "markers"))
  unlink(cfg$run_dir, recursive = TRUE)
})

test_that("pseudobulk takes any sample-level column and refuses cell-level ones", {
  skip_if_not(has_deseq2, "DESeq2 not installed")
  obj <- make_object()
  cfg <- make_run(obj, default_dge(contrasts = list(list(name = "tc", column = "condition", numerator = "treated",
                                                         denominator = "rest"))))
  summ <- run_s4(cfg, REPO)
  expect_equal(summ$method$engine, "pseudobulk"); expect_equal(summ$contrasts[[1]]$denominator, "rest")
  expect_true(any(vapply(summ$contrasts[[1]]$groups, function(g) !is.null(g$file), TRUE)))
  unlink(cfg$run_dir, recursive = TRUE)
  cfg <- make_run(obj, default_dge(contrasts = list(list(name = "ct", column = "cell_type_broad", numerator = "B",
                                                         denominator = "rest", within = "none"))))
  expect_error(run_s4(cfg, REPO), "one value per sample")
  unlink(cfg$run_dir, recursive = TRUE)
})

test_that("run_stage.R knows S4", {
  expect_true("S4" %in% .SCP_STAGES)
})
