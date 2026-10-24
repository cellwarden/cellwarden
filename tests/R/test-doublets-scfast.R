# S1a with the scfast engine: scrublet-style doublets + pre-clustering
# (R/stages/doublets_scfast.R, docs/scfast-integration.md "S1b doublets and pre-clustering")
# and the engine-resolution helper scp_resolve_engine() (R/stages/common.R).
#
#   Rscript -e 'testthat::test_file("tests/R/test-doublets-scfast.R")'   # from repo root
#
# Tests that need scfastR are skipped when it is not installed (R_LIBS may point at its library).
# Input: the two-sample QC-stress fixture tests/fixtures/stage_inputs (donor_A 10x mtx, donor_B
# .h5ad) with truth.tsv: 10 planted doublets per sample (sum of two cells of different types).

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
INPUTS <- file.path(REPO, "tests", "fixtures", "stage_inputs")
HAS_SCFAST <- requireNamespace("scfastR", quietly = TRUE)
skip_if_no_scfast <- function() skip_if_not(HAS_SCFAST, "scfastR is not installed")
quiet <- function(expr) { out <- NULL; utils::capture.output(out <- force(expr)); out }

# ------------------------------------------------------------------ scp_resolve_engine
test_that("scp_resolve_engine: seurat is always seurat; bad values are errors", {
  expect_identical(scp_resolve_engine(list(engine = "seurat")), "seurat")
  expect_identical(scp_resolve_engine(list(engine = "seurat"), device = "gpu"), "seurat")
  expect_error(scp_resolve_engine(list(engine = "rust")), "engine must be one of")
  expect_error(scp_resolve_engine(list(engine = c("auto", "seurat"))), "engine must be one of")
})

test_that("scp_resolve_engine: scfast on a GPU plan is an error, auto falls back to seurat", {
  expect_error(scp_resolve_engine(list(engine = "scfast"), device = "gpu"), "no GPU backend")
  expect_identical(scp_resolve_engine(list(engine = "auto"), device = "gpu"), "seurat")
  expect_identical(scp_resolve_engine(list(), device = "gpu"), "seurat")     # missing engine = auto
})

test_that("scp_resolve_engine: auto and scfast follow scfastR availability on the CPU", {
  if (HAS_SCFAST) {
    expect_identical(scp_resolve_engine(list(engine = "auto")), "scfast")
    expect_identical(scp_resolve_engine(list()), "scfast")
    expect_identical(scp_resolve_engine(list(engine = "scfast"), device = "cpu"), "scfast")
    expect_identical(scp_resolve_engine(list(engine = "auto"), device = "cpu"), "scfast")
  } else {
    expect_identical(scp_resolve_engine(list(engine = "auto")), "seurat")
    expect_error(scp_resolve_engine(list(engine = "scfast")), "scfastR")
  }
})

# ------------------------------------------------------------------ the detector on the fixture
load_donor_a <- function() {
  m <- as(as(readMM(gzfile(file.path(INPUTS, "donor_A", "matrix.mtx.gz"))), "CsparseMatrix"), "dgCMatrix")
  bc <- readLines(gzfile(file.path(INPUTS, "donor_A", "barcodes.tsv.gz")))
  colnames(m) <- paste0("donor_A_", bc)
  m
}
truth <- function() read.delim(file.path(INPUTS, "truth.tsv"), colClasses = "character")

test_that("detector: planted doublets are recalled with few false positives; clusters follow cell types", {
  skip_if_no_scfast()
  m <- load_donor_a()
  tr <- truth()
  tr <- tr[match(colnames(m), tr$cell_id), ]
  expect_false(anyNA(tr$injected))
  r <- quiet(scp_doublets_scfast(m, rep("donor_A", ncol(m)), seed = 42L))
  tab <- r$table
  expect_identical(tab$cell_id, colnames(m))
  expect_true(is.numeric(tab$doublet_score) && is.logical(tab$doublet_call) && is.character(tab$pre_cluster))
  expect_false(anyNA(tab$doublet_score))
  dbl <- tr$injected == "doublet"
  expect_equal(sum(dbl), 10)
  recall <- mean(tab$doublet_call[dbl])
  fp <- mean(tab$doublet_call[tr$injected == "none"])
  expect_gte(recall, 0.5)
  expect_lte(fp, 0.10)
  # the score ranks planted doublets above ordinary cells (AUC)
  y <- tab$doublet_score[dbl]; z <- tab$doublet_score[tr$injected == "none"]
  auc <- mean(outer(y, z, ">") + 0.5 * outer(y, z, "=="))
  expect_gte(auc, 0.85)
  # pre-clusters: a handful of clusters, each dominated by one cell type among singlets
  s <- tr$injected == "none"
  expect_gte(length(unique(tab$pre_cluster)), 3)
  expect_lte(length(unique(tab$pre_cluster)), 15)
  tb <- table(tab$pre_cluster[s], tr$cell_type[s])
  expect_gte(sum(apply(tb, 1, max)) / sum(tb), 0.85)
})

test_that("detector: deterministic for a fixed seed; small samples are skipped with NA/FALSE", {
  skip_if_no_scfast()
  m <- load_donor_a()
  sid <- rep("big", ncol(m)); sid[1:40] <- "tiny"
  a <- quiet(scp_doublets_scfast(m, sid, seed = 7L))
  b <- quiet(scp_doublets_scfast(m, sid, seed = 7L))
  expect_identical(a$table, b$table)
  tiny <- sid == "tiny"
  expect_true(all(is.na(a$table$doublet_score[tiny])))
  expect_false(any(a$table$doublet_call[tiny]))
  expect_false(anyNA(a$table$doublet_score[!tiny]))
  expect_true(any(grepl("sample tiny: 40 cells < 100", a$notes)))
  f <- scp_dbl_format(a$table)
  expect_true(all(f$doublet_score[tiny] == "NA"))
  expect_true(all(f$doublet_call %in% c("TRUE", "FALSE")))
})

# ------------------------------------------------------------------ S1a end to end
new_run <- function(engine, method = "scrublet", tag = "dbl") {
  rd <- file.path(tempdir(), sprintf("cellwarden-%s-%s", tag, basename(tempfile(""))))
  dir.create(file.path(rd, "config"), recursive = TRUE)
  sheet <- read.csv(file.path(INPUTS, "samples.csv"), colClasses = "character")
  sheet$path <- file.path(INPUTS, sheet$path)
  write.csv(sheet, file.path(rd, "config", "samples.csv"), row.names = FALSE, quote = TRUE)
  file.copy(file.path(REPO, "tests", "integration", "stage_configs", "qc_policy.yaml"),
            file.path(rd, "config", "qc_policy.yaml"))
  cfg <- list(project = "test", run_dir = rd, run_id = paste0("test-", tag), config_hash = strrep("a", 64),
              seed = 42L, samples = file.path(rd, "config", "samples.csv"),
              qc_policy = file.path(rd, "config", "qc_policy.yaml"), object_format = "rds",
              resources = list(threads = 1L, max_memory_gb = 4, gpu_mem_cap_gb = 2, nice = 10L),
              doublets = list(method = method), qc = list(review_action = "keep"),
              normalization = "lognorm", n_hvg = 500L, device = "cpu", engine = engine,
              integration = list(method = "none", batch_key = "sample_id"),
              pcs = "auto", n_neighbors = 20L, clustering = list(resolution = 0.5))
  yaml::write_yaml(cfg, file.path(rd, "config", "config.resolved.yaml"))
  rd
}
read_metrics <- function(rd) {
  cc <- c(cell_id = "character", sample_id = "character", doublet_score = "character",
          doublet_call = "character", pre_cluster = "character")
  f <- gzfile(file.path(rd, "S1", "qc_metrics.tsv.gz"))
  hdr <- strsplit(readLines(f, n = 1), "\t")[[1]]
  read.delim(gzfile(file.path(rd, "S1", "qc_metrics.tsv.gz")), check.names = FALSE, na.strings = "",
             colClasses = cc[names(cc) %in% hdr])
}

test_that("S1a engine scfast writes the S1b columns into qc_metrics and no h5ad; S1b inputs are complete", {
  skip_if_no_scfast()
  rd <- new_run("scfast")
  quiet(run_s0(scp_read_config(rd), REPO))
  quiet(run_s1a(scp_read_config(rd), REPO))
  expect_length(list.files(rd, pattern = "\\.h5ad$", recursive = TRUE), 0)
  met <- read_metrics(rd)
  expect_identical(colnames(met)[1:8], c("cell_id", "sample_id", "n_genes", "n_umis", "pct_mito", "pct_ribo",
                                         "pct_hb", "log10_genes_per_umi"))
  expect_identical(tail(colnames(met), 3), c("doublet_score", "doublet_call", "pre_cluster"))
  s0 <- scp_load_object(file.path(rd, "S0"))
  expect_equal(nrow(met), ncol(s0))
  expect_identical(met$cell_id, colnames(s0))
  expect_true(all(met$doublet_call %in% c("TRUE", "FALSE")))
  expect_false(any(is.na(met$pre_cluster) | met$pre_cluster == ""))
  sc <- suppressWarnings(as.numeric(met$doublet_score))
  expect_false(anyNA(sc))
  expect_true(all(sc >= 0 & sc <= 1))
  # recall on the planted doublets of both samples
  tr <- truth()
  tr <- tr[match(met$cell_id, tr$cell_id), ]
  dbl <- tr$injected == "doublet"
  expect_equal(sum(dbl), 20)
  expect_gte(mean(met$doublet_call[dbl] == "TRUE"), 0.5)
  expect_lte(mean(met$doublet_call[tr$injected == "none"] == "TRUE"), 0.10)
})

test_that("S1a engine scfast with doublets.method none writes NA/FALSE doublet columns but still pre-clusters", {
  skip_if_no_scfast()
  rd <- new_run("scfast", method = "none", tag = "none")
  quiet(run_s0(scp_read_config(rd), REPO))
  quiet(run_s1a(scp_read_config(rd), REPO))
  met <- read_metrics(rd)
  expect_true(all(met$doublet_score == "NA"))
  expect_true(all(met$doublet_call == "FALSE"))
  expect_gte(length(unique(met$pre_cluster)), 2)
})

test_that("S1a engine scfast replaces a stale metrics_counts.h5ad from an earlier seurat run", {
  skip_if_no_scfast()
  rd <- new_run("scfast", tag = "stale")
  quiet(run_s0(scp_read_config(rd), REPO))
  dir.create(file.path(rd, "S1"))
  writeLines("stale", file.path(rd, "S1", "metrics_counts.h5ad"))
  quiet(run_s1a(scp_read_config(rd), REPO))
  expect_false(file.exists(file.path(rd, "S1", "metrics_counts.h5ad")))
})

test_that("S1a engine seurat is unchanged: metrics_counts.h5ad written, no S1b columns", {
  rd <- new_run("seurat", tag = "seurat")
  quiet(run_s0(scp_read_config(rd), REPO))
  quiet(run_s1a(scp_read_config(rd), REPO))
  expect_true(file.exists(file.path(rd, "S1", "metrics_counts.h5ad")))
  met <- read_metrics(rd)
  expect_false(any(c("doublet_score", "doublet_call", "pre_cluster") %in% colnames(met)))
})

test_that("S1a engine scfast with scdblfinder is an error", {
  skip_if_no_scfast()
  rd <- new_run("scfast", method = "scdblfinder", tag = "scdbl")
  quiet(run_s0(scp_read_config(rd), REPO))
  expect_error(quiet(run_s1a(scp_read_config(rd), REPO)), "scdblfinder")
})
