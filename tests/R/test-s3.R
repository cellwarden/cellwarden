# Unit tests for stage S3 (R half): R/stages/s3_evidence.R, contract in
# docs/agents-spec.md section 4.
#
#   Rscript -e 'testthat::test_file("tests/R/test-s3.R")'   # from repo root
#
# A run directory is built once from tests/fixtures/stage_inputs (donor_A as a
# 10x mtx folder, donor_B as .h5ad; planted types T, B, Mono, NK, Platelet in
# truth.tsv) by running S0 and S2 through the stage functions (every cell
# kept; CPU plan forced with CELLWARDEN_PLAN_JSON), then run_s3 on it.

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
# Source only what S3 needs (run_stage.R sources every stage file, some of which other build steps add).
source(file.path(REPO, "R", "exchange", "exchange.R"))
for (f in c("common.R", "s0_ingest.R", "s2_core.R", "s3_evidence.R")) source(file.path(REPO, "R", "stages", f))
options(Seurat.object.assay.version = "v5")
INPUTS <- file.path(REPO, "tests", "fixtures", "stage_inputs")
TRUTH <- read.delim(file.path(INPUTS, "truth.tsv"), colClasses = "character", quote = "")

quiet <- function(expr) { out <- NULL; utils::capture.output(out <- force(expr)); out }

new_run <- function(tag, ...) {
  rd <- file.path(tempdir(), sprintf("cellwarden-s3-%s-%s", tag, basename(tempfile(""))))
  dir.create(file.path(rd, "config"), recursive = TRUE)
  samples <- data.frame(sample_id = c("donor_A", "donor_B"),
                        path = file.path(INPUTS, c("donor_A", "donor_B.h5ad")),
                        format = c("", "h5ad"), donor = c("D1", "D2"), condition = c("control", "control"),
                        batch = c("B1", "B2"), stringsAsFactors = FALSE)
  sf <- file.path(rd, "config", "samples.csv")
  write.csv(samples, sf, row.names = FALSE, quote = TRUE)
  pf <- file.path(rd, "config", "qc_policy.yaml")
  yaml::write_yaml(list(hard_limits = list(min_genes = 200, max_mito_pct = 25, min_umis = 400)), pf)
  cfg <- list(project = "test", run_dir = rd, run_id = paste0("test-", tag), config_hash = strrep("b", 64),
              seed = 42L, samples = sf, qc_policy = pf, object_format = "rds",
              resources = list(threads = 1L, max_memory_gb = 4, gpu_mem_cap_gb = 20, nice = 10L),
              doublets = list(method = "none"), qc = list(review_action = "keep"),
              normalization = "lognorm", n_hvg = 500L, device = "cpu",
              integration = list(method = "none", batch_key = "sample_id"),
              pcs = "auto", n_neighbors = 20L, clustering = list(resolution = 0.5),
              annotation = list(celltypist_model = "Immune_All_Low.pkl", celltypist = "auto", min_confidence = 0.6))
  over <- list(...)
  for (k in names(over)) cfg[[k]] <- over[[k]]
  yaml::write_yaml(cfg, file.path(rd, "config", "config.resolved.yaml"))
  rd
}

build_s2 <- function(rd) {
  cfg <- scp_read_config(rd)
  quiet(run_s0(cfg, REPO))
  obj <- scp_load_object(file.path(rd, "S0"))
  d <- data.frame(cell_id = colnames(obj), sample_id = as.character(obj$sample_id), decision = "keep",
                  reasons = "", policy_hash = "p", decided_by = "rules")
  dir.create(file.path(rd, "S1"), showWarnings = FALSE)
  scp_write_tsv_gz(d, file.path(rd, "S1", "qc_decisions.tsv.gz"))
  withr::local_envvar(CELLWARDEN_PLAN_JSON = '{"device":"cpu","plan":"cpu","reason":"test: cpu"}')
  quiet(run_s2(cfg, REPO))
  rd
}

RD <- build_s2(new_run("main"))
S3_SUMMARY <- quiet(run_s3(scp_read_config(RD), REPO))
S3 <- file.path(RD, "S3")
MARKERS <- scp_read_tsv(file.path(S3, "markers.tsv.gz"))
for (cn in c("avg_log2fc", "pct_in", "pct_out", "p_adj", "rank")) MARKERS[[cn]] <- as.numeric(MARKERS[[cn]])
SUMMARY <- jsonlite::fromJSON(file.path(S3, "cluster_summary.json"), simplifyVector = FALSE)
SIGS <- scp_read_tsv(file.path(S3, "signatures.tsv.gz"))
OBJ <- scp_load_object(file.path(RD, "S2"))
CLUSTERS <- levels(OBJ$seurat_clusters)

# Planted markers (tests/fixtures/make_fixture.py MARKERS and make_stage_inputs.py PLATELET_MARKERS).
PLANTED <- list(
  T = c("CD3D", "CD3E", "IL7R", "CD2", "LTB", "TRAC", "CD3G", "LEF1"),
  B = c("MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "HLA-DQA1", "TCL1A", "VPREB3"),
  Mono = c("CD14", "LYZ", "S100A8", "S100A9", "FCN1", "CST3", "LST1", "TYROBP"),
  NK = c("NKG7", "GNLY", "GZMB", "PRF1", "KLRD1", "GZMA", "CTSW", "FCGR3A"),
  Platelet = c("PPBP", "PF4", "GNG11"))

# cluster -> planted type for clusters whose purity (non-injected cells) is >= 0.9
pure_clusters <- function() {
  tr <- TRUTH[TRUTH$injected %in% c("none", "platelet"), ]
  cl <- as.character(OBJ$seurat_clusters)[match(tr$cell_id, colnames(OBJ))]
  ok <- !is.na(cl)
  tab <- table(cl[ok], tr$cell_type[ok])
  out <- list()
  for (k in rownames(tab)) {
    frac <- tab[k, ] / sum(tab[k, ])
    if (max(frac) >= 0.9 && sum(tab[k, ]) >= 10) out[[k]] <- names(frac)[which.max(frac)]
  }
  out
}
PURE <- pure_clusters()

# ================================================================ outputs
test_that("S3 writes every output file", {
  for (f in c("markers.tsv.gz", "cluster_summary.json", "signatures.tsv.gz", "cell_clusters.tsv.gz", "s3_r_summary.json"))
    expect_true(file.exists(file.path(S3, f)), info = f)
  expect_false(file.exists(file.path(S3, "exchange", "counts.h5ad")))   # celltypist is not importable here
  expect_identical(S3_SUMMARY$celltypist_input$written, FALSE)
})

test_that("markers.tsv.gz has the spec columns, top 50 per cluster, ranks 1..n by avg_log2fc", {
  expect_identical(colnames(MARKERS), c("cluster", "gene", "avg_log2fc", "pct_in", "pct_out", "p_adj", "rank"))
  expect_true(all(MARKERS$cluster %in% CLUSTERS))
  expect_true(all(CLUSTERS %in% MARKERS$cluster))
  for (k in CLUSTERS) {
    m <- MARKERS[MARKERS$cluster == k, ]
    expect_lte(nrow(m), 50)
    expect_equal(m$rank, seq_len(nrow(m)))
    expect_true(all(diff(m$avg_log2fc) <= 1e-9))
    expect_true(all(m$avg_log2fc > 0))
    expect_true(all(m$p_adj >= 0 & m$p_adj <= 1))
    expect_true(all(m$pct_in >= 0 & m$pct_in <= 100))
    expect_false(anyDuplicated(m$gene) > 0)
  }
  expect_match(S3_SUMMARY$markers$method, "FindAllMarkers|presto")
})

test_that("planted marker genes rank at the top of their clusters", {
  expect_gte(length(PURE), 4)
  expect_true(all(c("T", "B", "Mono", "NK") %in% unlist(PURE)))
  for (k in names(PURE)) {
    top <- head(MARKERS$gene[MARKERS$cluster == k], 10)
    hits <- intersect(PLANTED[[PURE[[k]]]], top)
    expect_true(length(hits) >= min(3, length(PLANTED[[PURE[[k]]]])), info = sprintf("cluster %s (%s): %s", k, PURE[[k]], paste(top, collapse = ",")))
    # the planted markers of the other types must not lead this cluster
    others <- setdiff(unlist(PLANTED[setdiff(names(PLANTED), PURE[[k]])]), PLANTED[[PURE[[k]]]])
    expect_false(top[[1]] %in% others, info = sprintf("cluster %s top marker %s", k, top[[1]]))
  }
})

test_that("cluster_summary.json has the spec keys and consistent counts", {
  expect_setequal(names(SUMMARY), CLUSTERS)
  tot <- 0
  for (k in CLUSTERS) {
    s <- SUMMARY[[k]]
    expect_setequal(names(s), c("n_cells", "pct_cells", "per_sample", "median_genes", "median_umis", "median_pct_mito", "top_markers"))
    expect_identical(s$n_cells, sum(OBJ$seurat_clusters == k))
    expect_equal(sum(unlist(s$per_sample)), s$n_cells)
    expect_setequal(names(s$per_sample), c("donor_A", "donor_B")[c("donor_A", "donor_B") %in% names(s$per_sample)])
    expect_true(s$median_genes > 0 && s$median_umis > 0)
    expect_true(s$median_pct_mito >= 0 && s$median_pct_mito <= 100)
    expect_lte(length(s$top_markers), 20)
    expect_identical(unlist(s$top_markers), head(MARKERS$gene[MARKERS$cluster == k], 20))
    tot <- tot + s$pct_cells
  }
  expect_equal(tot, 100, tolerance = 1e-3)
  # median genes/UMIs agree with the object's metadata
  k <- CLUSTERS[[1]]
  expect_equal(SUMMARY[[k]]$median_genes, median(OBJ$nFeature_RNA[OBJ$seurat_clusters == k]))
  expect_equal(SUMMARY[[k]]$median_umis, median(OBJ$nCount_RNA[OBJ$seurat_clusters == k]))
})

test_that("signatures.tsv.gz covers every built-in set per cluster and lineage sets peak in the right cluster", {
  expect_identical(colnames(SIGS), c("cluster", "signature", "category", "cell_type", "mean_score", "median_score",
                                     "pct_positive", "n_genes_set", "n_genes_found"))
  sets <- yaml::read_yaml(file.path(REPO, "R", "stages", "data", "signatures.yaml"))$sets
  need <- c("interferon", "inflammation", "proliferation", "hypoxia", "platelet", "erythrocyte", "T", "CD4_T", "CD8_T",
            "NK", "B", "plasma", "mono_CD14", "mono_CD16", "cDC", "pDC", "neutrophil", "mast", "epithelial",
            "endothelial", "fibroblast", "hepatocyte", "neuron", "glia")
  expect_true(all(need %in% names(sets)))
  for (nm in names(sets)) expect_gte(length(sets[[nm]]$genes), 5)
  expect_identical(nrow(SIGS), length(sets) * length(CLUSTERS))
  expect_setequal(unique(SIGS$signature), names(sets))
  SIGS$mean_score <- as.numeric(SIGS$mean_score)
  # a set with no gene in the data is NA, never an error
  none <- SIGS[as.integer(SIGS$n_genes_found) == 0, ]
  expect_true(all(is.na(none$mean_score)))
  expect_true(all(is.na(SIGS$mean_score[SIGS$signature == "hepatocyte"])))
  # the planted lineages score highest in their own cluster
  map <- c(T = "T", B = "B", Mono = "mono_CD14", NK = "NK", Platelet = "platelet")
  for (k in names(PURE)) {
    s <- SIGS[SIGS$signature == map[[PURE[[k]]]], ]
    expect_identical(s$cluster[which.max(s$mean_score)], k, info = PURE[[k]])
    expect_gt(max(s$mean_score), 0)
  }
})

test_that("s3_r_summary.json carries the metadata hint (values only), set coverage and timings", {
  s <- jsonlite::fromJSON(file.path(S3, "s3_r_summary.json"), simplifyVector = FALSE)
  expect_identical(s$stage, "S3")
  expect_identical(s$n_clusters, length(CLUSTERS))
  expect_identical(unlist(s$clusters), CLUSTERS)
  expect_setequal(names(s$sample_metadata_hint), c("donor", "condition", "batch"))
  expect_setequal(unlist(s$sample_metadata_hint$donor), c("D1", "D2"))
  expect_identical(unlist(s$sample_metadata_hint$condition), "control")   # still a JSON array of one
  expect_identical(s$signatures$sets$T$cell_type, "T cell")
  expect_gte(s$signatures$sets$T$n_genes_found, 5)
  expect_true(all(c("load", "markers", "signatures", "celltypist_input") %in% names(s$timings_s)))
  cc <- scp_read_tsv(file.path(S3, "cell_clusters.tsv.gz"))
  expect_identical(colnames(cc), c("cell_id", "cluster", "sample_id"))
  expect_identical(cc$cell_id, colnames(OBJ))
  expect_identical(cc$cluster, as.character(OBJ$seurat_clusters))
})

test_that("annotation config is validated and a custom signature file is honoured", {
  expect_error(s3_annotation_cfg(list(annotation = list(celltypist = "maybe"))), "annotation.celltypist")
  a <- s3_annotation_cfg(list())
  expect_identical(a$celltypist_model, "Immune_All_Low.pkl")
  expect_identical(a$celltypist, "auto")
  expect_equal(a$min_confidence, 0.6)
  custom <- tempfile(fileext = ".yaml")
  yaml::write_yaml(list(sets = list(Tsmall = list(category = "lineage", cell_type = "T cell",
                                                  genes = list("CD3D", "CD3E", "CD3G")))), custom)
  rd2 <- new_run("sig", signatures = custom)
  file.copy(file.path(RD, "S2"), rd2, recursive = TRUE)
  quiet(run_s3(scp_read_config(rd2), REPO))
  s <- scp_read_tsv(file.path(rd2, "S3", "signatures.tsv.gz"))
  expect_identical(unique(s$signature), "Tsmall")
  expect_identical(nrow(s), length(CLUSTERS))
  expect_error(s3_read_signatures(list(signatures = "/nonexistent/sig.yaml"), REPO), "signature file not found")
})

test_that("top-marker selection is lenient when a cluster has few significant genes", {
  df <- data.frame(cluster = c("a", "a", "a", "b"), gene = c("g1", "g2", "g3", "g4"),
                   avg_log2fc = c(2, 1, 3, 1), pct_in = 50, pct_out = 5, p_adj = c(0.5, 0.9, 0.01, 1e-5),
                   stringsAsFactors = FALSE)
  top <- s3_top_markers(df, c("a", "b"))
  expect_identical(attr(top, "lenient"), "a")
  a <- top[top$cluster == "a", ]
  expect_identical(a$gene, c("g3", "g1", "g2"))      # all kept (lenient), ranked by logFC
  expect_identical(a$rank, 1:3)
  expect_identical(top$gene[top$cluster == "b"], "g4")
  # with >= 5 significant genes only those are kept
  df2 <- data.frame(cluster = "c", gene = paste0("h", 1:8), avg_log2fc = 8:1, pct_in = 50, pct_out = 5,
                    p_adj = c(rep(1e-4, 5), 0.2, 0.3, 0.4), stringsAsFactors = FALSE)
  t2 <- s3_top_markers(df2, "c")
  expect_identical(t2$gene, paste0("h", 1:5))
  expect_length(attr(t2, "lenient"), 0)
})
