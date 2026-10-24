# cellwarden stage S1a: per-cell QC metrics (docs/stages-spec.md section 3.2).
#
# Inputs:  S0/object.rds (or .qs2), qc_policy.yaml (protected populations)
# Outputs: S1/metrics_counts.h5ad  exchange file, full mode (counts + log-normalised X, all cells);
#                                  not written when the engine resolves to scfast
#          S1/qc_metrics.tsv.gz    partial metrics (no S1b columns), gzip TSV, columns:
#            cell_id, sample_id, n_genes, n_umis, pct_mito, pct_ribo, pct_hb,
#            log10_genes_per_umi, protected_<name> ... (policy order)
#
# Engine scfast (scp_resolve_engine, docs/scfast-integration.md): S1a also runs the
# doublet detector and pre-clustering (R/stages/doublets_scfast.R) and appends
# doublet_score, doublet_call, pre_cluster to qc_metrics.tsv.gz; no .h5ad is written
# and S1b passes through. doublets.method none/scdblfinder are not scfast-aware:
# `none` writes NA/FALSE scores (pre_cluster still computed); scdblfinder is an error.
#
# Percentages are 0-100. log10_genes_per_umi = log10(n_genes) / log10(n_umis),
# NA when n_umis <= 1. protected_<name> is the fraction (0-1) of the cell's
# UMIs in that population's marker genes (matched on the original symbols);
# markers absent from the data contribute 0 and are logged as warnings.

# Column-sum of counts over a gene subset without densifying.
.s1a_colsum <- function(m, rows) {
  if (!length(rows)) return(numeric(ncol(m)))
  Matrix::colSums(m[rows, , drop = FALSE])
}

s1a_load_policy <- function(path) {
  pol <- yaml::read_yaml(path)
  if (!is.list(pol)) scp_stop("qc policy ", path, " is not a mapping")
  pp <- pol$protected_populations %||% list()
  if (!is.list(pp)) scp_stop("qc policy ", path, ": protected_populations must be a list")
  names_seen <- character()
  for (i in seq_along(pp)) {
    p <- pp[[i]]
    if (!is.list(p) || !is.character(p$name) || length(p$name) != 1 || !nzchar(p$name))
      scp_stop("qc policy ", path, ": protected_populations[", i, "] needs a name")
    if (!grepl("^[A-Za-z0-9_.-]+$", p$name))
      scp_stop("qc policy ", path, ": protected population name '", p$name, "' must match [A-Za-z0-9_.-]+")
    if (p$name %in% names_seen) scp_stop("qc policy ", path, ": duplicated protected population '", p$name, "'")
    names_seen <- c(names_seen, p$name)
    if (is.null(p$markers) || !length(p$markers))
      scp_stop("qc policy ", path, ": protected population '", p$name, "' has no markers")
    pp[[i]]$markers <- as.character(unlist(p$markers))
  }
  pol$protected_populations <- pp
  pol
}

#' Compute the S1a metric table for a Seurat object (genes x cells counts).
#' Returns list(metrics = data.frame, warnings = character()).
s1a_compute_metrics <- function(obj, policy, assay = "RNA") {
  m <- scp_counts(obj, assay)
  sym <- scp_symbols(obj, assay)
  fl <- scp_gene_flags(sym)
  n_umis <- Matrix::colSums(m)
  n_genes <- Matrix::colSums(m > 0)
  pct <- function(rows) {
    s <- .s1a_colsum(m, which(rows))
    ifelse(n_umis > 0, 100 * s / n_umis, NA_real_)
  }
  warn <- character()
  if (!any(fl$mito)) warn <- c(warn, "no mitochondrial genes (^MT-) found; pct_mito is 0")
  if (!any(fl$ribo)) warn <- c(warn, "no ribosomal genes (^RP[SL]) found; pct_ribo is 0")
  md <- obj[[]]
  sid <- if ("sample_id" %in% colnames(md)) as.character(md$sample_id) else as.character(md$orig.ident)
  df <- data.frame(
    cell_id = colnames(m),
    sample_id = sid,
    n_genes = as.integer(n_genes),
    n_umis = as.numeric(n_umis),
    pct_mito = pct(fl$mito),
    pct_ribo = pct(fl$ribo),
    pct_hb = pct(fl$hb),
    log10_genes_per_umi = ifelse(n_umis > 1 & n_genes > 0, log10(pmax(n_genes, 1)) / log10(pmax(n_umis, 2)), NA_real_),
    stringsAsFactors = FALSE, check.names = FALSE
  )
  if (all(df$n_umis == round(df$n_umis)) && max(df$n_umis, 0) < .Machine$integer.max) df$n_umis <- as.integer(df$n_umis)
  for (p in policy$protected_populations) {
    mk <- unique(p$markers)
    rows <- which(sym %in% mk)
    found <- unique(sym[rows])
    absent <- setdiff(mk, found)
    if (length(absent)) {
      warn <- c(warn, sprintf("protected population '%s': marker(s) not in the data, counted as 0: %s",
                              p$name, paste(absent, collapse = ", ")))
    }
    s <- .s1a_colsum(m, rows)
    df[[paste0("protected_", p$name)]] <- ifelse(n_umis > 0, s / n_umis, 0)
  }
  list(metrics = df, warnings = warn)
}

# Device the S1 stages would run on ("cpu" or "gpu"), for engine resolution: the device
# plan of S2 (same hook and fallbacks) when it can be computed, else the config value.
.s1a_plan_device <- function(cfg, repo, n_cells, n_genes) {
  if (identical(cfg$device, "cpu")) return("cpu")
  if (identical(cfg$device, "gpu") && !exists("s2_choose_plan", mode = "function")) return("gpu")
  if (!exists("s2_choose_plan", mode = "function")) return("cpu")
  tryCatch(s2_choose_plan(cfg, repo, n_cells, min(cfg$n_hvg, n_genes))$device,
           error = function(e) {
             scp_warn_log("device plan unavailable for engine resolution (", conditionMessage(e), "); using device: ", cfg$device)
             if (identical(cfg$device, "gpu")) "gpu" else "cpu"
           })
}

run_s1a <- function(cfg, repo) {
  in_dir <- file.path(cfg$run_dir, "S0")
  out_dir <- scp_stage_dir(cfg, "S1")
  policy_file <- scp_config_file(cfg, "qc_policy", "qc_policy.yaml")
  policy <- s1a_load_policy(policy_file)
  obj <- scp_timed("load S0 object", scp_load_object(in_dir))$value
  scp_log(sprintf("S1a: %d cells x %d genes; %d protected population(s) from %s",
                  ncol(obj), nrow(obj), length(policy$protected_populations), policy_file))

  res <- scp_timed("compute metrics", s1a_compute_metrics(obj, policy))$value
  for (w in res$warnings) scp_warn_log(w)
  plan_dev <- .s1a_plan_device(cfg, repo, ncol(obj), nrow(obj))
  engine <- scp_resolve_engine(cfg, plan_dev)
  scfast_dev <- if (identical(engine, "scfast") && identical(plan_dev, "gpu")) "cuda" else "cpu"
  scp_log("S1a: engine ", engine, " (config engine: ", cfg$engine %||% "auto", "; scfast device ", scfast_dev, ")")
  h5 <- file.path(out_dir, "metrics_counts.h5ad")

  if (identical(engine, "scfast")) {
    method <- cfg$doublets$method %||% "scrublet"
    if (identical(method, "scdblfinder"))
      scp_stop("doublets.method scdblfinder is not implemented for engine scfast; use scrublet or none")
    unlink(h5)  # no exchange file on this path; drop a stale one from an earlier engine
    dbl <- scp_timed("scfast doublets + pre-clustering",
                     scp_doublets_scfast(scp_counts(obj), res$metrics$sample_id, cfg$seed, device = scfast_dev))$value
    for (w in dbl$notes) scp_log(w)
    tab <- scp_dbl_format(dbl$table)
    if (identical(method, "none")) {
      tab$doublet_score <- "NA"
      tab$doublet_call <- "FALSE"
    }
    stopifnot(identical(tab$cell_id, res$metrics$cell_id))
    res$metrics$doublet_score <- tab$doublet_score
    res$metrics$doublet_call <- tab$doublet_call
    res$metrics$pre_cluster <- tab$pre_cluster
    scp_timed("write qc_metrics.tsv.gz", scp_write_tsv_gz(res$metrics, file.path(out_dir, "qc_metrics.tsv.gz")))
  } else {
    scp_timed("write qc_metrics.tsv.gz", scp_write_tsv_gz(res$metrics, file.path(out_dir, "qc_metrics.tsv.gz")))
    obj <- scp_timed("NormalizeData", Seurat::NormalizeData(obj, normalization.method = "LogNormalize",
                                                             scale.factor = 1e4, verbose = FALSE))$value
    scp_timed("write metrics_counts.h5ad",
              write_exchange(obj, h5, mode = "full",
                             run_id = cfg$run_id, stage = "S1a", config_hash = cfg$config_hash,
                             reductions = character(), graphs = FALSE))
    probs <- validate_exchange(h5)
    if (length(probs)) scp_stop("metrics_counts.h5ad failed validation:\n  ", paste(probs, collapse = "\n  "))
  }
  m <- res$metrics
  scp_log(sprintf("S1a done: %d cells; median genes %.0f, median UMIs %.0f, median pct_mito %.2f; peak memory %s GB",
                  nrow(m), stats::median(m$n_genes), stats::median(m$n_umis),
                  stats::median(m$pct_mito, na.rm = TRUE), format(scp_peak_mem_gb())))
  invisible(res)
}
