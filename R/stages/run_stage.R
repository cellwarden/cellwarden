#!/usr/bin/env Rscript
# Entry point for the R stages of cellwarden (docs/stages-spec.md section 1):
#
#   Rscript R/stages/run_stage.R --stage S0|S1a|S2 --run-dir DIR
#
# Reads <DIR>/config/config.resolved.yaml, applies thread caps from
# CELLWARDEN_THREADS, runs the stage, and exits 0 on success or 1 with an
# "ERROR" line on failure. Logs go to stdout/stderr; the runner redirects
# them to logs/<STAGE>.log.

.scp_script_dir <- function() {
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f)) return(dirname(normalizePath(f[[1]])))
  if (!is.null(sys.frames()[[1]]$ofile)) return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  normalizePath(file.path(getwd(), "R", "stages"), mustWork = FALSE)
}

scp_source_stages <- function(repo) {
  source(file.path(repo, "R", "exchange", "exchange.R"))
  for (f in c("common.R", "s0_ingest.R", "doublets_scfast.R", "s1a_metrics.R", "s2_core.R", "s3_evidence.R", "s4_dge.R", "s5_export.R"))
    source(file.path(repo, "R", "stages", f))
  invisible(repo)
}

.SCP_STAGES <- c("S0", "S1a", "S2", "S3", "S4", "S5")

scp_run_stage <- function(stage, run_dir, repo) {
  if (!(stage %in% .SCP_STAGES)) scp_stop("unknown stage '", stage, "' (R stages: ", paste(.SCP_STAGES, collapse = ", "), ")")
  if (!dir.exists(run_dir)) scp_stop("run directory not found: ", run_dir)
  suppressPackageStartupMessages({
    library(Seurat)
    library(SeuratObject)
  })
  options(Seurat.object.assay.version = "v5")
  cfg <- scp_read_config(run_dir)
  n <- scp_setup_threads(cfg)
  scp_log(sprintf("stage %s, run %s, run dir %s, threads %d, R %s, Seurat %s", stage, cfg$run_id, cfg$run_dir, n,
                  paste(R.version$major, R.version$minor, sep = "."), scp_pkg_version("Seurat")))
  switch(stage,
         S0 = run_s0(cfg, repo),
         S1a = run_s1a(cfg, repo),
         S2 = run_s2(cfg, repo),
         S3 = run_s3(cfg, repo),
         S4 = run_s4(cfg, repo),
         S5 = run_s5(cfg, repo))
  scp_log(sprintf("stage %s finished; peak memory %s GB", stage, format(scp_peak_mem_gb())))
  invisible(TRUE)
}

.scp_main <- function() {
  repo <- normalizePath(file.path(.scp_script_dir(), "..", ".."))
  args <- commandArgs(trailingOnly = TRUE)
  opt <- if (requireNamespace("optparse", quietly = TRUE)) {
    p <- optparse::OptionParser(usage = "Rscript R/stages/run_stage.R --stage S0|S1a|S2 --run-dir DIR",
                                option_list = list(
                                  optparse::make_option("--stage", type = "character"),
                                  optparse::make_option("--run-dir", type = "character", dest = "run_dir")))
    optparse::parse_args(p, args = args)
  } else {
    get <- function(k) { i <- match(k, args); if (is.na(i) || i == length(args)) NULL else args[[i + 1]] }
    list(stage = get("--stage"), run_dir = get("--run-dir"))
  }
  if (is.null(opt$stage) || is.null(opt$run_dir)) {
    cat("usage: Rscript R/stages/run_stage.R --stage S0|S1a|S2 --run-dir DIR\n", file = stderr())
    quit(status = 2, save = "no")
  }
  ok <- tryCatch({
    scp_source_stages(repo)
    withCallingHandlers(
      scp_run_stage(opt$stage, normalizePath(opt$run_dir, mustWork = FALSE), repo),
      warning = function(w) {
        msg <- conditionMessage(w)
        if (exists("scp_warn_log")) scp_warn_log("R warning: ", msg) else message("WARN: ", msg)
        invokeRestart("muffleWarning")
      })
    TRUE
  }, error = function(e) {
    msg <- sprintf("%s [ERROR] stage %s failed: %s", format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
                   opt$stage, conditionMessage(e))
    cat(msg, "\n", sep = "")
    cat(msg, "\n", sep = "", file = stderr())
    FALSE
  })
  quit(status = if (isTRUE(ok)) 0L else 1L, save = "no")
}

if (sys.nframe() == 0L || identical(Sys.getenv("CELLWARDEN_RUN_STAGE_MAIN"), "1")) .scp_main()
