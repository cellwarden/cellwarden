# cellwarden R stages: shared helpers (config, logging, object I/O, threads, JSON,
# gene patterns). Sourced by R/stages/run_stage.R together with
# R/exchange/exchange.R and the stage files. See docs/stages-spec.md.

suppressPackageStartupMessages({
  library(Matrix)
  library(jsonlite)
  library(yaml)
})

# --------------------------------------------------------------------------
# Errors, logging
# --------------------------------------------------------------------------

scp_stop <- function(...) stop(paste0(...), call. = FALSE)

.scp_utc <- function() format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")

# Timestamped log line to stdout (the runner redirects stdout to logs/<STAGE>.log).
scp_log <- function(..., level = "INFO") {
  msg <- paste0(...)
  cat(sprintf("%s [%s] %s\n", .scp_utc(), level, msg))
  flush(stdout())
  invisible(msg)
}
scp_warn_log <- function(...) scp_log(..., level = "WARN")

# Run expr, logging how long it took; returns list(value, seconds).
scp_timed <- function(label, expr) {
  t0 <- proc.time()[["elapsed"]]
  scp_log("start: ", label)
  value <- force(expr)
  secs <- round(proc.time()[["elapsed"]] - t0, 3)
  scp_log("done:  ", label, " (", secs, " s)")
  list(value = value, seconds = secs)
}

# Peak resident memory of this process in GB (Linux VmHWM), NA elsewhere.
scp_peak_mem_gb <- function() {
  f <- "/proc/self/status"
  if (!file.exists(f)) return(NA_real_)
  l <- tryCatch(readLines(f, warn = FALSE), error = function(e) character())
  h <- grep("^VmHWM:", l, value = TRUE)
  if (!length(h)) return(NA_real_)
  kb <- suppressWarnings(as.numeric(gsub("[^0-9]", "", h[[1]])))
  if (is.na(kb)) NA_real_ else round(kb / 1024^2, 4)
}

scp_pkg_version <- function(p) tryCatch(as.character(utils::packageVersion(p)), error = function(e) "not-installed")

scp_has_pkg <- function(p) requireNamespace(p, quietly = TRUE)

# --------------------------------------------------------------------------
# Engine resolution (docs/scfast-integration.md, "Config")
# --------------------------------------------------------------------------

#' `scfastR::scfast_info()` as a list, or NULL when scfastR is not installed (or the
#' call fails). Tests replace this binding to mock a CUDA build.
scp_scfast_info <- function() {
  if (!scp_has_pkg("scfastR")) return(NULL)
  tryCatch(as.list(scfastR::scfast_info()), error = function(e) NULL)
}

#' CUDA state of the installed scfastR: `list(built, available)` (logicals).
#' `scfast_info()` fields `cuda_built` / `cuda_available` are read; a missing
#' field is FALSE (`cuda` is accepted as a fallback for `cuda_built`), so a
#' scfastR without the CUDA feature never reports a usable device.
scp_scfast_cuda <- function(info = scp_scfast_info()) {
  flag <- function(x) is.logical(x) && length(x) == 1 && isTRUE(x)
  if (is.null(info)) return(list(built = FALSE, available = FALSE))
  built <- flag(info$cuda_built) || (is.null(info$cuda_built) && flag(info$cuda))
  available <- built && flag(info$cuda_available)
  list(built = built, available = available)
}

#' Resolve `engine` / `engine_gpu` to "scfast" or "seurat".
#'
#' @param cfg  resolved config; a missing `engine` is "auto", a missing
#'   `engine_gpu` is "auto".
#' @param device  device the stage will run on, "cpu" or "gpu" (the device plan,
#'   not the config value "auto"). Any other value is treated as "cpu".
#'
#' CPU plan: `seurat` is always "seurat"; `scfast` needs scfastR (error with an
#' install hint if missing); `auto` is "scfast" when scfastR is installed.
#'
#' GPU plan (docs/scfast-integration.md, "Pipeline wiring"): `engine_gpu: rapids`
#' is "seurat" (rapids through the exchange); `engine_gpu: scfast` is "scfast"
#' (in process, device cuda) and errors when scfastR was not built with CUDA or
#' no device is visible; `engine_gpu: auto` is "scfast" when scfastR reports
#' `cuda_available`, else "seurat". `engine: seurat` is always "seurat";
#' `engine: scfast` on a GPU plan that cannot run scfast on CUDA is an error.
scp_resolve_engine <- function(cfg, device = "cpu") {
  eng <- cfg[["engine"]]  # [[ ]]: `$` would partially match engine_gpu
  if (is.null(eng)) eng <- "auto"
  if (!(is.character(eng) && length(eng) == 1 && !is.na(eng) && eng %in% c("auto", "scfast", "seurat")))
    scp_stop("engine must be one of auto/scfast/seurat (got '", paste(eng, collapse = ","), "')")
  eng_gpu <- cfg[["engine_gpu"]]
  if (is.null(eng_gpu)) eng_gpu <- "auto"
  if (!(is.character(eng_gpu) && length(eng_gpu) == 1 && !is.na(eng_gpu) && eng_gpu %in% c("auto", "scfast", "rapids")))
    scp_stop("engine_gpu must be one of auto/scfast/rapids (got '", paste(eng_gpu, collapse = ","), "')")
  gpu <- is.character(device) && length(device) == 1 && identical(device, "gpu")
  if (eng == "seurat") return("seurat")
  have <- scp_has_pkg("scfastR")
  install_hint <- "install it with `R CMD INSTALL rust/scfast-r` (needs cargo >= 1.80) or set engine: seurat"
  if (!gpu) {
    if (eng == "scfast") {
      if (!have) scp_stop("engine: scfast needs the R package scfastR, which is not installed; ", install_hint)
      return("scfast")
    }
    return(if (have) "scfast" else "seurat")
  }
  # GPU plan
  if (eng_gpu == "rapids") {
    if (eng == "scfast") scp_stop("engine: scfast with engine_gpu: rapids has no GPU backend; use device: cpu, ",
                                  "engine_gpu: auto/scfast, or engine: auto/seurat")
    return("seurat")
  }
  cuda <- scp_scfast_cuda()
  if (cuda$available) return("scfast")
  why <- if (!have) "scfastR is not installed"
         else if (!cuda$built) "scfastR was built without the cuda feature"
         else "no CUDA device is visible to scfastR"
  if (eng_gpu == "scfast")
    scp_stop("engine_gpu: scfast needs scfastR built with CUDA and a visible device (", why, "); ",
             "use engine_gpu: auto/rapids or device: cpu")
  if (eng == "scfast")
    scp_stop("engine: scfast has no GPU backend here (", why, "); use device: cpu (or engine: auto/seurat)")
  "seurat"
}

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

.SCP_DEFAULTS <- list(
  project = "cellwarden", seed = 42L, object_format = "rds",
  resources = list(threads = 1L, max_memory_gb = 16, gpu_mem_cap_gb = 20, nice = 10L),
  doublets = list(method = "scrublet"),
  qc = list(review_action = "keep"),
  normalization = "lognorm", n_hvg = 2000L, device = "auto",
  integration = list(method = "none", batch_key = "sample_id"),
  pcs = "auto", n_neighbors = 20L,
  clustering = list(resolution = 0.5),
  run_id = "none", config_hash = "none"
)

# Fill missing keys from defaults (nested one level, as in the schema).
.scp_fill <- function(x, d) {
  for (k in names(d)) {
    if (is.null(x[[k]])) x[[k]] <- d[[k]]
    else if (is.list(d[[k]]) && is.list(x[[k]])) x[[k]] <- .scp_fill(x[[k]], d[[k]])
  }
  x
}

#' Read <run-dir>/config/config.resolved.yaml, fill defaults for anything the
#' runner left out, and validate the keys S0-S2 use. Reports every problem at once.
scp_read_config <- function(run_dir) {
  f <- file.path(run_dir, "config", "config.resolved.yaml")
  if (!file.exists(f)) scp_stop("resolved config not found: ", f)
  cfg <- yaml::read_yaml(f)
  if (!is.list(cfg)) scp_stop("resolved config is not a mapping: ", f)
  cfg <- .scp_fill(cfg, .SCP_DEFAULTS)
  cfg$run_dir <- normalizePath(run_dir, mustWork = TRUE)
  p <- character()
  chk <- function(ok, msg) if (!isTRUE(ok)) p <<- c(p, msg)
  one_of <- function(v, allowed, key)
    chk(is.character(v) && length(v) == 1 && v %in% allowed,
        sprintf("%s must be one of %s (got '%s')", key, paste(allowed, collapse = "/"), paste(v, collapse = ",")))
  one_of(cfg$object_format, c("rds", "qs2"), "object_format")
  one_of(cfg$normalization, c("lognorm", "sct"), "normalization")
  one_of(cfg$device, c("auto", "cpu", "gpu"), "device")
  one_of(cfg$integration$method, c("none", "harmony"), "integration.method")
  one_of(cfg$qc$review_action, c("keep", "exclude"), "qc.review_action")
  num1 <- function(v) is.numeric(v) && length(v) == 1 && !is.na(v)
  chk(num1(cfg$n_hvg) && cfg$n_hvg >= 50, "n_hvg must be a number >= 50")
  chk(num1(cfg$n_neighbors) && cfg$n_neighbors >= 2, "n_neighbors must be a number >= 2")
  chk(num1(cfg$clustering$resolution) && cfg$clustering$resolution > 0, "clustering.resolution must be > 0")
  chk(num1(cfg$seed), "seed must be a number")
  chk(identical(cfg$pcs, "auto") || (num1(cfg$pcs) && cfg$pcs >= 2),
      "pcs must be 'auto' or an integer >= 2")
  chk(is.character(cfg$integration$batch_key) && length(cfg$integration$batch_key) == 1,
      "integration.batch_key must be a string")
  if (length(p)) scp_stop("invalid config ", f, ":\n  ", paste(p, collapse = "\n  "))
  cfg$seed <- as.integer(cfg$seed)
  cfg$n_hvg <- as.integer(cfg$n_hvg)
  cfg$n_neighbors <- as.integer(cfg$n_neighbors)
  cfg$run_id <- as.character(cfg$run_id)
  cfg$config_hash <- as.character(cfg$config_hash)
  cfg
}

# Path of a config file: the resolved key if it exists, else the copy in the run dir.
scp_config_file <- function(cfg, key, default_name) {
  v <- cfg[[key]]
  if (is.character(v) && length(v) == 1 && nzchar(v) && file.exists(v)) return(normalizePath(v))
  cp <- file.path(cfg$run_dir, "config", default_name)
  if (file.exists(cp)) return(normalizePath(cp))
  scp_stop("config key '", key, "' does not point to an existing file (",
           if (is.null(v)) "unset" else v, ") and ", cp, " does not exist")
}

scp_stage_dir <- function(cfg, stage) {
  d <- file.path(cfg$run_dir, stage)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

# --------------------------------------------------------------------------
# Threads
# --------------------------------------------------------------------------

#' Thread caps from CELLWARDEN_THREADS (falls back to resources.threads):
#' Seurat::setThreads() when that Seurat version has it, data.table and BLAS
#' (RhpcBLASctl) when installed; future plan sequential.
scp_setup_threads <- function(cfg) {
  n <- suppressWarnings(as.integer(Sys.getenv("CELLWARDEN_THREADS", "")))
  if (is.na(n) || n < 1) n <- as.integer(cfg$resources$threads %||% 1L)
  if (is.na(n) || n < 1) n <- 1L
  if (scp_has_pkg("Seurat") && exists("setThreads", envir = asNamespace("Seurat"), inherits = FALSE)) {
    tryCatch(get("setThreads", envir = asNamespace("Seurat"))(n),
             error = function(e) scp_warn_log("Seurat::setThreads(", n, ") failed: ", conditionMessage(e)))
  }
  if (scp_has_pkg("data.table")) data.table::setDTthreads(n)
  if (scp_has_pkg("RhpcBLASctl")) {
    try(RhpcBLASctl::blas_set_num_threads(n), silent = TRUE)
    try(RhpcBLASctl::omp_set_num_threads(n), silent = TRUE)
  }
  if (scp_has_pkg("future")) future::plan("sequential")
  mem <- suppressWarnings(as.numeric(Sys.getenv("CELLWARDEN_MAX_MEMORY_GB", "")))
  if (is.na(mem)) mem <- as.numeric(cfg$resources$max_memory_gb %||% 16)
  options(future.globals.maxSize = mem * 1024^3)
  invisible(n)
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# --------------------------------------------------------------------------
# Object save / load (rds or qs2; readers accept both)
# --------------------------------------------------------------------------

.scp_atomic <- function(path, writer) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp-", Sys.getpid())
  ok <- FALSE
  on.exit(if (!ok) unlink(tmp), add = TRUE)
  writer(tmp)
  if (!file.rename(tmp, path)) {
    if (!file.copy(tmp, path, overwrite = TRUE)) scp_stop("could not move ", tmp, " to ", path)
    unlink(tmp)
  }
  ok <- TRUE
  invisible(path)
}

#' Save obj as <stage_dir>/object.<rds|qs2>; removes a stale file of the other
#' format. qs2 is used only when configured and installed (else rds, with a log warning).
scp_save_object <- function(obj, stage_dir, format = "rds") {
  if (identical(format, "qs2") && !scp_has_pkg("qs2")) {
    scp_warn_log("object_format qs2 requested but the qs2 package is not installed; writing .rds")
    format <- "rds"
  }
  path <- file.path(stage_dir, paste0("object.", format))
  other <- file.path(stage_dir, paste0("object.", if (format == "rds") "qs2" else "rds"))
  if (format == "qs2") {
    nt <- suppressWarnings(as.integer(Sys.getenv("CELLWARDEN_THREADS", "1")))
    if (is.na(nt) || nt < 1) nt <- 1L
    .scp_atomic(path, function(p) qs2::qs_save(obj, p, nthreads = nt))
  } else {
    .scp_atomic(path, function(p) saveRDS(obj, p))
  }
  if (file.exists(other)) unlink(other)
  invisible(path)
}

#' Read a Seurat object from a stage dir (object.qs2 or object.rds) or a file path.
scp_load_object <- function(path) {
  if (dir.exists(path)) {
    cand <- file.path(path, c("object.qs2", "object.rds"))
    cand <- cand[file.exists(cand)]
    if (!length(cand)) scp_stop("no object.rds or object.qs2 in ", path)
    path <- cand[which.max(file.mtime(cand))]
  }
  scp_read_object_file(path)
}

scp_read_object_file <- function(path) {
  if (!file.exists(path)) scp_stop("object file not found: ", path)
  if (grepl("\\.qs2?$", path, ignore.case = TRUE)) {
    if (grepl("\\.qs2$", path, ignore.case = TRUE) && scp_has_pkg("qs2")) return(qs2::qs_read(path))
    if (grepl("\\.qs$", path, ignore.case = TRUE) && scp_has_pkg("qs")) return(qs::qread(path))
    scp_stop("cannot read ", path, ": package ", if (grepl("qs2$", path)) "qs2" else "qs", " is not installed")
  }
  readRDS(path)
}

# --------------------------------------------------------------------------
# JSON / TSV writing
# --------------------------------------------------------------------------

scp_write_json <- function(x, path) {
  .scp_atomic(path, function(p)
    jsonlite::write_json(x, p, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null", na = "null"))
}

#' Gzip TSV, header row, no quoting, NA written as empty field.
scp_write_tsv_gz <- function(df, path) {
  .scp_atomic(path, function(p) {
    if (scp_has_pkg("data.table")) {
      data.table::fwrite(df, p, sep = "\t", quote = FALSE, na = "", compress = "gzip")
    } else {
      con <- gzfile(p, "wb")
      on.exit(close(con))
      utils::write.table(df, con, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
    }
  })
}

scp_read_tsv <- function(path) {
  if (scp_has_pkg("data.table") && (!grepl("\\.gz$", path) || scp_has_pkg("R.utils"))) {
    return(as.data.frame(data.table::fread(path, sep = "\t", colClasses = list(character = 1:2),
                                           na.strings = "", data.table = FALSE)))
  }
  con <- if (grepl("\\.gz$", path)) gzfile(path) else file(path)
  utils::read.delim(con, colClasses = "character", na.strings = "", quote = "", check.names = FALSE)
}

# --------------------------------------------------------------------------
# Gene patterns (spec 3.2)
# --------------------------------------------------------------------------

SCP_MITO_RE <- "^MT-"
SCP_RIBO_RE <- "^RP[SL]"
SCP_HB_RE <- "^HB[ABDEGMQZ][0-9]*$"
# Non-haemoglobin genes that share the HB prefix. The regex already excludes
# them (they do not match HB[ABDEGMQZ]<digits>$); the list documents intent
# and guards against regex edits.
SCP_HB_EXCLUDE <- c("HBP1", "HBEGF", "HBS1L", "HBAP1", "HBBP1", "HBZP1", "HBEGF-AS1")

scp_gene_flags <- function(symbols) {
  list(mito = grepl(SCP_MITO_RE, symbols),
       ribo = grepl(SCP_RIBO_RE, symbols),
       hb = grepl(SCP_HB_RE, symbols) & !(symbols %in% SCP_HB_EXCLUDE))
}

# Original gene symbols for an assay (feature metadata gene_symbol_original,
# falling back to the feature names).
scp_symbols <- function(obj, assay = "RNA") {
  fm <- tryCatch(obj[[assay]][[]], error = function(e) NULL)
  g <- rownames(obj[[assay]])
  if (!is.null(fm) && "gene_symbol_original" %in% colnames(fm)) {
    s <- as.character(fm$gene_symbol_original)
    bad <- is.na(s) | !nzchar(s)
    s[bad] <- g[bad]
    return(s)
  }
  g
}

# counts layer as dgCMatrix (converts BPCells / other Matrix classes in memory).
scp_counts <- function(obj, assay = "RNA") {
  m <- SeuratObject::LayerData(obj, assay = assay, layer = "counts")
  scp_as_dgc(m)
}

scp_as_dgc <- function(m) {
  if (inherits(m, "dgCMatrix")) return(m)
  if (inherits(m, "IterableMatrix")) {
    if (!scp_has_pkg("BPCells")) scp_stop("counts are a BPCells matrix but BPCells is not installed")
    return(as(m, "dgCMatrix"))
  }
  if (inherits(m, "DelayedArray") || inherits(m, "DelayedMatrix")) m <- as(m, "dgCMatrix")
  if (!is(m, "Matrix")) m <- Matrix::Matrix(as.matrix(m), sparse = TRUE)
  as(as(as(m, "CsparseMatrix"), "generalMatrix"), "dMatrix")
}
