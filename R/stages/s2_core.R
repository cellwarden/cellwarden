# cellwarden stage S2: core processing (docs/stages-spec.md section 3.5).
#
# Inputs:  S0/object.rds (or .qs2), S1/qc_decisions.tsv.gz
# Outputs: S2/object.rds (filtered, normalised, HVGs, reductions, graphs, clusters)
#          S2/exchange/slim.h5ad + S2/exchange/results.h5ad (GPU path only)
#          S2/summary.json
#
# Device / memory plan: `python -m cellwarden.resources choose-plan` (spec section 5)
# prints {"device", "plan", "reason"}. CELLWARDEN_PLAN_JSON (a JSON string, or a
# path to a JSON file) replaces that call; it is a test hook.
#
# Leiden: Seurat >= 5.1 runs FindClusters(algorithm = 4) through leidenbase.
# Seurat 5.0.x routes algorithm 4 through Python leidenalg via reticulate, which
# is fragile (here it fails on a NumPy ABI mismatch), so for those versions,
# and whenever FindClusters fails, S2 runs Leiden itself with
# leidenbase::leiden_find_partition on the SNN graph using the same settings
# as FindClusters (RBConfiguration partition, n.iter = 10, labels ordered by
# cluster size, 0-based).

.S2_PCA_MAX <- 50L
.S2_PCA_MIN <- 10L
.S2_PCS_AUTO_SAMPLE <- 20000L

# --------------------------------------------------------------------------
# Python helpers
# --------------------------------------------------------------------------

s2_python <- function() {
  p <- Sys.getenv("CELLWARDEN_PYTHON", "")
  if (nzchar(p)) p else "python3"
}

# Environment for Python subprocesses: <repo>/python on PYTHONPATH.
.s2_py_env <- function(repo, extra = character()) {
  pp <- Sys.getenv("PYTHONPATH", "")
  pyp <- file.path(repo, "python")
  c(paste0("PYTHONPATH=", if (nzchar(pp)) paste(pyp, pp, sep = .Platform$path.sep) else pyp), extra)
}

# Run python with args; returns list(status, stdout, stderr). Never throws.
s2_run_python <- function(repo, args, extra_env = character()) {
  so <- tempfile(); se <- tempfile()
  on.exit(unlink(c(so, se)), add = TRUE)
  st <- tryCatch(system2(s2_python(), shQuote(args), stdout = so, stderr = se,
                         env = .s2_py_env(repo, extra_env)),
                 error = function(e) { writeLines(conditionMessage(e), se); 127L })
  list(status = as.integer(st),
       stdout = if (file.exists(so)) readLines(so, warn = FALSE) else character(),
       stderr = if (file.exists(se)) readLines(se, warn = FALSE) else character())
}

# Parse the last JSON object printed on stdout.
.s2_parse_json_out <- function(lines) {
  txt <- paste(lines, collapse = "\n")
  pos <- gregexpr("\\{", txt)[[1]]
  if (pos[[1]] < 0) return(NULL)
  # try progressively later '{' until one parses to the end
  for (p in pos) {
    v <- tryCatch(jsonlite::fromJSON(substring(txt, p), simplifyVector = TRUE), error = function(e) NULL)
    if (is.list(v)) return(v)
  }
  NULL
}

s2_python_has <- function(repo, module) {
  r <- s2_run_python(repo, c("-c", paste0("import ", module)))
  r$status == 0
}

# --------------------------------------------------------------------------
# Device / memory plan
# --------------------------------------------------------------------------

.s2_check_plan <- function(p, src) {
  if (!is.list(p)) scp_stop("device plan from ", src, " is not a JSON object")
  dev <- p$device; plan <- p$plan
  if (!(is.character(dev) && length(dev) == 1 && dev %in% c("cpu", "gpu")))
    scp_stop("device plan from ", src, ": 'device' must be cpu or gpu (got '", paste(dev, collapse = ","), "')")
  if (!(is.character(plan) && length(plan) == 1 && plan %in% c("in_memory", "managed_memory", "cpu")))
    scp_stop("device plan from ", src, ": 'plan' must be in_memory, managed_memory or cpu (got '", paste(plan, collapse = ","), "')")
  if (dev == "cpu" && plan != "cpu") scp_stop("device plan from ", src, ": device cpu needs plan cpu")
  if (dev == "gpu" && plan == "cpu") scp_stop("device plan from ", src, ": device gpu cannot have plan cpu")
  list(device = dev, plan = plan, reason = as.character(p$reason %||% ""), source = src)
}

s2_choose_plan <- function(cfg, repo, n_cells, n_hvg) {
  hook <- Sys.getenv("CELLWARDEN_PLAN_JSON", "")
  if (nzchar(hook)) {
    txt <- if (file.exists(hook)) paste(readLines(hook, warn = FALSE), collapse = "\n") else hook
    p <- tryCatch(jsonlite::fromJSON(txt, simplifyVector = TRUE),
                  error = function(e) scp_stop("CELLWARDEN_PLAN_JSON is not valid JSON: ", conditionMessage(e)))
    return(.s2_check_plan(p, "CELLWARDEN_PLAN_JSON"))
  }
  device <- cfg$device
  if (identical(cfg$normalization, "sct")) {
    if (device == "gpu") scp_stop("normalization: sct runs on the CPU only; set device: cpu or auto")
    return(list(device = "cpu", plan = "cpu", reason = "normalization sct is CPU only", source = "S2"))
  }
  cap <- suppressWarnings(as.numeric(Sys.getenv("CELLWARDEN_GPU_MEM_CAP_GB", "")))
  if (is.na(cap)) cap <- as.numeric(cfg$resources$gpu_mem_cap_gb %||% 20)
  args <- c("-m", "cellwarden.resources", "choose-plan", "--n-cells", as.character(n_cells),
            "--n-hvg", as.character(n_hvg), "--device", device, "--gpu-mem-cap-gb", format(cap))
  r <- s2_run_python(repo, args)
  p <- if (r$status == 0) .s2_parse_json_out(r$stdout) else NULL
  if (is.null(p)) {
    err <- paste(tail(c(r$stdout, r$stderr), 10), collapse = "\n  ")
    if (device %in% c("cpu", "auto")) {
      scp_warn_log("cellwarden.resources choose-plan failed (exit ", r$status, "); device is ", device,
                   ", continuing on the CPU path:\n  ", err)
      return(list(device = "cpu", plan = "cpu", source = "S2",
                  reason = paste0("device: ", device, " in config; choose-plan unavailable, so the CPU path is used")))
    }
    scp_stop("device plan: `", s2_python(), " ", paste(args, collapse = " "), "` failed (exit ", r$status, "):\n  ", err)
  }
  .s2_check_plan(p, "cellwarden.resources choose-plan")
}

# --------------------------------------------------------------------------
# QC decisions
# --------------------------------------------------------------------------

s2_read_decisions <- function(path) {
  if (!file.exists(path)) scp_stop("QC decisions not found: ", path, " (run S1c first)")
  d <- scp_read_tsv(path)
  need <- c("cell_id", "decision")
  miss <- setdiff(need, colnames(d))
  if (length(miss)) scp_stop(path, " is missing column(s): ", paste(miss, collapse = ", "))
  bad <- setdiff(unique(d$decision), c("keep", "exclude", "review"))
  if (length(bad)) scp_stop(path, ": unknown decision value(s): ", paste(bad, collapse = ", "))
  if (anyDuplicated(d$cell_id)) scp_stop(path, ": duplicated cell_id values")
  d
}

# --------------------------------------------------------------------------
# PCs: elbow rule
# --------------------------------------------------------------------------

#' Elbow rule on PCA standard deviations: pct_i = stdev_i / sum(stdev) * 100;
#' co1 = first PC where the cumulative pct exceeds 90% and pct_i < 5%;
#' co2 = last PC where the drop to the next PC exceeds 0.1 percentage points, + 1;
#' n = min(co1, co2), clamped to [lo, hi] and to the number of PCs available.
s2_elbow_pcs <- function(stdev, lo = .S2_PCA_MIN, hi = .S2_PCA_MAX) {
  stdev <- as.numeric(stdev)
  n_av <- length(stdev)
  if (n_av == 0) scp_stop("no PCA standard deviations for the elbow rule")
  pct <- stdev / sum(stdev) * 100
  cum <- cumsum(pct)
  co1 <- which(cum > 90 & pct < 5)[1]
  d <- pct[-n_av] - pct[-1]
  big <- which(d > 0.1)
  co2 <- if (length(big)) max(big) + 1L else 1L
  raw <- min(c(co1, co2), na.rm = TRUE)
  n <- as.integer(max(lo, min(hi, raw)))
  n <- min(n, n_av)
  list(n = n, co1 = if (is.na(co1)) NA_integer_ else as.integer(co1), co2 = as.integer(co2),
       raw = as.integer(raw), available = n_av)
}

# PCA stdevs of the scaled HVG matrix on up to .S2_PCS_AUTO_SAMPLE cells
# (irlba), used to pick PCs before the GPU path, where PCA runs on the GPU.
.s2_sample_pca_stdev <- function(obj, hvg, npcs, seed, assay = "RNA") {
  cells <- colnames(obj)
  if (length(cells) > .S2_PCS_AUTO_SAMPLE) {
    set.seed(seed)
    cells <- sort(sample(cells, .S2_PCS_AUTO_SAMPLE))
  }
  x <- SeuratObject::LayerData(obj, assay = assay, layer = "data")[hvg, cells, drop = FALSE]
  x <- Matrix::t(scp_as_dgc(x))                     # cells x genes
  mu <- Matrix::colMeans(x)
  sdv <- sqrt(pmax(Matrix::colMeans(x^2) - mu^2, 0) * nrow(x) / max(1, nrow(x) - 1))
  sdv[sdv == 0] <- 1
  z <- sweep(sweep(as.matrix(x), 2, mu, "-"), 2, sdv, "/")
  z[z > 10] <- 10
  k <- min(npcs, min(dim(z)) - 1L)
  set.seed(seed)
  pc <- irlba::irlba(z, nv = k, center = FALSE, scale = FALSE)
  pc$d / sqrt(max(1, nrow(z) - 1))
}

# --------------------------------------------------------------------------
# Leiden
# --------------------------------------------------------------------------

.s2_seurat_has_leidenbase_clusters <- function() {
  utils::packageVersion("Seurat") >= "5.1.0"
}

#' Leiden on a Seurat SNN graph via leidenbase, with FindClusters' settings
#' (RBConfigurationVertexPartition, 10 iterations); labels "0".."k-1" by
#' decreasing cluster size.
s2_leiden_leidenbase <- function(snn, resolution, seed, n_iter = 10L) {
  if (!scp_has_pkg("leidenbase") || !scp_has_pkg("igraph"))
    scp_stop("Leiden clustering needs the leidenbase and igraph R packages")
  g <- igraph::graph_from_adjacency_matrix(as(snn, "CsparseMatrix"), mode = "undirected",
                                           weighted = TRUE, diag = FALSE)
  res <- leidenbase::leiden_find_partition(g, partition_type = "RBConfigurationVertexPartition",
                                           edge_weights = igraph::E(g)$weight,
                                           resolution_parameter = resolution,
                                           seed = as.integer(seed), num_iter = as.integer(n_iter))
  memb <- res$membership
  sizes <- table(memb)
  ord <- names(sizes)[order(-as.integer(sizes), as.integer(names(sizes)))]
  lab <- match(as.character(memb), ord) - 1L
  f <- factor(lab, levels = seq_along(ord) - 1L)
  names(f) <- rownames(snn)
  f
}

# --------------------------------------------------------------------------
# Harmony (R)
# --------------------------------------------------------------------------

.s2_run_harmony_r <- function(obj, batch_key, dims, seed, hp = list()) {
  set.seed(seed)
  # harmony >= 1.0: RunHarmony.Seurat(object, group.by.vars, reduction.use, dims.use, reduction.save, ...)
  args <- list(obj, group.by.vars = batch_key, reduction.use = "pca", dims.use = dims,
               reduction.save = "harmony", verbose = FALSE)
  for (k in c("theta", "lambda", "sigma", "nclust")) if (!is.null(hp[[k]])) args[[k]] <- hp[[k]]
  if (!is.null(hp$max_iter)) args$max_iter <- hp$max_iter
  do.call(harmony::RunHarmony, args)
}

# scfast's Harmony (scfastR::RunHarmonyFast, Rust): same algorithm and defaults as harmony 2.x,
# 6-40x faster; used wherever scfastR provides it unless integration.engine is r.
.s2_has_fast_harmony <- function() {
  scp_has_pkg("scfastR") && exists("RunHarmonyFast", envir = asNamespace("scfastR"), inherits = FALSE)
}

.s2_harmony_impl <- function(cfg) {
  eng <- cfg$integration$engine %||% "auto"
  if (eng == "scfast" && !.s2_has_fast_harmony())
    scp_stop("integration.engine is scfast but the installed scfastR has no RunHarmonyFast; reinstall scfastR (rust/scfast-r) or use engine auto")
  if (eng == "r" && !scp_has_pkg("harmony"))
    scp_stop("integration.engine is r but the R 'harmony' package is not installed")
  if (eng %in% c("auto", "scfast") && .s2_has_fast_harmony()) "scfast" else if (scp_has_pkg("harmony")) "r" else "none"
}

.s2_run_harmony <- function(obj, batch_key, dims, seed, cfg, device = "cpu") {
  hp <- cfg$integration$harmony %||% list()
  if (.s2_harmony_impl(cfg) == "scfast") {
    args <- list(obj, group.by.vars = batch_key, reduction = "pca", dims.use = dims,
                 reduction.save = "harmony", seed = seed, device = "cpu")
    for (k in c("theta", "lambda", "sigma", "nclust")) if (!is.null(hp[[k]])) args[[k]] <- hp[[k]]
    if (!is.null(hp$max_iter)) args$max.iter <- hp$max_iter
    obj <- do.call(scfastR::RunHarmonyFast, args)
    attr(obj, "s2_harmony_impl") <- paste0("scfast Harmony (Rust, CPU; scfastR ", scp_pkg_version("scfastR"), ")")
    return(obj)
  }
  obj <- .s2_run_harmony_r(obj, batch_key, dims, seed, hp)
  attr(obj, "s2_harmony_impl") <- paste0("harmony (R ", scp_pkg_version("harmony"), ")")
  obj
}

# --------------------------------------------------------------------------
# Engine (docs/scfast-integration.md): auto | scfast | seurat
# --------------------------------------------------------------------------

#' Resolve cfg$engine / cfg$engine_gpu for S2 given the device plan
#' (scp_resolve_engine in common.R): `auto` picks scfast when scfastR is installed
#' on a CPU plan, or built with CUDA with a visible device on a GPU plan
#' (`engine_gpu` forces either). An explicit `scfast` additionally needs lognorm
#' normalisation. Returns "scfast" or "seurat".
s2_resolve_engine <- function(cfg, plan, assay = "RNA") {
  eng <- cfg[["engine"]] %||% "auto"
  if (!(is.character(eng) && length(eng) == 1 && eng %in% c("auto", "scfast", "seurat")))
    scp_stop("engine must be auto, scfast or seurat (got '", paste(eng, collapse = ","), "')")
  if (eng == "seurat") return("seurat")
  if (assay != "RNA") {
    if (eng == "scfast") scp_stop("engine: scfast supports lognorm normalisation only (normalization: sct needs engine: seurat)")
    return("seurat")
  }
  scp_resolve_engine(cfg, device = if (identical(plan$device, "gpu")) "gpu" else "cpu")
}

#' Device scfastR runs on for an S2 path: "cuda" on gpu_scfast, else "cpu".
.s2_scfast_device <- function(path) if (identical(path, "gpu_scfast")) "cuda" else "cpu"

# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------

run_s2 <- function(cfg, repo) {
  out_dir <- scp_stage_dir(cfg, "S2")
  seed <- cfg$seed
  set.seed(seed)
  timings <- list()
  tm <- function(label, expr) {
    r <- scp_timed(label, expr)
    timings[[label]] <<- r$seconds
    r$value
  }

  obj <- tm("load", scp_load_object(file.path(cfg$run_dir, "S0")))
  dec <- s2_read_decisions(file.path(cfg$run_dir, "S1", "qc_decisions.tsv.gz"))
  cells_in <- ncol(obj)

  # 1. subset
  unknown <- setdiff(dec$cell_id, colnames(obj))
  if (length(unknown)) scp_stop(length(unknown), " cell(s) in qc_decisions are not in the S0 object, e.g. ",
                                paste(head(unknown, 5), collapse = ", "))
  undecided <- setdiff(colnames(obj), dec$cell_id)
  if (length(undecided)) scp_stop(length(undecided), " S0 cell(s) have no QC decision, e.g. ",
                                  paste(head(undecided, 5), collapse = ", "), " (rerun S1)")
  review_action <- cfg$qc$review_action
  keep_set <- dec$cell_id[dec$decision == "keep" | (dec$decision == "review" & review_action == "keep")]
  keep_cells <- colnames(obj)[colnames(obj) %in% keep_set]          # object order
  if (length(keep_cells) < 3) scp_stop("only ", length(keep_cells), " cell(s) pass QC; nothing to process")
  dec_counts <- as.list(table(factor(dec$decision, levels = c("keep", "review", "exclude"))))
  obj <- tm("subset", subset(obj, cells = keep_cells))
  scp_log(sprintf("S2: %d of %d cells kept (keep %d, review %d [%s], exclude %d)", ncol(obj), cells_in,
                  dec_counts$keep, dec_counts$review, review_action, dec_counts$exclude))
  # drop genes? No: the gene set stays that of S0 so counts checksums stay comparable.
  if ("decision" %in% colnames(obj[[]])) obj$decision <- NULL
  dmap <- dec$decision[match(colnames(obj), dec$cell_id)]
  obj$qc_decision <- factor(dmap, levels = c("keep", "review"))

  n_hvg <- min(cfg$n_hvg, nrow(obj))
  assay <- "RNA"

  # 4. device / memory plan (before normalisation: on the scfast engine the fused
  # PreprocessFast() replaces NormalizeData + FindVariableFeatures, docs/scfast-integration.md Phase 9)
  plan <- s2_choose_plan(cfg, repo, ncol(obj), n_hvg)
  scp_log(sprintf("device plan: device=%s plan=%s (%s) [%s]", plan$device, plan$plan, plan$reason, plan$source))

  # 2-3. normalise + variable genes
  pre_impl <- "Seurat::NormalizeData + FindVariableFeatures(vst)"
  if (!identical(cfg$normalization, "sct")) {
    # Same engine decision as below; the harmony fallback (exchange_cpu) keeps the Seurat
    # preprocessing because that path runs with engine seurat.
    eng0 <- tryCatch(s2_resolve_engine(cfg, plan, assay), error = function(e) "seurat")
    harmony_fallback <- identical(cfg$integration$method, "harmony") && .s2_harmony_impl(cfg) == "none"
    if (eng0 == "scfast" && !harmony_fallback &&
        exists("PreprocessFast", envir = asNamespace("scfastR"), inherits = FALSE)) {
      pre_impl <- "scfastR::PreprocessFast (fused normalise + vst + scaling statistics)"
    }
  }
  if (identical(cfg$normalization, "sct")) {
    obj <- tm("sctransform", Seurat::SCTransform(obj, vst.flavor = "v2", variable.features.n = n_hvg,
                                                 seed.use = seed, verbose = FALSE))
    assay <- "SCT"
    pre_impl <- "Seurat::SCTransform"
  } else if (startsWith(pre_impl, "scfastR")) {
    obj <- tm("preprocess", scfastR::PreprocessFast(obj, n_hvg = n_hvg, assay = assay, scale.factor = 1e4,
                                                    loess.span = 0.3, verbose = FALSE))
  } else {
    obj <- tm("normalize", Seurat::NormalizeData(obj, normalization.method = "LogNormalize",
                                                 scale.factor = 1e4, verbose = FALSE))
    obj <- tm("hvg", Seurat::FindVariableFeatures(obj, selection.method = "vst", nfeatures = n_hvg, verbose = FALSE))
  }
  hvg <- SeuratObject::VariableFeatures(obj, assay = assay)
  n_hvg_used <- length(hvg)

  integ <- cfg$integration$method
  batch_key <- cfg$integration$batch_key
  if (integ == "harmony") {
    if (!(batch_key %in% colnames(obj[[]]))) scp_stop("integration.batch_key '", batch_key, "' is not a cell metadata column")
    nb <- length(unique(obj[[batch_key, drop = TRUE]]))
    if (nb < 2) scp_warn_log("integration: harmony requested but '", batch_key, "' has only ", nb, " level(s)")
  }
  path <- if (plan$device == "gpu") "gpu" else "cpu"
  path_reason <- plan$reason
  if (path == "cpu" && integ == "harmony" && .s2_harmony_impl(cfg) == "none") {
    if (identical(cfg$device, "auto") && s2_python_has(repo, "harmonypy")) {
      path <- "exchange_cpu"
      path_reason <- paste0(plan$reason, "; R harmony not installed, so the exchange path runs Harmony with harmonypy on the CPU")
      if (identical(tryCatch(s2_resolve_engine(cfg, plan, assay), error = function(e) "seurat"), "scfast"))
        scp_warn_log("engine scfast: the R 'harmony' package is not installed; falling back to the exchange_cpu path (harmonypy)")
      else scp_log(path_reason)
    } else {
      scp_stop("integration.method is harmony but the R 'harmony' package is not installed",
               if (identical(cfg$device, "auto")) " and Python harmonypy is not available" else "",
               "; install harmony (R) or harmonypy (Python, with device: auto), or set integration.method: none")
    }
  }
  engine <- "seurat"
  if (path %in% c("cpu", "gpu")) engine <- s2_resolve_engine(cfg, plan, assay)
  if (path == "cpu" && engine == "scfast") {
    path <- "scfast"
    path_reason <- paste0(plan$reason, "; engine scfast (in-process, no exchange files)")
  } else if (path == "gpu" && engine == "scfast") {
    # docs/scfast-integration.md "Pipeline wiring": scfastR built with CUDA and a device visible
    # (or engine_gpu: scfast): same in-process code as cpu_scfast with device = "cuda"; UMAP on CPU.
    if (integ == "harmony" && .s2_harmony_impl(cfg) == "none")
      scp_stop("integration.method is harmony but neither scfastR's RunHarmonyFast nor the R 'harmony' package is installed; ",
               "install harmony (R), set engine_gpu: rapids (harmonypy through the exchange), or set integration.method: none")
    path <- "gpu_scfast"
    path_reason <- paste0(plan$reason, "; engine scfast on CUDA (in-process, no exchange files; UMAP on CPU)")
  }
  if (path %in% c("exchange_cpu", "gpu") && assay != "RNA") scp_stop("the exchange (GPU) path supports lognorm normalisation only")
  scfast_dev <- .s2_scfast_device(path)

  # 7. PCs
  pcs_setting <- cfg$pcs
  pcs_info <- list(setting = pcs_setting)
  npcs_compute <- min(.S2_PCA_MAX, n_hvg_used - 1L, ncol(obj) - 1L)
  if (!identical(pcs_setting, "auto")) {
    pcs_req <- as.integer(pcs_setting)
    if (pcs_req > npcs_compute) scp_warn_log("pcs = ", pcs_req, " exceeds the ", npcs_compute, " PCs available; using ", npcs_compute)
    npcs_compute_fixed <- min(pcs_req, ncol(obj) - 1L, n_hvg_used - 1L)
  }

  leiden_impl <- NA_character_
  umap_impl <- NA_character_
  scfast_ver <- NULL
  if (path %in% c("scfast", "gpu_scfast")) {
    scfast_ver <- as.character(scfastR::scfast_info()$version %||% scfastR::scfast_info()$package)
    if (integ == "harmony") integration_impl <- NA_character_
  }
  integration_impl <- if (integ == "harmony") NA_character_ else "none"
  gpu_summary <- NULL
  exchange_files <- NULL

  if (path %in% c("cpu", "scfast", "gpu_scfast")) {
    use_fast <- path %in% c("scfast", "gpu_scfast")
    # 5. CPU path (Seurat, or scfastR in process)
    # RunPCAFast scales implicitly from the data layer, so ScaleData is skipped on the scfast path.
    if (assay == "RNA" && !use_fast) obj <- tm("scale", Seurat::ScaleData(obj, features = hvg, verbose = FALSE))
    npcs_run <- if (identical(pcs_setting, "auto")) npcs_compute else max(npcs_compute_fixed, 2L)
    obj <- if (use_fast) {
      tm("pca", scfastR::RunPCAFast(obj, assay = assay, features = hvg, npcs = npcs_run, seed.use = seed,
                                    device = scfast_dev, verbose = FALSE))
    } else tm("pca", Seurat::RunPCA(obj, assay = assay, features = hvg, npcs = npcs_run, seed.use = seed, verbose = FALSE))
    if (identical(pcs_setting, "auto")) {
      el <- s2_elbow_pcs(SeuratObject::Stdev(obj, reduction = "pca"))
      pcs_used <- el$n
      pcs_info <- c(pcs_info, list(method = "elbow", elbow = el, basis = "full PCA"))
    } else pcs_used <- npcs_run
    dims <- seq_len(pcs_used)
    rep_red <- "pca"
    if (integ == "harmony") {
      obj <- tm("harmony", .s2_run_harmony(obj, batch_key, dims, seed, cfg))
      rep_red <- "harmony"
      integration_impl <- attr(obj, "s2_harmony_impl")
      attr(obj, "s2_harmony_impl") <- NULL
      dims <- seq_len(min(pcs_used, ncol(SeuratObject::Embeddings(obj, "harmony"))))
    }
    if (use_fast) {
      obj <- tm("neighbors", scfastR::FindNeighborsFast(obj, reduction = rep_red, dims = dims, k.param = cfg$n_neighbors,
                                                        graph.name = paste0(assay, c("_nn", "_snn")),
                                                        random.seed = seed, device = scfast_dev, verbose = FALSE))
    } else {
      obj <- tm("neighbors", Seurat::FindNeighbors(obj, reduction = rep_red, dims = dims, k.param = cfg$n_neighbors,
                                                   graph.name = paste0(assay, c("_nn", "_snn")), verbose = FALSE))
    }
    snn_name <- paste0(assay, "_snn")
    cl <- NULL
    if (use_fast) {
      t0 <- proc.time()[["elapsed"]]
      obj <- scfastR::FindClustersFast(obj, graph.name = snn_name, resolution = cfg$clustering$resolution,
                                       random.seed = seed, verbose = FALSE)
      timings[["leiden"]] <- round(proc.time()[["elapsed"]] - t0, 3)
      cl <- obj$seurat_clusters
      leiden_impl <- paste0("scfastR::FindClustersFast (Leiden), scfastR ", scfast_ver)
    } else if (.s2_seurat_has_leidenbase_clusters()) {
      cl <- tryCatch({
        o2 <- Seurat::FindClusters(obj, graph.name = snn_name, algorithm = 4, resolution = cfg$clustering$resolution,
                                   random.seed = seed, verbose = FALSE)
        leiden_impl <- paste0("Seurat::FindClusters(algorithm = 4), Seurat ", scp_pkg_version("Seurat"))
        o2$seurat_clusters
      }, error = function(e) { scp_warn_log("FindClusters(algorithm = 4) failed: ", conditionMessage(e),
                                            "; using leidenbase directly"); NULL })
    }
    if (is.null(cl)) {
      t0 <- proc.time()[["elapsed"]]
      cl <- s2_leiden_leidenbase(obj[[snn_name]], cfg$clustering$resolution, seed)
      leiden_impl <- paste0("leidenbase::leiden_find_partition ", scp_pkg_version("leidenbase"),
                            " on ", snn_name, " (RBConfigurationVertexPartition, 10 iterations)")
      timings[["leiden"]] <- round(proc.time()[["elapsed"]] - t0, 3)
    }
    cl <- factor(as.character(cl)[match(colnames(obj), names(cl) %||% colnames(obj))], levels = levels(cl))
    names(cl) <- colnames(obj)
    obj$seurat_clusters <- cl
    obj[[paste0(snn_name, "_res.", cfg$clustering$resolution)]] <- cl
    SeuratObject::Idents(obj) <- cl
    # Seurat's one-time notice that uwot became the default is not a problem here
    umap_impl <- if (use_fast) paste0("scfastR::RunUMAPFast, scfastR ", scfast_ver)
                 else paste0("Seurat::RunUMAP(umap.method = \"uwot\"), uwot ", scp_pkg_version("uwot"))
    obj <- if (use_fast) tm("umap", scfastR::RunUMAPFast(obj, reduction = rep_red, dims = dims, n.neighbors = cfg$n_neighbors,
                                                         seed.use = seed, verbose = FALSE)) else
      tm("umap", withCallingHandlers(
      Seurat::RunUMAP(obj, reduction = rep_red, dims = dims, n.neighbors = cfg$n_neighbors,
                      umap.method = "uwot", seed.use = seed, verbose = FALSE),
      warning = function(w) if (grepl("default method for RunUMAP has changed", conditionMessage(w))) invokeRestart("muffleWarning")))
    # scale.data is never kept (exchange-spec: recomputed where needed); keeps the object small
    if (assay == "RNA" && "scale.data" %in% SeuratObject::Layers(obj[[assay]]))
      SeuratObject::LayerData(obj, assay = assay, layer = "scale.data") <- NULL
  } else {
    # 6. GPU path (exchange): slim -> cellwarden.gpu_core -> merge
    if (identical(pcs_setting, "auto")) {
      sd <- tm("pcs_auto_estimate", .s2_sample_pca_stdev(obj, hvg, npcs_compute, seed, assay))
      el <- s2_elbow_pcs(sd)
      pcs_used <- el$n
      pcs_info <- c(pcs_info, list(method = "elbow", elbow = el,
                                   basis = sprintf("irlba PCA of scaled HVGs on %d cells in R", min(ncol(obj), .S2_PCS_AUTO_SAMPLE))))
    } else pcs_used <- max(npcs_compute_fixed, 2L)
    ex_dir <- file.path(out_dir, "exchange")
    dir.create(ex_dir, recursive = TRUE, showWarnings = FALSE)
    slim <- file.path(ex_dir, "slim.h5ad")
    results <- file.path(ex_dir, "results.h5ad")
    unlink(c(slim, results))
    tm("write_slim", write_exchange(obj, slim, mode = "slim", run_id = cfg$run_id, stage = "S2",
                                    config_hash = cfg$config_hash, assay = assay,
                                    reductions = character(), graphs = FALSE))
    gpu_dev <- if (path == "exchange_cpu") "cpu"
               else if (identical(Sys.getenv("CELLWARDEN_ALLOW_CPU_AS_GPU"), "1")) "cpu" else "gpu"
    args <- c("-m", "cellwarden.gpu_core", "--in", slim, "--out", results,
              "--n-pcs", as.character(pcs_used), "--n-neighbors", as.character(cfg$n_neighbors),
              "--resolution", format(cfg$clustering$resolution), "--seed", as.character(seed),
              "--device", gpu_dev, "--run-id", cfg$run_id)
    if (integ == "harmony") args <- c(args, "--batch-key", batch_key)
    extra_env <- c(paste0("CELLWARDEN_GPU_PLAN=", plan$plan))
    scp_log("running: ", s2_python(), " ", paste(args, collapse = " "))
    t0 <- proc.time()[["elapsed"]]
    r <- s2_run_python(repo, args, extra_env)
    timings[["gpu_core"]] <- round(proc.time()[["elapsed"]] - t0, 3)
    for (l in r$stderr) if (nzchar(l)) scp_log("gpu_core: ", l)
    if (r$status != 0 || !file.exists(results)) {
      scp_stop("cellwarden.gpu_core failed (exit ", r$status, "):\n  ", paste(tail(c(r$stdout, r$stderr), 15), collapse = "\n  "))
    }
    gpu_summary <- .s2_parse_json_out(r$stdout)
    obj <- tm("merge", merge_exchange_results(obj, results, overwrite = TRUE, assay = assay))
    pcs_used <- if (!is.null(gpu_summary$n_pcs)) as.integer(gpu_summary$n_pcs) else pcs_used
    umap_impl <- paste0("cellwarden.gpu_core (", gpu_summary$device %||% gpu_dev, ")")
    leiden_impl <- paste0("cellwarden.gpu_core (", gpu_summary$device %||% gpu_dev, ")")
    if (integ == "harmony") integration_impl <- paste0("harmony via cellwarden.gpu_core (", gpu_summary$harmony_impl %||% "harmonypy/rapids",
                                                       ", ", gpu_summary$device %||% gpu_dev, ")")
    exchange_files <- list(slim = slim, results = results,
                           slim_sha256 = digest::digest(file = slim, algo = "sha256"),
                           results_sha256 = digest::digest(file = results, algo = "sha256"))
  }

  cl <- obj$seurat_clusters
  sizes <- table(cl)
  obj@misc$cellwarden_s2 <- list(run_id = cfg$run_id, config_hash = cfg$config_hash, device = plan$device,
                             plan = plan$plan, path = path, engine = engine, pcs = pcs_used, created_utc = .scp_utc())
  tm("save", scp_save_object(obj, out_dir, cfg$object_format))

  gpu_peak <- gpu_summary$peak_gpu_pool_bytes
  summary <- list(
    stage = "S2", run_id = cfg$run_id, config_hash = cfg$config_hash, created_utc = .scp_utc(),
    device = plan$device, plan = plan$plan, reason = plan$reason, plan_source = plan$source,
    path = switch(path, cpu = "cpu_seurat", scfast = "cpu_scfast", gpu_scfast = "gpu_scfast",
                  gpu = "gpu_exchange", exchange_cpu = "exchange_cpu"),
    engine = engine, scfast_version = scfast_ver,
    scfast_device = if (path %in% c("scfast", "gpu_scfast")) scfast_dev else NULL,
    path_reason = path_reason,
    gpu_core_device = if (path %in% c("cpu", "scfast")) NULL
                      else if (path == "gpu_scfast") "cuda" else (gpu_summary$device %||% NA),
    cells_in = cells_in, cells_out = ncol(obj),
    qc = list(review_action = review_action, decisions = dec_counts),
    normalization = cfg$normalization, preprocessing = pre_impl, n_hvg = n_hvg_used,
    pcs = pcs_used, pcs_detail = pcs_info,
    n_neighbors = cfg$n_neighbors, resolution = cfg$clustering$resolution, seed = seed,
    integration = list(method = integ, batch_key = if (integ == "harmony") batch_key else NULL,
                       implementation = integration_impl),
    clustering = list(implementation = leiden_impl),
    umap = list(implementation = umap_impl),
    clusters = length(sizes), cluster_sizes = as.list(sizes),
    reductions = SeuratObject::Reductions(obj), graphs = SeuratObject::Graphs(obj),
    timings_s = timings, total_s = round(sum(unlist(timings)), 3),
    peak_memory = list(r_peak_rss_gb = scp_peak_mem_gb(),
                       gpu_core_peak_host_rss_gb = if (!is.null(gpu_summary$peak_host_rss_bytes))
                         round(gpu_summary$peak_host_rss_bytes / 1024^3, 4) else NULL,
                       gpu_pool_peak_gb = if (!is.null(gpu_peak)) round(gpu_peak / 1024^3, 4) else NULL),
    gpu_core = gpu_summary,
    exchange = exchange_files,
    packages = list(R = paste(R.version$major, R.version$minor, sep = "."), Seurat = scp_pkg_version("Seurat"),
                    SeuratObject = scp_pkg_version("SeuratObject"), harmony = scp_pkg_version("harmony"),
                    leidenbase = scp_pkg_version("leidenbase"), scfastR = scp_pkg_version("scfastR"), uwot = scp_pkg_version("uwot"))
  )
  scp_write_json(summary, file.path(out_dir, "summary.json"))
  scp_log(sprintf("S2 done: %d cells, %d PCs, %d clusters, path %s", ncol(obj), pcs_used, length(sizes), summary$path))
  invisible(summary)
}
