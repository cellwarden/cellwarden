# Benchmark Seurat (the baseline engine) on one prepared dataset.
# Same steps, parameters and JSONL record format as bench/run_py.py
# (docs/RUST_ENGINE.md section 8).
#
#   Rscript bench/run_seurat.R --data bench/results/run1/data/n2638 \
#       --out bench/results/run1 --reps 3 --threads 8
#
# Threads: set OMP/BLAS thread variables before starting R (bench/run_bench.sh
# does this); they are recorded with each run.

T_PROCESS <- proc.time()[["elapsed"]]

parse_args <- function(args) {
  opt <- list(data = NULL, out = NULL, reps = 3L, threads = NA_integer_, n_hvg = 2000L,
              n_pcs = 30L, dims = 10L, k = 20L, resolution = 0.5, seed = 0L, skip_umap = FALSE)
  i <- 1L
  while (i <= length(args)) {
    key <- gsub("-", "_", sub("^--", "", args[[i]]))
    if (key == "skip_umap") { opt$skip_umap <- TRUE; i <- i + 1L; next }
    if (!key %in% names(opt)) stop("unknown argument --", args[[i]])
    val <- args[[i + 1L]]
    opt[[key]] <- if (is.numeric(opt[[key]])) as(val, class(opt[[key]])) else val
    i <- i + 2L
  }
  if (is.null(opt$data) || is.null(opt$out)) stop("--data and --out are required")
  opt
}
opt <- parse_args(commandArgs(trailingOnly = TRUE))

# ---------------------------------------------------------------- helpers (base R only)

reset_peak_rss <- function() {
  tryCatch({ cat("5", file = "/proc/self/clear_refs"); TRUE }, error = function(e) FALSE,
           warning = function(w) FALSE)
}
peak_rss_bytes <- function() {
  st <- tryCatch(readLines("/proc/self/status"), error = function(e) character())
  hwm <- grep("^VmHWM:", st, value = TRUE)
  if (length(hwm)) return(as.numeric(strsplit(trimws(sub("VmHWM:", "", hwm)), " +")[[1]][1]) * 1024)
  NA_real_  # macOS: no per-step peak available from base R
}
git_commit <- function() {
  root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
  out <- tryCatch(system2("git", c("-C", root, "rev-parse", "--short", "HEAD"), stdout = TRUE, stderr = FALSE),
                  error = function(e) "unknown")
  if (length(out)) out[[1]] else "unknown"
}

data_dir <- normalizePath(opt$data)
h5 <- file.path(data_dir, "matrix.h5")
n_cells <- length(readLines(file.path(data_dir, "cells.tsv"))) - 1L
prep_json <- file.path(dirname(data_dir), "prep.json")
dir.create(opt$out, recursive = TRUE, showWarnings = FALSE)
results <- file.path(opt$out, "results.jsonl")
dataset <- basename(data_dir)

state <- new.env()
state$rep <- 0L
state$warmup <- TRUE
base <- list(schema = "1.0", engine = "seurat", cells = n_cells, genes = NA, dataset = dataset,
             threads = if (is.na(opt$threads)) NULL else opt$threads, commit = git_commit(), device = "cpu")

write_rec <- function(fields) {
  rec <- c(base, list(rep = state$rep, warmup = state$warmup), fields)
  line <- jsonlite::toJSON(rec, auto_unbox = TRUE, null = "null", digits = NA, na = "null")
  cat(line, "\n", file = results, append = TRUE, sep = "")
}

step <- function(name, expr, kind = "compute") {
  reset_ok <- reset_peak_rss()
  t0 <- proc.time()[["elapsed"]]
  res <- tryCatch(list(ok = TRUE, value = force(expr)),
                  error = function(e) list(ok = FALSE, error = conditionMessage(e)))
  wall <- proc.time()[["elapsed"]] - t0
  rss <- peak_rss_bytes()
  write_rec(list(step = name, kind = kind, status = if (res$ok) "ok" else "error",
                 error = if (res$ok) NULL else res$error, wall_s = round(wall, 6),
                 transfer_s = 0, peak_rss_gb = round(rss / 1e9, 4),
                 rss_scope = if (reset_ok && !is.na(rss)) "step" else "process", peak_gpu_gb = NULL))
  if (!res$ok) stop(sprintf("step %s failed: %s", name, res$error), call. = FALSE)
  res$value
}

# ---------------------------------------------------------------- start-up

t0 <- proc.time()[["elapsed"]]
ok <- tryCatch({
  suppressPackageStartupMessages({ library(Seurat); library(hdf5r); library(leidenbase); library(igraph); library(jsonlite) })
  TRUE
}, error = function(e) { message("seurat: start-up failed: ", conditionMessage(e)); FALSE })
if (!ok) {
  # jsonlite may be the missing package, so write this record by hand
  cat(sprintf('{"schema":"1.0","engine":"seurat","cells":%d,"dataset":"%s","rep":0,"warmup":true,"step":"startup","kind":"startup","status":"unavailable","wall_s":null}\n',
              n_cells, dataset), file = results, append = TRUE)
  quit(status = 0)
}
if (file.exists(prep_json)) base$genes <- jsonlite::fromJSON(prep_json)$genes_qc
write_rec(list(step = "startup", kind = "startup", status = "ok", error = NULL,
               wall_s = round(proc.time()[["elapsed"]] - t0, 6),
               process_to_ready_s = round(proc.time()[["elapsed"]] - T_PROCESS, 6),
               peak_rss_gb = round(peak_rss_bytes() / 1e9, 4)))

# ---------------------------------------------------------------- environment

thread_vars <- c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS")
cpu_model <- tryCatch(sub(".*: ", "", grep("model name", readLines("/proc/cpuinfo"), value = TRUE)[1]),
                      error = function(e) NA)
gpus <- tryCatch(system2("nvidia-smi", c("--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                                          "--format=csv,noheader,nounits"), stdout = TRUE, stderr = FALSE),
                 error = function(e) character(), warning = function(w) character())
si <- sessionInfo()
env <- list(node = Sys.info()[["nodename"]], platform = R.version.string, cpu_model = cpu_model,
            logical_cpus = parallel::detectCores(),
            loadavg = tryCatch(as.numeric(strsplit(readLines("/proc/loadavg"), " ")[[1]][1:3]), error = function(e) NULL),
            threads_env = as.list(Sys.getenv(thread_vars)), gpus = gpus, commit = base$commit,
            blas = si$BLAS, lapack = si$LAPACK,
            # Seurat >= 5.1 runs Leiden (algorithm = 4) through leidenbase; older 5.0.x calls
            # Python leidenalg via reticulate (set RETICULATE_PYTHON to a Python that has it).
            leiden_backend = if (packageVersion("Seurat") >= "5.1.0") "leidenbase" else if (requireNamespace("leidenbase", quietly = TRUE)) "leidenbase (direct)" else
              sprintf("leidenalg via reticulate (%s)", Sys.getenv("RETICULATE_PYTHON", "default python")),
            packages = lapply(c(Seurat = "Seurat", SeuratObject = "SeuratObject", leidenbase = "leidenbase",
                                uwot = "uwot", Matrix = "Matrix"), function(p) as.character(packageVersion(p))),
            time_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"), config = opt)
dir.create(file.path(opt$out, "env"), showWarnings = FALSE)
writeLines(jsonlite::toJSON(env, auto_unbox = TRUE, pretty = TRUE, null = "null"),
           file.path(opt$out, "env", sprintf("seurat_%s.json", dataset)))
base$node <- env$node

# ---------------------------------------------------------------- parity outputs

write_outputs <- function(obj, outdir) {
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  pcs <- Embeddings(obj, "pca")[, seq_len(opt$n_pcs), drop = FALSE]
  nn <- FindNeighbors(obj, dims = seq_len(opt$dims), k.param = opt$k, return.neighbor = TRUE,
                      compute.SNN = FALSE, verbose = FALSE)
  idx <- Indices(nn[["RNA.nn"]])  # cells x k, 1-based, usually self first
  idx <- idx[match(colnames(obj), Cells(nn[["RNA.nn"]])), , drop = FALSE]
  self_i <- seq_len(nrow(idx))
  knn <- t(vapply(self_i, function(i) {
    r <- idx[i, ]
    r <- if (any(r == i)) r[r != i] else r[-length(r)]
    as.integer(r[seq_len(ncol(idx) - 1L)] - 1L)  # 0-based, self excluded
  }, integer(ncol(idx) - 1L)))
  con <- file(file.path(outdir, "pcs.f32"), "wb"); writeBin(as.numeric(t(pcs)), con, size = 4, endian = "little"); close(con)
  con <- file(file.path(outdir, "knn.i32"), "wb"); writeBin(as.integer(t(knn)), con, size = 4, endian = "little"); close(con)
  writeLines(colnames(obj), file.path(outdir, "barcodes.txt"))
  writeLines(as.character(obj$seurat_clusters), file.path(outdir, "labels.txt"))
  writeLines(jsonlite::toJSON(list(n_cells = ncol(obj), n_pcs = ncol(pcs), k = ncol(knn), order = "row-major",
                                   knn_base = 0L, knn_self = "excluded"), auto_unbox = TRUE),
             file.path(outdir, "meta.json"))
}

# leidenbase (and Seurat >= 5.1 Leiden) require seed >= 1; bench default is 0.
seurat_leiden_seed <- function(seed) {
  s <- as.integer(seed)
  if (s < 1L) 1L else s
}

# Prefer leidenbase on the SNN graph (avoids reticulate / leidenalg on Seurat 5.0.x).
seurat_find_clusters_leiden <- function(obj, resolution, seed, n.iter) {
  leiden_seed <- seurat_leiden_seed(seed)
  if (requireNamespace("leidenbase", quietly = TRUE) && requireNamespace("igraph", quietly = TRUE)) {
    gname <- paste0(DefaultAssay(obj), "_snn")
    snn <- obj@graphs[[gname]]
    g <- igraph::graph_from_adjacency_matrix(as(snn, "CsparseMatrix"), mode = "undirected",
                                            weighted = TRUE, diag = FALSE)
    res <- leidenbase::leiden_find_partition(
      g, partition_type = "RBConfigurationVertexPartition",
      edge_weights = igraph::E(g)$weight, resolution_parameter = resolution,
      seed = leiden_seed, num_iter = as.integer(n.iter)
    )
    obj$seurat_clusters <- factor(res$membership)
    return(obj)
  }
  FindClusters(obj, algorithm = 4, resolution = resolution, random.seed = leiden_seed,
               n.iter = n.iter, verbose = FALSE)
}

# ---------------------------------------------------------------- repeats

dims <- seq_len(opt$dims)  # leading PCs for neighbours and UMAP, as in seurat_pbmc3k.R
for (rep in 0:opt$reps) {
  state$rep <- rep
  state$warmup <- rep == 0L
  ok <- tryCatch({
    obj <- step("read", CreateSeuratObject(Read10X_h5(h5), min.cells = 0, min.features = 0), kind = "io")
    obj <- step("normalize", NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = 1e4, verbose = FALSE))
    obj <- step("hvg", FindVariableFeatures(obj, selection.method = "vst", nfeatures = opt$n_hvg, verbose = FALSE))
    obj <- step("scale", ScaleData(obj, features = VariableFeatures(obj), verbose = FALSE))
    obj <- step("pca", RunPCA(obj, npcs = opt$n_pcs, seed.use = opt$seed, verbose = FALSE))
    obj <- step("neighbors", FindNeighbors(obj, dims = dims, k.param = opt$k, verbose = FALSE))
    obj <- step("cluster", seurat_find_clusters_leiden(obj, opt$resolution, opt$seed, 2L))
    if (!opt$skip_umap) obj <- step("umap", RunUMAP(obj, dims = dims, seed.use = opt$seed, verbose = FALSE))
    TRUE
  }, error = function(e) { message("seurat rep ", rep, ": ", conditionMessage(e)); FALSE })
  if (!ok) quit(status = 1)
  if (rep == 1L || (opt$reps == 0L && rep == 0L)) write_outputs(obj, file.path(opt$out, "outputs", "seurat", dataset))
  rm(obj); invisible(gc())
}
cat(sprintf("seurat %s: done (%d measured repeats)\n", dataset, opt$reps))
