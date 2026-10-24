# scfastR: Seurat-facing wrappers around the scfast Rust kernels (docs/RUST_ENGINE.md).
#
#   PreprocessFast()    replaces NormalizeData() + FindVariableFeatures(vst) -> data layer, VariableFeatures
#   RunPCAFast()        replaces ScaleData() + RunPCA()     -> DimReduc "pca"
#   FindNeighborsFast() replaces FindNeighbors()            -> Graphs "<assay>_nn", "<assay>_snn"
#   FindClustersFast()  replaces FindClusters(algorithm=4)  -> seurat_clusters, Idents()
#   RunUMAPFast()       replaces RunUMAP()                  -> DimReduc "umap"
#   RunHarmonyFast()    replaces harmony::RunHarmony()      -> DimReduc "harmony"
#
# Low-level, Seurat-free: scfast_pca(), scfast_knn(), scfast_snn(), scfast_leiden(),
# scfast_harmony().
# The native calls (rs_*) are in R/extendr-wrappers.R; this file does argument checking,
# naming and the Seurat bookkeeping.

`%||%` <- function(a, b) if (is.null(a)) b else a

# ------------------------------------------------------------------ helpers

# Evaluate a native call; re-signal its error with classes "scfast_error" and, when a kernel
# is still a stub (unimplemented!()), "scfast_not_implemented".
.rust <- function(expr, call = sys.call(-1L)) {
  force(call)
  tryCatch(expr, error = function(e) {
    msg <- conditionMessage(e)
    cls <- c(if (grepl("not implemented", msg, fixed = TRUE)) "scfast_not_implemented",
             "scfast_error", "error", "condition")
    stop(structure(class = cls, list(message = msg, call = call)))
  })
}

.is_whole <- function(x) is.numeric(x) && all(is.finite(x)) && all(x == round(x))

.check_count <- function(x, name, min = 1) {
  if (!.is_whole(x) || length(x) != 1L || x < min) {
    stop(sprintf("'%s' must be a single whole number >= %s", name, min), call. = FALSE)
  }
  invisible(as.integer(x))
}

.check_seed <- function(x, name = "seed") {
  if (!.is_whole(x) || length(x) != 1L || x < 0) {
    stop(sprintf("'%s' must be a single non-negative whole number", name), call. = FALSE)
  }
  invisible(as.numeric(x))
}

.check_flag <- function(x, name) {
  if (!is.logical(x) || length(x) != 1L || is.na(x)) {
    stop(sprintf("'%s' must be TRUE or FALSE", name), call. = FALSE)
  }
  invisible(x)
}

.check_string <- function(x, name) {
  if (!is.character(x) || length(x) != 1L || is.na(x) || !nzchar(x)) {
    stop(sprintf("'%s' must be a single non-empty string", name), call. = FALSE)
  }
  invisible(x)
}

.check_seurat <- function(object) {
  if (!inherits(object, "Seurat")) {
    stop("'object' must be a Seurat object", call. = FALSE)
  }
  invisible(object)
}

# PCA / kNN device (docs/scfast-integration.md Phase 6). Validates the string and resolves
# it to the device the kernels will run on: "cpu", or "cuda:<i>" when scfastR was built with
# the `cuda` feature and a CUDA device is usable ("auto" picks it; an explicit "cuda" that is
# not usable is an error). The requested string is kept as attribute "requested" so callers
# can fall back to the CPU on a GPU failure only when the user asked for "auto".
.check_device <- function(device) {
  if (!is.null(attr(device, "requested"))) return(invisible(device)) # already resolved
  .check_string(device, "device")
  device <- tolower(device)
  if (!grepl("^(auto|cpu|cuda|gpu)(:[0-9]+)?$", device) || device %in% c("auto:0", "cpu:0")) {
    stop("'device' must be \"auto\", \"cpu\", \"cuda\" or \"cuda:<index>\"", call. = FALSE)
  }
  if (startsWith(device, "cuda") || startsWith(device, "gpu")) {
    if (!isTRUE(rs_info()$cuda_built)) {
      stop("scfastR was built without CUDA support; use device = \"cpu\" or \"auto\"", call. = FALSE)
    }
  }
  resolved <- .rust(rs_device(device))
  attr(resolved, "requested") <- device
  invisible(resolved)
}

# Run `fn(device)`; when the device came from "auto" and the GPU fails (an "scfast <kernel>:
# GPU: ..." error, e.g. out of device memory), warn and run again on the CPU.
.with_cpu_fallback <- function(device, fn) {
  requested <- attr(device, "requested") %||% device
  device <- as.character(device)
  if (device == "cpu" || requested != "auto") return(fn(device))
  tryCatch(fn(device), scfast_error = function(e) {
    msg <- conditionMessage(e)
    if (!grepl("^scfast [a-z_]+: GPU: ", msg)) stop(e)
    warning(sprintf("scfast: CUDA failed, falling back to the CPU: %s", sub("^scfast [a-z_]+: GPU: ", "", msg)),
            call. = FALSE)
    fn("cpu")
  })
}

# Any sparse/dense numeric matrix -> dgCMatrix. A dgCMatrix (or subclass, e.g. Graph) is
# returned as is: no copy.
.as_dgc <- function(m, name) {
  if (methods::is(m, "dgCMatrix")) return(m)
  if (!(is.matrix(m) || methods::is(m, "Matrix"))) {
    stop(sprintf("'%s' must be a dgCMatrix (or a matrix coercible to one)", name), call. = FALSE)
  }
  m <- methods::as(m, "CsparseMatrix")
  if (!methods::is(m, "generalMatrix")) m <- methods::as(m, "generalMatrix")
  if (!methods::is(m, "dMatrix")) m <- methods::as(m, "dMatrix")
  if (!methods::is(m, "dgCMatrix")) {
    stop(sprintf("could not convert '%s' to a dgCMatrix", name), call. = FALSE)
  }
  m
}

.timed <- function(expr) {
  t0 <- proc.time()[["elapsed"]]
  value <- expr
  list(value = value, seconds = proc.time()[["elapsed"]] - t0)
}

# ------------------------------------------------------------------ low-level API

#' Implicitly scaled PCA of a sparse expression matrix
#'
#' Runs the scfast randomised PCA on the centred and scaled matrix without building it:
#' the result matches `ScaleData()` + `RunPCA()` on the same features, except that
#' Seurat's `scale.max = 10` clipping is not applied.
#'
#' The `@p`, `@i` and `@x` arrays of `dgc` (genes x cells) are read in place by Rust as
#' the cells x genes CSR matrix they already are: no copy and no transpose.
#'
#' @param dgc A genes x cells `dgCMatrix` of log-normalised values (other matrix classes
#'   are converted, which copies).
#' @param npcs Number of components.
#' @param seed Random seed (non-negative whole number). Same seed, same result.
#' @param oversample,power.iters Randomised SVD settings; `NULL` keeps the engine defaults.
#' @param device `"auto"` (CUDA when scfastR was built with the `cuda` feature and a device
#'   is usable, else CPU), `"cpu"`, `"cuda"` or `"cuda:<index>"`. With `"auto"`, a GPU
#'   failure (e.g. out of device memory) falls back to the CPU with a warning. On CUDA the
#'   sparse products run on the device in single precision (results match the CPU to the
#'   parity thresholds, not bitwise).
#' @return A list with `embeddings` (cells x npcs, `U * d` as in Seurat), `loadings`
#'   (genes x npcs), `stdev` (`d / sqrt(n_cells - 1)`), the per-gene `gene_mean` and
#'   `gene_sd` used for scaling, and `device` (`"cpu"` or `"cuda:<index>"`).
#' @export
scfast_pca <- function(dgc, npcs = 30, seed = 0, oversample = NULL, power.iters = NULL, device = "auto") {
  dgc <- .as_dgc(dgc, "dgc")
  npcs <- .check_count(npcs, "npcs")
  seed <- .check_seed(seed)
  if (!is.null(oversample)) .check_count(oversample, "oversample", min = 0)
  if (!is.null(power.iters)) .check_count(power.iters, "power.iters", min = 0)
  device <- .check_device(device)
  if (npcs >= min(dim(dgc))) {
    stop(sprintf("'npcs' (%d) must be less than min(genes, cells) = %d", npcs, min(dim(dgc))), call. = FALSE)
  }
  res <- .pca_native(dgc, npcs, seed, oversample, power.iters, device)
  pcs <- paste0("PC_", seq_len(ncol(res$embeddings)))
  dimnames(res$embeddings) <- list(colnames(dgc), pcs)
  dimnames(res$loadings) <- list(rownames(dgc), pcs)
  names(res$gene_mean) <- names(res$gene_sd) <- rownames(dgc)
  res
}

.pca_native <- function(dgc, npcs, seed, oversample = NULL, power.iters = NULL, device = "cpu") {
  .with_cpu_fallback(device, function(dev) {
    .rust(rs_pca_csc(dgc@p, dgc@i, dgc@x, nrow(dgc), ncol(dgc), as.integer(npcs), as.numeric(seed),
                     as.integer(oversample %||% -1L), as.integer(power.iters %||% -1L), dev))
  })
}

#' k nearest neighbours of every row of a matrix
#'
#' @param mat Numeric matrix, cells x dimensions (e.g. PCA embeddings).
#' @param k Neighbours per cell, counting the cell itself (Seurat `k.param`); at least 2.
#' @param method `"auto"` (exact below the engine's cell-count threshold, HNSW above),
#'   `"exact"` or `"hnsw"`.
#' @param seed Seed for HNSW construction (exact search ignores it).
#' @param hnsw.m,ef.construction,ef.search HNSW settings, used with `method = "hnsw"`
#'   (`"auto"` uses the same defaults: 16, 200 and `max(2k, 64)`). HNSW results are not
#'   bit-reproducible across runs; exact search is.
#' @param auto.threshold Cell count where `"auto"` switches to HNSW; `NULL` keeps the
#'   engine default.
#' @param device `"auto"`, `"cpu"`, `"cuda"` or `"cuda:<index>"` (see [scfast_pca()]). On
#'   CUDA the search is exact brute force (cuBLAS GEMM + top-k) at every size and `method`
#'   is ignored; `k` must be at most 129.
#' @return A list with `idx` (n x k integer, 1-based, the cell itself first), `dist`
#'   (n x k Euclidean distances) and `device` (`"cpu"` or `"cuda:<index>"`).
#' @export
scfast_knn <- function(mat, k = 20, method = "auto", seed = 0, hnsw.m = 16,
                       ef.construction = 200, ef.search = NULL, auto.threshold = NULL,
                       device = "auto") {
  if (is.data.frame(mat)) mat <- as.matrix(mat)
  if (!is.matrix(mat) || !is.numeric(mat)) {
    stop("'mat' must be a numeric matrix (cells x dimensions)", call. = FALSE)
  }
  if (storage.mode(mat) != "double") storage.mode(mat) <- "double"
  k <- .check_count(k, "k", min = 2)
  .check_string(method, "method")
  if (!method %in% c("auto", "exact", "hnsw")) {
    stop("'method' must be one of \"auto\", \"exact\", \"hnsw\"", call. = FALSE)
  }
  seed <- .check_seed(seed)
  if (k > nrow(mat)) {
    stop(sprintf("'k' (%d) is larger than the number of cells (%d)", k, nrow(mat)), call. = FALSE)
  }
  hnsw.m <- .check_count(hnsw.m, "hnsw.m")
  ef.construction <- .check_count(ef.construction, "ef.construction")
  ef.search <- .check_count(ef.search %||% max(64L, 2L * k), "ef.search")
  if (!is.null(auto.threshold)) .check_count(auto.threshold, "auto.threshold", min = 0)
  device <- .check_device(device)
  res <- .with_cpu_fallback(device, function(dev) {
    .rust(rs_knn(mat, nrow(mat), ncol(mat), k, method, seed, hnsw.m, ef.construction,
                 ef.search, as.numeric(auto.threshold %||% -1), dev))
  })
  rownames(res$idx) <- rownames(res$dist) <- rownames(mat)
  res
}

#' Shared-nearest-neighbour graph from kNN indices
#'
#' Seurat's `ComputeSNN()`: Jaccard index of the neighbour sets (each including the cell
#' itself), entries below `prune` removed, diagonal kept.
#'
#' @param idx Integer matrix n x k of 1-based neighbour indices, the cell itself first (the
#'   `idx` element of [scfast_knn()]).
#' @param prune Seurat `prune.SNN`.
#' @return A symmetric n x n `dgCMatrix`, named by `rownames(idx)`.
#' @export
scfast_snn <- function(idx, prune = 1 / 15) {
  if (!is.matrix(idx) || !.is_whole(idx)) {
    stop("'idx' must be a matrix of whole numbers (1-based neighbour indices)", call. = FALSE)
  }
  if (storage.mode(idx) != "integer") storage.mode(idx) <- "integer"
  if (!is.numeric(prune) || length(prune) != 1L || !is.finite(prune) || prune < 0 || prune > 1) {
    stop("'prune' must be a single number in [0, 1]", call. = FALSE)
  }
  g <- .snn_native(idx, prune)
  if (!is.null(rownames(idx))) g@Dimnames <- list(rownames(idx), rownames(idx))
  g
}

.snn_native <- function(idx, prune) {
  parts <- .rust(rs_snn(idx, nrow(idx), ncol(idx), as.numeric(prune)))
  methods::new("dgCMatrix", i = parts$i, p = parts$p, x = parts$x, Dim = c(parts$n, parts$n))
}

#' Leiden clustering of a graph
#'
#' Leiden with the RB objective (what Seurat's `algorithm = 4` and Scanpy optimise): on the
#' CPU with single-clustering, or on a GPU with cuGraph when scfastR was built with the
#' `cugraph` feature and `libcugraph_c.so` loads. A non-symmetric graph is symmetrised as
#' `(G + t(G)) / 2`, with a warning.
#'
#' @param graph Square weighted adjacency (`dgCMatrix` or Seurat `Graph`), e.g.
#'   `object[["RNA_snn"]]`.
#' @param resolution RB resolution (Seurat `resolution`).
#' @param seed Random seed (Seurat `random.seed`).
#' @param n.iter Leiden iterations (Seurat `n.iter`, leidenalg `n_iterations`; cuGraph
#'   `max_level`).
#' @param device `"auto"` (cuGraph when available, else CPU), `"cpu"`, `"cuda"` or
#'   `"cuda:<index>"` (an error when cuGraph is not available). The device used is returned
#'   in the `"device"` attribute.
#' @return Integer vector of 0-based labels numbered by decreasing cluster size, named by
#'   `colnames(graph)`, with attribute `device` (`"cpu"` or `"cuda:<index>"`).
#' @export
scfast_leiden <- function(graph, resolution = 0.5, seed = 0, n.iter = 2, device = "auto") {
  device <- .check_leiden_device(device)
  graph <- .prepare_graph(graph)
  .check_resolution(resolution)
  if (length(resolution) != 1L) stop("'resolution' must be a single number", call. = FALSE)
  labels <- .leiden_native(graph, resolution, .check_seed(seed), .check_count(n.iter, "n.iter"), device)
  names(labels) <- colnames(graph)
  attr(labels, "device") <- device
  labels
}

# Leiden device string -> the device the native side will run on ("cpu" or "cuda:<i>").
# "auto" resolves to cuGraph only when scfastR was built with the `cugraph` feature, the
# library loads and a CUDA device is usable; an explicit "cuda" that is not usable errors.
# Separate from .check_device() (PCA / kNN, Phase 6): the two GPU paths are built and
# detected independently.
.check_leiden_device <- function(device) {
  .check_string(device, "device")
  device <- tolower(device)
  if (!grepl("^(auto|cpu|cuda|gpu)(:[0-9]+)?$", device) || device %in% c("auto:0", "cpu:0")) {
    stop("'device' must be \"auto\", \"cpu\", \"cuda\" or \"cuda:<index>\"", call. = FALSE)
  }
  .rust(rs_leiden_device(device))$resolved
}

.check_resolution <- function(resolution) {
  if (!is.numeric(resolution) || length(resolution) < 1L || any(!is.finite(resolution)) || any(resolution < 0)) {
    stop("'resolution' must be non-negative number(s)", call. = FALSE)
  }
  invisible(resolution)
}

.prepare_graph <- function(graph) {
  graph <- .as_dgc(graph, "graph")
  if (nrow(graph) != ncol(graph)) stop("'graph' must be a square matrix", call. = FALSE)
  if (!.rust(rs_is_symmetric(graph@p, graph@i, graph@x, ncol(graph), 1e-6))) {
    warning("graph is not symmetric; clustering (G + t(G)) / 2", call. = FALSE)
    graph <- .as_dgc((graph + Matrix::t(graph)) / 2, "graph")
  }
  graph
}

.leiden_native <- function(graph, resolution, seed, n.iter, device = "cpu") {
  .rust(rs_leiden_on(graph@p, graph@i, graph@x, ncol(graph), as.numeric(resolution),
                     as.numeric(seed), as.integer(n.iter), device))
}

#' Engine information
#'
#' @return A list: package and engine versions, rayon thread count (set the environment
#'   variable `RAYON_NUM_THREADS` before the first scfast call to change it), and the CUDA
#'   state: `cuda_built` (feature compiled in; `cuda` is the same value), `cuda_available`
#'   (driver loads and a device is visible), `device_count`, `device_name`, `free_gb`,
#'   `total_gb` (device 0) and `cuda_error` (why CUDA is unavailable, else `NULL`).
#' @export
scfast_info <- function() {
  c(list(package = as.character(utils::packageVersion("scfastR"))), rs_info())
}

# ------------------------------------------------------------------ Seurat: PCA

# The assay's "data" layer exactly as stored (no LayerData() subsetting, which copies),
# with its feature and cell names.
.data_layer <- function(object, assay) {
  a <- object[[assay]]
  if (inherits(a, "Assay5")) {
    layers <- SeuratObject::Layers(a)
    if (!"data" %in% layers) {
      if (any(startsWith(layers, "data."))) {
        stop(sprintf("assay '%s' has split data layers (%s); run JoinLayers() first",
                     assay, paste(grep("^data\\.", layers, value = TRUE), collapse = ", ")), call. = FALSE)
      }
      stop(sprintf("assay '%s' has no 'data' layer; run NormalizeData() first", assay), call. = FALSE)
    }
    m <- methods::slot(a, "layers")[["data"]]
    features <- SeuratObject::Features(a, layer = "data")
    cells <- SeuratObject::Cells(a, layer = "data")
  } else if (inherits(a, "Assay")) {
    m <- methods::slot(a, "data")
    features <- rownames(m)
    cells <- colnames(m)
  } else {
    stop(sprintf("assay '%s' has unsupported class '%s'", assay, class(a)[1]), call. = FALSE)
  }
  if (!identical(dim(m), c(length(features), length(cells)))) {
    stop(sprintf("assay '%s': the data layer is not stored as features x cells", assay), call. = FALSE)
  }
  list(m = m, features = features, cells = cells)
}

# The genes x cells dgCMatrix passed to Rust: the data layer restricted to `features`
# (one copy of the HVG rows; none when all features are used in stored order).
.pca_input <- function(object, assay, features = NULL, verbose = FALSE) {
  layer <- .data_layer(object, assay)
  if (is.null(features)) {
    features <- SeuratObject::VariableFeatures(object, assay = assay)
    if (length(features) == 0L) {
      stop(sprintf("assay '%s' has no variable features; run FindVariableFeatures() first or pass 'features'", assay),
           call. = FALSE)
    }
  }
  if (!is.character(features)) stop("'features' must be a character vector", call. = FALSE)
  features <- unique(features)
  idx <- match(features, layer$features)
  if (anyNA(idx)) {
    warning(sprintf("%d of %d requested features are not in assay '%s' and are skipped: %s",
                    sum(is.na(idx)), length(features), assay,
                    paste(utils::head(features[is.na(idx)], 10), collapse = ", ")), call. = FALSE)
    features <- features[!is.na(idx)]
    idx <- idx[!is.na(idx)]
  }
  if (length(idx) < 2L) stop("fewer than 2 usable features", call. = FALSE)
  m <- layer$m
  if (!identical(idx, seq_len(nrow(m)))) m <- m[idx, , drop = FALSE]
  m <- .as_dgc(m, "data layer")
  # Seurat drops zero-variance features before PCA; all-zero rows are the case that occurs.
  nnz <- tabulate(m@i + 1L, nbins = nrow(m))
  if (any(nnz == 0L)) {
    keep <- nnz > 0L
    warning(sprintf("%d features have zero variance and are skipped: %s", sum(!keep),
                    paste(utils::head(features[!keep], 10), collapse = ", ")), call. = FALSE)
    m <- m[keep, , drop = FALSE]
    features <- features[keep]
  }
  list(m = m, features = features, cells = layer$cells)
}

#' PCA on the log-normalised data with implicit scaling (no ScaleData)
#'
#' Replaces `ScaleData()` + `RunPCA()`. The assay's `data` layer restricted to the variable
#' features is passed to Rust in place, and the scaled matrix is never built. Stores a
#' standard `DimReduc`, so `ElbowPlot()`, `DimPlot()`, `FindNeighbors()`, `RunUMAP()`
#' work unchanged.
#'
#' @param object A Seurat object (after `NormalizeData()` and `FindVariableFeatures()`).
#' @param assay Assay to use; default `DefaultAssay(object)`.
#' @param features Features to use; default `VariableFeatures()`.
#' @param npcs Number of components (capped at `min(features, cells) - 1`, as in Seurat).
#' @param seed.use Random seed (`seed` also works, by partial matching).
#' @param reduction.name,reduction.key Name and key of the stored reduction.
#' @param device `"auto"` (CUDA when scfastR was built with the `cuda` feature and a device
#'   is usable, else CPU), `"cpu"`, `"cuda"` or `"cuda:<index>"`; see [scfast_pca()]. The
#'   device used is recorded in `Misc(reduction)$scfast$device`.
#' @param oversample,power.iters Randomised SVD accuracy knobs (extra random directions and
#'   power iterations); `NULL` keeps the engine defaults (10 and 4). Raise them (e.g. 40 and
#'   8) when the spectrum is flat beyond the leading components and later PCs must match
#'   an exact SVD closely; cost grows roughly linearly with `power.iters`.
#' @param verbose Print progress messages.
#' @return The Seurat object with the reduction added.
#' @export
RunPCAFast <- function(object, assay = NULL, features = NULL, npcs = 30, seed.use = 0,
                       reduction.name = "pca", reduction.key = "PC_", device = "auto",
                       oversample = NULL, power.iters = NULL, verbose = TRUE) {
  .check_seurat(object)
  assay <- assay %||% SeuratObject::DefaultAssay(object)
  .check_string(assay, "assay")
  if (!assay %in% SeuratObject::Assays(object)) {
    stop(sprintf("assay '%s' not found in the object", assay), call. = FALSE)
  }
  npcs <- .check_count(npcs, "npcs")
  seed.use <- .check_seed(seed.use, "seed.use")
  .check_string(reduction.name, "reduction.name")
  .check_string(reduction.key, "reduction.key")
  device <- .check_device(device)
  if (!is.null(oversample)) .check_count(oversample, "oversample", min = 0)
  if (!is.null(power.iters)) .check_count(power.iters, "power.iters", min = 0)
  .check_flag(verbose, "verbose")

  input <- .pca_input(object, assay, features, verbose)
  max.pcs <- min(dim(input$m)) - 1L
  if (max.pcs < 1L) stop("need at least 2 features and 2 cells", call. = FALSE)
  if (npcs > max.pcs) {
    warning(sprintf("npcs (%d) reduced to %d (min(features, cells) - 1)", npcs, max.pcs), call. = FALSE)
    npcs <- max.pcs
  }
  run <- .timed(.pca_native(input$m, npcs, seed.use, oversample, power.iters, device))
  res <- run$value
  pcs <- paste0(reduction.key, seq_len(ncol(res$embeddings)))
  dimnames(res$embeddings) <- list(input$cells, pcs)
  dimnames(res$loadings) <- list(input$features, pcs)
  names(res$gene_mean) <- names(res$gene_sd) <- input$features
  reduc <- SeuratObject::CreateDimReducObject(
    embeddings = res$embeddings,
    loadings = res$loadings,
    stdev = res$stdev,
    key = reduction.key,
    assay = assay,
    misc = list(
      # Every scaled feature has unit variance (no clipping), as total.variance in RunPCA.
      total.variance = length(input$features),
      scfast = list(kernel = "implicit_scaled_pca", engine = scfast_info()$version,
                    seed = seed.use, oversample = oversample, power.iters = power.iters,
                    seconds = run$seconds, device = res$device,
                    gene_mean = res$gene_mean, gene_sd = res$gene_sd)
    )
  )
  if (verbose) {
    message(sprintf("scfast PCA: %d cells x %d features, %d PCs in %.2f s on %s (no ScaleData)",
                    length(input$cells), length(input$features), npcs, run$seconds, res$device))
  }
  object[[reduction.name]] <- reduc
  object <- SeuratObject::LogSeuratCommand(object = object)
  object
}

# ------------------------------------------------------------------ Seurat: neighbours

#' kNN and SNN graphs on a reduction
#'
#' Replaces `FindNeighbors()` (Annoy + `ComputeSNN`). Stores the kNN graph (0/1 adjacency,
#' the cell itself included) and the SNN graph (Jaccard weights, pruned) as Seurat `Graph`
#' objects named `<assay>_nn` and `<assay>_snn`, exactly where `FindNeighbors()` puts them.
#'
#' @param object A Seurat object with the reduction.
#' @param reduction Reduction to use.
#' @param dims Dimensions of the reduction to use.
#' @param k.param Neighbours per cell, including the cell itself.
#' @param prune.SNN SNN weights below this are set to zero.
#' @param method `"auto"`, `"exact"` or `"hnsw"` (see [scfast_knn()]).
#' @param graph.name `NULL` (default names), or two names (kNN graph, SNN graph). As in
#'   Seurat, a single name stores only the kNN graph.
#' @param random.seed Seed for HNSW (exact search ignores it).
#' @param device `"auto"`, `"cpu"`, `"cuda"` or `"cuda:<index>"`; see [scfast_knn()]. The
#'   SNN step always runs on the CPU.
#' @param verbose Print progress messages.
#' @return The Seurat object with the graphs added.
#' @export
FindNeighborsFast <- function(object, reduction = "pca", dims = 1:10, k.param = 20,
                              prune.SNN = 1 / 15, method = "auto", graph.name = NULL,
                              random.seed = 0, device = "auto", verbose = TRUE) {
  .check_seurat(object)
  .check_string(reduction, "reduction")
  if (!reduction %in% SeuratObject::Reductions(object)) {
    stop(sprintf("reduction '%s' not found; run RunPCAFast() or RunPCA() first", reduction), call. = FALSE)
  }
  k.param <- .check_count(k.param, "k.param", min = 2)
  if (!is.numeric(prune.SNN) || length(prune.SNN) != 1L || !is.finite(prune.SNN) || prune.SNN < 0 || prune.SNN > 1) {
    stop("'prune.SNN' must be a single number in [0, 1]", call. = FALSE)
  }
  .check_string(method, "method")
  if (!method %in% c("auto", "exact", "hnsw")) {
    stop("'method' must be one of \"auto\", \"exact\", \"hnsw\"", call. = FALSE)
  }
  random.seed <- .check_seed(random.seed, "random.seed")
  device <- .check_device(device)
  .check_flag(verbose, "verbose")

  reduc <- object[[reduction]]
  assay <- SeuratObject::DefaultAssay(reduc)
  emb <- SeuratObject::Embeddings(reduc)
  if (!.is_whole(dims) || length(dims) < 1L || any(dims < 1)) {
    stop("'dims' must be positive whole numbers", call. = FALSE)
  }
  if (max(dims) > ncol(emb)) stop("More dimensions specified in dims than have been computed", call. = FALSE)
  if (anyDuplicated(dims)) stop("'dims' contains duplicates", call. = FALSE)
  data.use <- emb[, dims, drop = FALSE]
  cells <- rownames(data.use)
  n <- nrow(data.use)
  if (n < k.param) {
    warning("k.param set larger than number of cells. Setting k.param to number of cells - 1.", call. = FALSE)
    k.param <- n - 1L
    if (k.param < 2L) stop("need at least 3 cells", call. = FALSE)
  }
  if (is.null(graph.name)) {
    graph.name <- paste0(assay, "_", c("nn", "snn"))
  } else if (!is.character(graph.name) || !length(graph.name) %in% 1:2 || anyNA(graph.name)) {
    stop("'graph.name' must be NULL or one or two names", call. = FALSE)
  }
  if (length(graph.name) == 1L) {
    message("Only one graph name supplied, storing nearest-neighbor graph only")
  }

  t0 <- proc.time()[["elapsed"]]
  nn <- scfast_knn(data.use, k = k.param, method = method, seed = random.seed, device = device)
  t.knn <- proc.time()[["elapsed"]] - t0
  # Same construction as Seurat: row i has a 1 at each of its k neighbours (itself included).
  nn.graph <- Matrix::sparseMatrix(i = rep(seq_len(n), times = k.param), j = as.vector(nn$idx),
                                   x = 1, dims = c(n, n), dimnames = list(cells, cells))
  graphs <- list(SeuratObject::as.Graph(nn.graph))
  t.snn <- 0
  if (length(graph.name) == 2L) {
    t1 <- proc.time()[["elapsed"]]
    snn <- .snn_native(nn$idx, prune.SNN)
    snn@Dimnames <- list(cells, cells)
    graphs[[2L]] <- SeuratObject::as.Graph(snn)
    t.snn <- proc.time()[["elapsed"]] - t1
  }
  for (ii in seq_along(graph.name)) {
    SeuratObject::DefaultAssay(graphs[[ii]]) <- assay
    object[[graph.name[[ii]]]] <- graphs[[ii]]
  }
  if (verbose) {
    message(sprintf("scfast neighbours: %d cells, k = %d, %d dims: kNN %.2f s on %s, SNN %.2f s",
                    n, k.param, length(dims), t.knn, nn$device, t.snn))
  }
  object <- SeuratObject::LogSeuratCommand(object = object)
  object
}

# ------------------------------------------------------------------ Seurat: UMAP

# a, b of the low-dimensional similarity 1 / (1 + a d^(2b)) for min_dist / spread.
.find_ab <- function(spread, min.dist) {
  if (requireNamespace("uwot", quietly = TRUE) && exists("find_ab_params", envir = asNamespace("uwot"))) {
    return(unname(utils::getFromNamespace("find_ab_params", "uwot")(spread, min.dist)))
  }
  x <- seq(0, spread * 3, length.out = 300)
  y <- ifelse(x < min.dist, 1, exp(-(x - min.dist) / spread))
  unname(stats::coef(stats::nls(y ~ 1 / (1 + a * x^(2 * b)), start = list(a = 1, b = 1))))
}

#' UMAP with the scfast engine, replacing `RunUMAP()`
#'
#' Seurat's `RunUMAP()` defaults (uwot, cosine metric, `min.dist` 0.3, spread 1): the rows of
#' the reduction are L2-normalised, their kNN (scfast, cell included) gives cosine distances
#' `||u - v||^2 / 2`, and the scfast parallel SGD lays out the fuzzy graph from the first two
#' dimensions.
#'
#' @param object A Seurat object with the reduction.
#' @param reduction Reduction to use.
#' @param dims Dimensions of the reduction to use.
#' @param n.neighbors Neighbours per cell, including the cell itself.
#' @param nn.method kNN search, `"exact"`, `"hnsw"` or `"auto"` (see [scfast_knn()]).
#' @param min.dist,spread Curve of the low-dimensional similarity.
#' @param n.epochs `NULL`: 500 below 10,000 cells, else 200.
#' @param negative.sample.rate,learning.rate SGD settings.
#' @param seed.use Random seed.
#' @param reduction.name,reduction.key Name and key of the stored DimReduc.
#' @param verbose Print progress messages.
#' @return The Seurat object with the reduction added.
#' @export
RunUMAPFast <- function(object, reduction = "pca", dims = 1:10, n.neighbors = 30, nn.method = "exact", min.dist = 0.3,
                        spread = 1, n.epochs = NULL, negative.sample.rate = 5, learning.rate = 1,
                        seed.use = 42, reduction.name = "umap", reduction.key = "umap_", verbose = TRUE) {
  .check_seurat(object)
  .check_string(reduction, "reduction")
  if (!reduction %in% SeuratObject::Reductions(object)) {
    stop(sprintf("reduction '%s' not found", reduction), call. = FALSE)
  }
  min.dist <- .check_positive_number(min.dist, "min.dist")
  spread <- .check_positive_number(spread, "spread")
  learning.rate <- .check_positive_number(learning.rate, "learning.rate")
  negative.sample.rate <- .check_count(negative.sample.rate, "negative.sample.rate", min = 0)
  seed.use <- .check_seed(seed.use, "seed.use")
  .check_flag(verbose, "verbose")
  reduc <- object[[reduction]]
  emb <- SeuratObject::Embeddings(reduc)
  if (!.is_whole(dims) || length(dims) < 2L || any(dims < 1) || anyDuplicated(dims)) {
    stop("'dims' must be at least two distinct positive whole numbers", call. = FALSE)
  }
  if (max(dims) > ncol(emb)) stop("More dimensions specified in dims than have been computed", call. = FALSE)
  x <- emb[, dims, drop = FALSE]
  n <- nrow(x)
  k <- min(.check_count(n.neighbors, "n.neighbors", min = 2), n)
  n.epochs <- .check_count(n.epochs %||% if (n <= 10000L) 500L else 200L, "n.epochs")
  ab <- .find_ab(spread, min.dist)

  run <- .timed({
    nn <- scfast_knn(x / sqrt(rowSums(x^2)), k = k, method = nn.method, seed = seed.use, device = "cpu")
    .rust(rs_umap(nn$idx, nn$dist^2 / 2, x[, 1:2], n, k, ab[[1]], ab[[2]], n.epochs,
                  negative.sample.rate, learning.rate, seed.use))
  })
  um <- run$value
  dimnames(um) <- list(rownames(x), paste0(reduction.key, 1:2))
  object[[reduction.name]] <- SeuratObject::CreateDimReducObject(embeddings = um, key = reduction.key,
                                                                 assay = SeuratObject::DefaultAssay(reduc))
  if (verbose) {
    message(sprintf("scfast UMAP: %d cells, k = %d, %d dims, %d epochs in %.2f s", n, k, length(dims), n.epochs, run$seconds))
  }
  SeuratObject::LogSeuratCommand(object = object)
}

# ------------------------------------------------------------------ Seurat: clustering

# Seurat's GroupSingletons(): each cluster of one cell joins the cluster it is most connected
# to (mean SNN weight to that cluster's cells), processed in order of first appearance; ties
# are broken as Seurat does (set.seed(1); sample()), but the caller's RNG state is restored.
.group_singletons <- function(labels, graph, verbose) {
  sizes <- tabulate(labels + 1L, nbins = max(labels) + 1L)
  singletons <- intersect(unique(labels), which(sizes == 1L) - 1L)
  if (length(singletons) == 0L) return(labels)
  cluster.ids <- setdiff(unique(labels), singletons)
  if (length(cluster.ids) == 0L) return(labels)
  had.seed <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  if (had.seed) old.seed <- get(".Random.seed", envir = globalenv(), inherits = FALSE)
  on.exit(if (had.seed) assign(".Random.seed", old.seed, envir = globalenv())
          else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE))
            rm(".Random.seed", envir = globalenv()), add = TRUE)
  for (s in singletons) {
    cell <- which(labels == s)
    rng <- if (graph@p[cell] < graph@p[cell + 1L]) seq.int(graph@p[cell] + 1L, graph@p[cell + 1L]) else integer()
    nbr <- graph@i[rng] + 1L
    w <- graph@x[rng]
    keep <- nbr != cell
    nbr.lab <- labels[nbr[keep]]
    w <- w[keep]
    sizes <- tabulate(labels + 1L, nbins = max(labels) + 1L)
    conn <- vapply(cluster.ids, function(j) sum(w[nbr.lab == j]) / sizes[j + 1L], numeric(1))
    best <- cluster.ids[which(conn == max(conn))]
    if (length(best) > 1L) {
      set.seed(1)
      best <- best[sample.int(length(best), 1L)]
    }
    labels[cell] <- best
  }
  if (verbose) {
    message(paste(length(singletons), "singletons identified.", length(unique(labels)), "final clusters."))
  }
  labels
}

#' Leiden clustering of the SNN graph
#'
#' Replaces `FindClusters(algorithm = 4)` (leidenalg, RB objective) with the scfast Leiden
#' (CPU, or cuGraph on a GPU with `device`). Writes the same outputs: a meta data column
#' `<graph.name>_res.<resolution>` per resolution, `seurat_clusters`, and `Idents()` (factor
#' with levels 0..n-1).
#'
#' @param object A Seurat object with an SNN graph (from `FindNeighborsFast()` or
#'   `FindNeighbors()`).
#' @param graph.name Graph to cluster; default `<DefaultAssay>_snn`.
#' @param resolution Resolution(s); one column per value, `Idents()` from the last.
#' @param random.seed Random seed (`seed = ` is accepted as an alias).
#' @param n.iter Leiden iterations.
#' @param group.singletons Merge single-cell clusters into their most connected cluster,
#'   as Seurat does.
#' @param device `"auto"` (cuGraph Leiden when scfastR was built with the `cugraph` feature
#'   and a CUDA device is usable, else CPU), `"cpu"`, `"cuda"` or `"cuda:<index>"`.
#' @param verbose Print progress messages.
#' @param ... `seed` (alias of `random.seed`); `algorithm` (only 4 / "leiden") and `method`
#'   (ignored) are accepted so a `FindClusters()` call can be renamed unchanged.
#' @return The Seurat object with clusters added.
#' @export
FindClustersFast <- function(object, graph.name = NULL, resolution = 0.5, random.seed = 0,
                             n.iter = 2, group.singletons = TRUE, device = "auto", verbose = TRUE, ...) {
  .check_seurat(object)
  device <- .check_leiden_device(device)
  dots <- list(...)
  if (length(dots) && (is.null(names(dots)) || any(!nzchar(names(dots))))) {
    stop("unnamed arguments in '...'", call. = FALSE)
  }
  unknown <- setdiff(names(dots), c("seed", "algorithm", "method"))
  if (length(unknown)) {
    stop(sprintf("unused argument(s): %s", paste(unknown, collapse = ", ")), call. = FALSE)
  }
  if (!is.null(dots$seed)) {
    if (!missing(random.seed)) stop("give either 'random.seed' or 'seed', not both", call. = FALSE)
    random.seed <- dots$seed
  }
  if (!is.null(dots$algorithm) && !(tolower(as.character(dots$algorithm)) %in% c("4", "leiden"))) {
    stop("FindClustersFast() implements Leiden only (algorithm = 4)", call. = FALSE)
  }
  graph.name <- graph.name %||% paste0(SeuratObject::DefaultAssay(object), "_snn")
  .check_string(graph.name, "graph.name")
  .check_resolution(resolution)
  random.seed <- .check_seed(random.seed, "random.seed")
  n.iter <- .check_count(n.iter, "n.iter")
  .check_flag(group.singletons, "group.singletons")
  .check_flag(verbose, "verbose")
  if (!graph.name %in% names(object)) {
    stop("Provided graph.name not present in Seurat object", call. = FALSE)
  }
  graph <- object[[graph.name]]
  if (!methods::is(graph, "Graph")) {
    stop("Provided graph.name does not correspond to a graph object.", call. = FALSE)
  }
  cells <- colnames(graph)
  if (!setequal(cells, SeuratObject::Cells(object))) {
    stop(sprintf("graph '%s' does not cover the same cells as the object", graph.name), call. = FALSE)
  }
  graph <- .prepare_graph(graph)

  results <- data.frame(row.names = cells)
  for (r in resolution) {
    run <- .timed(.leiden_native(graph, r, random.seed, n.iter, device))
    labels <- run$value
    if (group.singletons) labels <- .group_singletons(labels, graph, verbose)
    col <- paste0(graph.name, "_res.", r)
    results[[col]] <- factor(labels, levels = sort(unique(labels)))
    if (verbose) {
      message(sprintf("scfast Leiden (%s): %d clusters at resolution %s in %.2f s",
                      device, length(unique(labels)), format(r), run$seconds))
    }
  }
  object <- SeuratObject::AddMetaData(object, metadata = results)
  last <- names(results)[ncol(results)]
  SeuratObject::Idents(object = object, replace = TRUE) <- object[[last, drop = TRUE]]
  lv <- levels(SeuratObject::Idents(object))
  SeuratObject::Idents(object) <- factor(SeuratObject::Idents(object), levels = lv[order(as.numeric(lv))])
  object[["seurat_clusters"]] <- SeuratObject::Idents(object)
  object <- SeuratObject::LogSeuratCommand(object = object)
  object
}

# ------------------------------------------------------------------ Harmony

# Advanced Harmony settings: harmony_options() defaults, overridden by a list (or the
# result of harmony::harmony_options()) with any of these names.
.harmony_options <- function(.options) {
  opts <- list(alpha = 0.2, tau = 0, block.size = 0.05, max.iter.cluster = 4,
               epsilon.cluster = 1e-3, epsilon.harmony = 1e-2, batch.prop.cutoff = 1e-5)
  if (is.null(.options)) return(opts)
  if (!is.list(.options)) stop("'.options' must be NULL or a list such as harmony::harmony_options()", call. = FALSE)
  unknown <- setdiff(names(.options), names(opts))
  if (length(unknown)) stop(sprintf("unknown '.options': %s", paste(unknown, collapse = ", ")), call. = FALSE)
  opts[names(.options)] <- .options
  for (nm in names(opts)) {
    v <- opts[[nm]]
    if (!is.numeric(v) || length(v) != 1L || is.na(v)) stop(sprintf("'.options$%s' must be a single number", nm), call. = FALSE)
  }
  opts
}

#' Harmony batch integration of an embedding matrix
#'
#' Runs Harmony (Korsunsky et al. 2019) in the scfast engine: the algorithm of the harmony
#' R package 2.x (`RunHarmony()`; soft k-means with a batch-diversity penalty on the
#' cosine-normalised cells, then a per-cluster ridge regression of the batch effect),
#' with the same defaults. Computes in double precision, in parallel; the result is
#' identical for a given `seed` on any number of threads, but not equal to harmony's own
#' (which draws from R's random number generator), see `init.centroids`.
#'
#' @param emb Numeric matrix, cells x dimensions (e.g. PCA embeddings).
#' @param batches One batch label per row of `emb` (a vector or factor; unused factor
#'   levels are dropped, `NA` is an error). One batch variable.
#' @param theta Diversity penalty (0 disables it).
#' @param lambda Ridge penalty; `NULL` (the harmony 2.x default) estimates it per cluster
#'   and batch as `alpha` times the expected number of cells. harmony 1.x defaulted to 1.
#' @param sigma Width of the soft k-means clusters.
#' @param nclust Number of clusters; `NULL` is `min(round(cells / 30), 100)`.
#' @param max.iter Maximum rounds of clustering + correction (`RunHarmony(max_iter)`).
#' @param seed Seed for the k-means initialisation and the block order.
#' @param device `"auto"` or `"cpu"` (there is no GPU Harmony yet, so both run on the CPU);
#'   `"cuda"` is an error.
#' @param early.stop `FALSE` runs all `max.iter` rounds (`RunHarmony(early_stop)`).
#' @param .options `NULL`, `harmony::harmony_options()` or a list with any of `alpha`,
#'   `tau`, `block.size`, `max.iter.cluster`, `epsilon.cluster`, `epsilon.harmony`,
#'   `batch.prop.cutoff`.
#' @param init.centroids Optional nclust x dimensions matrix of starting centroids in
#'   place of k-means (e.g. `t(harmonyObj$Y)` of a harmony object after
#'   `init_cluster_cpp()`, for a like-for-like comparison with the R package).
#' @param kmeans.nstart,kmeans.iter Seeded k-means starts and Lloyd steps for the initial
#'   centroids (harmony 2.x: 1 and 10).
#' @return A list with `embeddings` (the corrected matrix, cells x dimensions, named as
#'   `emb`), `objective` (per round, starting with the initial value), `objective_kmeans`
#'   (per clustering pass), `kmeans_rounds`, `iters`, `converged`, `nclust` and `device`.
#' @export
scfast_harmony <- function(emb, batches, theta = 2, lambda = NULL, sigma = 0.1, nclust = NULL,
                           max.iter = 10, seed = 0, device = "auto", early.stop = TRUE,
                           .options = NULL, init.centroids = NULL, kmeans.nstart = 1, kmeans.iter = 10) {
  if (is.data.frame(emb)) emb <- as.matrix(emb)
  if (!is.matrix(emb) || !is.numeric(emb)) {
    stop("'emb' must be a numeric matrix (cells x dimensions)", call. = FALSE)
  }
  if (storage.mode(emb) != "double") storage.mode(emb) <- "double"
  if (is.data.frame(batches)) {
    if (ncol(batches) != 1L) stop("'batches' must be one batch variable", call. = FALSE)
    batches <- batches[[1L]]
  }
  if (length(batches) != nrow(emb)) {
    stop(sprintf("'batches' has %d labels for %d cells", length(batches), nrow(emb)), call. = FALSE)
  }
  if (anyNA(batches)) stop("'batches' contains NA", call. = FALSE)
  batches <- droplevels(as.factor(batches))
  for (nm in c("theta", "sigma")) {
    v <- get(nm)
    if (!is.numeric(v) || length(v) != 1L || !is.finite(v)) stop(sprintf("'%s' must be a single number", nm), call. = FALSE)
  }
  if (!is.null(lambda)) .check_positive_number(lambda, "lambda")
  if (!is.null(nclust)) nclust <- .check_count(nclust, "nclust")
  max.iter <- .check_count(max.iter, "max.iter")
  seed <- .check_seed(seed)
  .check_flag(early.stop, "early.stop")
  kmeans.nstart <- .check_count(kmeans.nstart, "kmeans.nstart")
  kmeans.iter <- .check_count(kmeans.iter, "kmeans.iter", min = 0)
  opts <- .harmony_options(.options)
  if (!is.null(init.centroids)) {
    if (!is.matrix(init.centroids) || !is.numeric(init.centroids) || ncol(init.centroids) != ncol(emb)) {
      stop(sprintf("'init.centroids' must be a numeric matrix with %d columns", ncol(emb)), call. = FALSE)
    }
    if (!is.null(nclust) && nrow(init.centroids) != nclust) {
      stop("'init.centroids' must have 'nclust' rows", call. = FALSE)
    }
    storage.mode(init.centroids) <- "double"
  }
  device <- .check_device(device)
  # No CUDA Harmony yet: "auto" means the CPU; an explicit CUDA device reaches the kernel's error.
  dev <- if (identical(attr(device, "requested"), "auto")) "cpu" else as.character(device)
  res <- .rust(rs_harmony(emb, nrow(emb), ncol(emb), as.integer(batches), nlevels(batches),
                          as.numeric(theta), as.numeric(lambda %||% -1), as.numeric(sigma),
                          as.integer(nclust %||% -1L), max.iter, as.integer(opts$max.iter.cluster),
                          as.numeric(opts$epsilon.cluster),
                          if (early.stop) as.numeric(opts$epsilon.harmony) else -Inf,
                          as.numeric(opts$block.size), as.numeric(opts$tau), as.numeric(opts$alpha),
                          as.numeric(opts$batch.prop.cutoff), kmeans.nstart, kmeans.iter, seed, dev,
                          init.centroids))
  dimnames(res$embeddings) <- dimnames(emb)
  res
}

#' Harmony integration of a Seurat reduction
#'
#' Replaces `harmony::RunHarmony()` on a Seurat object: corrects the `dims.use` columns
#' of `reduction` for the batch variable `group.by.vars` with [scfast_harmony()] and
#' stores the result as a `DimReduc` (key `"harmony_"` for the default name, the assay of
#' `reduction`, `stdev` the column standard deviations, as `RunHarmony()`). Unlike
#' `RunHarmony()`, no feature loadings are projected (`project.dim`), since that needs a
#' `scale.data` layer that the scfast pipeline never builds.
#'
#' @param object A Seurat object with the reduction (e.g. after [RunPCAFast()]).
#' @param group.by.vars Name of the meta data column with the batch labels (one variable).
#' @param reduction Reduction to correct (`RunHarmony(reduction.use)`).
#' @param dims.use Columns of the reduction to use; `NULL` is all.
#' @param reduction.save Name of the new reduction; its key is `Key(reduction.save)`.
#' @param theta,lambda,sigma,nclust,max.iter,seed,device,early.stop,.options See
#'   [scfast_harmony()]; defaults as `RunHarmony()` (harmony 2.x: `lambda = NULL`).
#' @param verbose Print a summary line.
#' @return The Seurat object with the reduction added; `Misc(object[[reduction.save]])$scfast`
#'   records the parameters, `iters`, `converged`, the objective and the run time.
#' @export
RunHarmonyFast <- function(object, group.by.vars, reduction = "pca", dims.use = NULL,
                           reduction.save = "harmony", theta = 2, lambda = NULL, sigma = 0.1,
                           nclust = NULL, max.iter = 10, seed = 0, device = "auto",
                           verbose = FALSE, early.stop = TRUE, .options = NULL) {
  .check_seurat(object)
  .check_string(reduction, "reduction")
  .check_string(reduction.save, "reduction.save")
  .check_flag(verbose, "verbose")
  if (!is.character(group.by.vars) || length(group.by.vars) != 1L || is.na(group.by.vars)) {
    stop("'group.by.vars' must be the name of one meta data column", call. = FALSE)
  }
  if (!reduction %in% SeuratObject::Reductions(object)) {
    stop(sprintf("reduction '%s' not found; run RunPCAFast() or RunPCA() first", reduction), call. = FALSE)
  }
  if (!group.by.vars %in% colnames(object[[]])) {
    stop(sprintf("'%s' is not a meta data column", group.by.vars), call. = FALSE)
  }
  reduc <- object[[reduction]]
  emb <- SeuratObject::Embeddings(reduc)
  dims.use <- dims.use %||% seq_len(ncol(emb))
  if (!.is_whole(dims.use) || any(dims.use < 1) || anyDuplicated(dims.use)) {
    stop("'dims.use' must be distinct positive whole numbers", call. = FALSE)
  }
  if (max(dims.use) > ncol(emb)) stop("More dimensions specified in dims.use than have been computed", call. = FALSE)
  if (length(dims.use) < 2L) stop("only specified one dimension in dims.use", call. = FALSE)
  batches <- object[[group.by.vars]][rownames(emb), 1L]
  run <- .timed(scfast_harmony(emb[, dims.use, drop = FALSE], batches, theta = theta, lambda = lambda,
                               sigma = sigma, nclust = nclust, max.iter = max.iter, seed = seed,
                               device = device, early.stop = early.stop, .options = .options))
  res <- run$value
  key <- SeuratObject::Key(reduction.save, quiet = TRUE)
  emb.h <- res$embeddings
  colnames(emb.h) <- paste0(key, seq_len(ncol(emb.h)))
  red <- SeuratObject::CreateDimReducObject(
    embeddings = emb.h,
    stdev = as.numeric(apply(emb.h, 2, stats::sd)),
    assay = SeuratObject::DefaultAssay(reduc),
    key = key,
    misc = list(scfast = list(
      kernel = "harmony", engine = scfast_info()$version, reduction = reduction,
      dims.use = dims.use, group.by.vars = group.by.vars,
      params = c(list(theta = theta, lambda = lambda, sigma = sigma, nclust = res$nclust,
                      max.iter = max.iter, seed = seed, early.stop = early.stop),
                 .harmony_options(.options)),
      iters = res$iters, converged = res$converged, objective = res$objective,
      objective_kmeans = res$objective_kmeans, kmeans_rounds = res$kmeans_rounds,
      seconds = run$seconds, device = res$device
    ))
  )
  if (verbose) {
    message(sprintf("scfast Harmony: %d cells x %d dims, %d batches, %d clusters, %d rounds (%s) in %.2f s on %s",
                    nrow(emb.h), ncol(emb.h), length(unique(batches)), res$nclust, res$iters,
                    if (res$converged) "converged" else "max.iter reached", run$seconds, res$device))
  }
  object[[reduction.save]] <- red
  object <- SeuratObject::LogSeuratCommand(object = object)
  object
}

# ------------------------------------------------------------------ preprocessing (Phase 9)

.check_positive_number <- function(x, name) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x <= 0) {
    stop(sprintf("'%s' must be a single positive number", name), call. = FALSE)
  }
  invisible(as.numeric(x))
}

#' Fused log-normalisation and vst variable-feature statistics of a counts matrix
#'
#' One streaming pass over the raw counts computes what `NormalizeData()`,
#' `FindVariableFeatures(selection.method = "vst")` and the per-gene statistics of
#' `ScaleData()` compute: per-cell library sizes, `log1p(counts / library_size *
#' scale.factor)`, the vst mean / variance / loess-expected variance / clipped standardised
#' variance of every gene, the top `n_hvg` genes by standardised variance, and the mean and
#' standard deviation of their log-normalised values (what the implicit PCA scales with).
#'
#' The `@p`, `@i` and `@x` arrays of `counts` (genes x cells) are read in place; the
#' returned `data` matrix shares `@p` and `@i` with `counts`.
#'
#' The loess (local quadratic, tricube, span `loess.span`) is R's default
#' `loess(log10(variance) ~ log10(mean))` (`surface = "interpolate"`) that Seurat calls,
#' so the expected variances match Seurat's to ~1e-11 (log10).
#'
#' @param counts A genes x cells `dgCMatrix` of raw counts (other matrix classes are
#'   converted, which copies).
#' @param n_hvg Number of variable features (Seurat `nfeatures`).
#' @param scale.factor `NormalizeData(scale.factor)`.
#' @param loess.span `FindVariableFeatures(loess.span)`.
#' @param clip.max Clip for the standardised values; `NULL` (default) means
#'   `sqrt(n_cells)`, as Seurat's `clip.max = "auto"`.
#' @return A list: `data` (genes x cells `dgCMatrix` of log-normalised values),
#'   `library_size` (per cell, named), `hvg` (feature names, most variable first),
#'   `hvg_stats` (data frame with one row per gene: `mean`, `variance`, `variance.expected`,
#'   `variance.standardized`, `variable`, `rank`), `hvg_mean` and `hvg_sd` (named by HVG),
#'   `clip` and `seconds`.
#' @export
scfast_preprocess <- function(counts, n_hvg = 2000, scale.factor = 1e4, loess.span = 0.3, clip.max = NULL) {
  counts <- .as_dgc(counts, "counts")
  n_hvg <- .check_count(n_hvg, "n_hvg")
  scale.factor <- .check_positive_number(scale.factor, "scale.factor")
  loess.span <- .check_positive_number(loess.span, "loess.span")
  clip <- if (is.null(clip.max)) -1 else .check_positive_number(clip.max, "clip.max")
  if (nrow(counts) < 1L || ncol(counts) < 1L) stop("'counts' is empty", call. = FALSE)
  if (n_hvg > nrow(counts)) {
    warning(sprintf("n_hvg (%d) reduced to the number of features (%d)", n_hvg, nrow(counts)), call. = FALSE)
    n_hvg <- nrow(counts)
  }
  run <- .timed(.preprocess_native(counts, n_hvg, scale.factor, loess.span, clip))
  .preprocess_result(run$value, counts, run$seconds)
}

.preprocess_native <- function(counts, n_hvg, scale.factor, loess.span, clip) {
  .rust(rs_preprocess_csc(counts@p, counts@i, counts@x, nrow(counts), ncol(counts), as.integer(n_hvg),
                          as.numeric(scale.factor), as.numeric(loess.span), as.numeric(clip)))
}

# Native result -> the documented list (data dgCMatrix sharing @p / @i with the counts).
.preprocess_result <- function(res, counts, seconds) {
  features <- rownames(counts) %||% paste0("feature", seq_len(nrow(counts)))
  cells <- colnames(counts)
  data <- methods::new("dgCMatrix", i = counts@i, p = counts@p, x = res$data, Dim = dim(counts),
                       Dimnames = list(rownames(counts), cells))
  rank <- rep(NA_integer_, nrow(counts))
  rank[res$hvg] <- seq_along(res$hvg)
  stats <- data.frame(mean = res$mean, variance = res$variance, variance.expected = res$variance_expected,
                      variance.standardized = res$variance_standardized, variable = !is.na(rank), rank = rank,
                      row.names = features, stringsAsFactors = FALSE)
  lib <- res$library_size
  names(lib) <- cells
  hvg <- features[res$hvg]
  hvg_mean <- res$hvg_mean
  hvg_sd <- res$hvg_sd
  names(hvg_mean) <- names(hvg_sd) <- hvg
  list(data = data, library_size = lib, hvg = hvg, hvg_stats = stats, hvg_mean = hvg_mean, hvg_sd = hvg_sd,
       clip = res$clip, seconds = seconds)
}

# The assay's "counts" layer exactly as stored, with feature and cell names.
.counts_layer <- function(object, assay) {
  a <- object[[assay]]
  if (inherits(a, "Assay5")) {
    layers <- SeuratObject::Layers(a)
    if (!"counts" %in% layers) {
      if (any(startsWith(layers, "counts."))) {
        stop(sprintf("assay '%s' has split counts layers (%s); run JoinLayers() first",
                     assay, paste(grep("^counts\\.", layers, value = TRUE), collapse = ", ")), call. = FALSE)
      }
      stop(sprintf("assay '%s' has no 'counts' layer", assay), call. = FALSE)
    }
    m <- methods::slot(a, "layers")[["counts"]]
    features <- SeuratObject::Features(a, layer = "counts")
    cells <- SeuratObject::Cells(a, layer = "counts")
  } else if (inherits(a, "Assay")) {
    m <- methods::slot(a, "counts")
    features <- rownames(m)
    cells <- colnames(m)
  } else {
    stop(sprintf("assay '%s' has unsupported class '%s'", assay, class(a)[1]), call. = FALSE)
  }
  if (!identical(dim(m), c(length(features), length(cells)))) {
    stop(sprintf("assay '%s': the counts layer is not stored as features x cells", assay), call. = FALSE)
  }
  if (length(cells) != ncol(object) || !setequal(cells, SeuratObject::Cells(object))) {
    stop(sprintf("assay '%s': the counts layer does not cover all cells of the object", assay), call. = FALSE)
  }
  m <- .as_dgc(m, "counts layer")
  if (is.null(dimnames(m)) || is.null(rownames(m))) dimnames(m) <- list(features, cells)
  list(m = m, features = features, cells = cells)
}

#' Normalise, find variable features and compute scaling statistics in one pass
#'
#' Replaces `NormalizeData(normalization.method = "LogNormalize")` +
#' `FindVariableFeatures(selection.method = "vst")` (+ the per-gene statistics of
#' `ScaleData()`, which `RunPCAFast()` applies implicitly) with one streaming pass over the
#' assay's counts (see [scfast_preprocess()]). Writes the `data` layer, the variable
#' features (and the `vf_vst_counts_*` feature meta data that `HVFInfo()` and
#' `VariableFeaturePlot()` read: pass `selection.method = "vst"` to them, as no
#' `FindVariableFeatures` command is logged), and `Misc(object, "scfast")$hvg_stats`.
#'
#' @param object A Seurat object with raw counts.
#' @param n_hvg Number of variable features (Seurat `nfeatures`).
#' @param assay Assay to use; default `DefaultAssay(object)`.
#' @param scale.factor `NormalizeData(scale.factor)`.
#' @param loess.span `FindVariableFeatures(loess.span)`.
#' @param clip.max `NULL` (`sqrt(n_cells)`, Seurat's `"auto"`) or a positive number.
#' @param verbose Print a progress message.
#' @return The Seurat object with the `data` layer, variable features and statistics added.
#' @export
PreprocessFast <- function(object, n_hvg = 2000, assay = NULL, scale.factor = 1e4, loess.span = 0.3,
                           clip.max = NULL, verbose = TRUE) {
  .check_seurat(object)
  assay <- assay %||% SeuratObject::DefaultAssay(object)
  .check_string(assay, "assay")
  if (!assay %in% SeuratObject::Assays(object)) {
    stop(sprintf("assay '%s' not found in the object", assay), call. = FALSE)
  }
  n_hvg <- .check_count(n_hvg, "n_hvg")
  scale.factor <- .check_positive_number(scale.factor, "scale.factor")
  loess.span <- .check_positive_number(loess.span, "loess.span")
  clip <- if (is.null(clip.max)) -1 else .check_positive_number(clip.max, "clip.max")
  .check_flag(verbose, "verbose")

  layer <- .counts_layer(object, assay)
  n_feat <- nrow(layer$m)
  if (n_hvg > n_feat) {
    warning(sprintf("n_hvg (%d) reduced to the number of features (%d)", n_hvg, n_feat), call. = FALSE)
    n_hvg <- n_feat
  }
  run <- .timed(.preprocess_native(layer$m, n_hvg, scale.factor, loess.span, clip))
  res <- .preprocess_result(run$value, layer$m, run$seconds)

  a <- object[[assay]]
  v5 <- inherits(a, "Assay5")
  SeuratObject::LayerData(a, layer = "data") <- res$data
  # Feature meta data in Seurat's own column names, so HVFInfo() / VariableFeaturePlot() work.
  stats <- res$hvg_stats
  prefix <- if (v5) "vf_vst_counts_" else "vst."
  cols <- paste0(prefix, c("mean", "variance", "variance.expected", "variance.standardized", "variable", "rank"))
  if (v5) {
    a[["var.features"]] <- NULL
    a[["var.features.rank"]] <- NULL
  }
  for (k in seq_along(cols)) {
    vals <- stats[[k]]
    names(vals) <- rownames(stats)
    a[[cols[k]]] <- vals
  }
  SeuratObject::VariableFeatures(a) <- res$hvg
  object[[assay]] <- a
  misc <- SeuratObject::Misc(object, slot = "scfast") %||% list()
  misc$hvg_stats <- stats
  misc$preprocess <- list(kernel = "preprocess", engine = scfast_info()$version, assay = assay,
                          n_hvg = n_hvg, scale.factor = scale.factor, loess.span = loess.span, clip = res$clip,
                          seconds = run$seconds, library_size = res$library_size,
                          hvg_mean = res$hvg_mean, hvg_sd = res$hvg_sd)
  SeuratObject::Misc(object, slot = "scfast") <- misc
  if (verbose) {
    message(sprintf("scfast preprocess: %d cells x %d features, %d variable features in %.2f s (one pass: normalise + vst + scaling stats)",
                    length(layer$cells), n_feat, length(res$hvg), run$seconds))
  }
  object <- SeuratObject::LogSeuratCommand(object = object)
  object
}
