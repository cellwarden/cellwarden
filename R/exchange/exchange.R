# cellwarden exchange format (Seurat <-> AnnData), R side.
#
# Implements docs/exchange-spec.md v1.0 with hdf5r (no anndataR needed):
#
#   sc_counts_checksum(m)                 counts fingerprint (spec section 6)
#   write_exchange(obj, path, mode, ...)  Seurat -> .h5ad  (full / slim)
#   read_exchange(path)                   .h5ad -> plain R list (Seurat orientation)
#   read_exchange_seurat(path, assay)     full-mode .h5ad -> Seurat v5 object
#   validate_exchange(path)               problems against the spec (spec section 8)
#   merge_exchange_results(obj, path,...) results .h5ad -> Seurat (spec section 7)
#
# Usage: source("R/exchange/exchange.R"). Needs Matrix, hdf5r, digest and
# SeuratObject/Seurat (v5 APIs only: LayerData, VariableFeatures, Embeddings,
# CreateDimReducObject, as.Graph, Idents).
#
# Orientation: AnnData is cells x genes, Seurat genes x cells. A genes x cells
# dgCMatrix (CSC) has exactly the arrays of the cells x genes CSR matrix, so
# sparse matrices are written and read without densifying or transposing.
#
# Checksum canonical form (agreed with python/cellwarden/exchange.py): cells x genes
# CSR, duplicates summed, column indices sorted within rows, explicit zeros
# removed; then SHA-256 over shape (n_cells, n_genes) as int64 LE, indptr and
# indices as int64 LE, data as float64 LE.

suppressPackageStartupMessages({
  library(Matrix)
  library(hdf5r)
})

.SCX_SCHEMA_VERSION <- "1.0"
.SCX_SUPPORTED_MAJOR <- 1L
.SCX_MODES <- c("full", "slim", "results")
.SCX_COMMON_KEYS <- c("schema_version", "mode", "run_id", "stage", "writer",
                      "created_utc", "config_hash", "n_cells", "n_genes",
                      "cells_sha256", "genes_sha256", "packages")
.SCX_MODE_KEYS <- list(full = "counts_checksum",
                       slim = "source_counts_checksum",
                       results = c("source_counts_checksum", "cluster_key"))
.SCX_STRING_KEYS <- c("schema_version", "mode", "run_id", "stage", "writer",
                      "created_utc", "config_hash", "cells_sha256", "genes_sha256",
                      "counts_checksum", "source_counts_checksum", "cluster_key")
.SCX_INT32_MAX <- 2147483647

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

.scx_stop <- function(...) stop(paste0("[exchange] ", ...), call. = FALSE)
.scx_warn <- function(...) warning(paste0("[exchange] ", ...), call. = FALSE)

# integer64 (bit64, used by hdf5r for int64 that does not fit an R integer) -> double
.scx_num <- function(x) {
  if (inherits(x, "integer64")) {
    if (!requireNamespace("bit64", quietly = TRUE)) .scx_stop("int64 values need the bit64 package")
    return(bit64::as.double.integer64(x))
  }
  x
}

# A genes x cells (or any) matrix as a valid dgCMatrix: sorted row indices,
# no duplicates. Copies only when needed.
.scx_as_dgc <- function(m) {
  if (!inherits(m, "dgCMatrix")) {
    if (!is(m, "Matrix")) m <- Matrix::Matrix(as.matrix(m), sparse = TRUE)
    m <- as(as(as(m, "CsparseMatrix"), "generalMatrix"), "dMatrix")
  }
  if (!isTRUE(validObject(m, test = TRUE))) {
    # unsorted / duplicated indices (e.g. built with new()): go through triplets
    tm <- new("dgTMatrix", i = m@i, j = rep.int(seq_len(ncol(m)) - 1L, diff(m@p)),
              x = m@x, Dim = m@Dim, Dimnames = m@Dimnames)
    m <- as(tm, "CsparseMatrix")
  }
  m
}

# Non-negative integers < 2^53 as little-endian int64 bytes.
.scx_le_int64 <- function(v) {
  v <- as.double(v)
  if (length(v) == 0) return(raw(0))
  if (any(v < 0 | v >= 2^53 | v != floor(v))) .scx_stop("int64 encoder: value out of range")
  hi <- floor(v / 4294967296)
  lo <- v - hi * 4294967296
  big <- lo >= 2147483648
  lo[big] <- lo[big] - 4294967296
  writeBin(as.integer(rbind(lo, hi)), raw(), size = 4L, endian = "little")
}

.scx_sha256_raw <- function(r) digest::digest(r, algo = "sha256", serialize = FALSE)

# Declare or convert strings to UTF-8 without mangling. enc2utf8() on strings marked
# "unknown" in a non-UTF-8 locale (e.g. LC_CTYPE=C under a bare Rscript) escapes
# bytes > 127 as literal "<c3><ab>" text; bytes that are already valid UTF-8 only
# need their encoding declared.
.scx_utf8 <- function(x) {
  x <- as.character(x)
  ok <- !is.na(x) & validUTF8(x)
  if (any(ok)) Encoding(x[ok]) <- "UTF-8"
  if (any(!ok & !is.na(x))) x[!ok & !is.na(x)] <- enc2utf8(x[!ok & !is.na(x)])
  x
}

.scx_names_sha256 <- function(x) {
  s <- .scx_utf8(paste(as.character(x), collapse = "\n"))
  .scx_sha256_raw(charToRaw(s))
}

.scx_utc_now <- function() format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")

.scx_packages <- function() {
  v <- function(p) tryCatch(as.character(utils::packageVersion(p)), error = function(e) "not-installed")
  list(R = paste(R.version$major, R.version$minor, sep = "."),
       Seurat = v("Seurat"), SeuratObject = v("SeuratObject"),
       hdf5r = v("hdf5r"), Matrix = v("Matrix"))
}

# --------------------------------------------------------------------------
# Counts checksum (spec section 6)
# --------------------------------------------------------------------------

#' Counts checksum of a Seurat-orientation (genes x cells) matrix.
#'
#' Hex SHA-256 of the cells x genes CSR form (duplicates summed, indices
#' sorted, explicit zeros dropped). Accepts dgCMatrix, other Matrix classes or
#' a base matrix. Streams through a temporary file so memory stays bounded.
sc_counts_checksum <- function(m) {
  m <- Matrix::drop0(.scx_as_dgc(m))
  m <- .scx_as_dgc(m)
  n_cells <- ncol(m); n_genes <- nrow(m)
  tf <- tempfile(fileext = ".bin")
  con <- file(tf, open = "wb")
  on.exit({ try(close(con), silent = TRUE); unlink(tf) }, add = TRUE)
  chunk <- 4194304L
  stream <- function(x, fun) {
    n <- length(x); s <- 1L
    while (s <= n) {
      e <- min(n, s + chunk - 1L)
      writeBin(fun(x[s:e]), con)
      s <- e + 1L
    }
  }
  writeBin(.scx_le_int64(c(n_cells, n_genes)), con)
  stream(m@p, .scx_le_int64)
  # indices are < 2^31: low word = value, high word = 0
  stream(m@i, function(x) writeBin(as.integer(rbind(x, 0L)), raw(), size = 4L, endian = "little"))
  stream(m@x, function(x) writeBin(as.double(x), raw(), size = 8L, endian = "little"))
  close(con)
  digest::digest(file = tf, algo = "sha256")
}

# --------------------------------------------------------------------------
# HDF5 writing primitives (AnnData on-disk spec)
# --------------------------------------------------------------------------

.scx_str_type <- function() {
  st <- H5T_STRING$new(size = Inf)
  st$set_cset(h5const$H5T_CSET_UTF8)
  st
}

.scx_attr_str <- function(node, name, value) {
  node$create_attr(name, .scx_utf8(value), dtype = .scx_str_type(),
                   space = H5S$new("scalar"))
  invisible(NULL)
}

.scx_attr_str_vec <- function(node, name, values) {
  values <- .scx_utf8(values)
  if (length(values) == 0) {
    node$create_attr(name, dtype = .scx_str_type(), space = H5S$new(dims = 0, maxdims = 0))
  } else {
    node$create_attr(name, values, dtype = .scx_str_type(),
                     space = H5S$new(dims = length(values), maxdims = length(values)))
  }
  invisible(NULL)
}

.scx_enc <- function(node, type, version) {
  .scx_attr_str(node, "encoding-type", type)
  .scx_attr_str(node, "encoding-version", version)
  invisible(node)
}

.scx_dataset <- function(parent, name, x, dtype) {
  n <- length(x)
  if (n == 0) {
    d <- parent$create_dataset(name, dtype = dtype, space = H5S$new(dims = 0, maxdims = 0),
                               chunk_dims = NULL)
  } else if (is.null(dim(x)) && n < 4096) {
    d <- parent$create_dataset(name, x, dtype = dtype, chunk_dims = NULL)
  } else {
    d <- parent$create_dataset(name, x, dtype = dtype, chunk_dims = "auto", gzip_level = 4)
  }
  d
}

.scx_write_array <- function(parent, name, x, dtype) {
  d <- .scx_dataset(parent, name, x, dtype)
  .scx_enc(d, "array", "0.2.0")
  invisible(d)
}

.scx_write_bool_array <- function(parent, name, x) {
  x <- as.logical(x)
  if (anyNA(x)) .scx_stop("bool array '", name, "' contains NA")
  .scx_write_array(parent, name, x, H5T_LOGICAL$new(include_NA = FALSE))
}

.scx_write_string_array <- function(parent, name, x) {
  x <- .scx_utf8(x)
  d <- .scx_dataset(parent, name, x, .scx_str_type())
  .scx_enc(d, "string-array", "0.2.0")
  invisible(d)
}

.scx_code_type <- function(n_levels) {
  if (n_levels < 128) h5types$H5T_NATIVE_INT8
  else if (n_levels < 32768) h5types$H5T_NATIVE_INT16
  else h5types$H5T_NATIVE_INT32
}

.scx_write_categorical <- function(parent, name, f) {
  g <- parent$create_group(name)
  .scx_enc(g, "categorical", "0.2.0")
  g$create_attr("ordered", is.ordered(f), dtype = H5T_LOGICAL$new(include_NA = FALSE),
                space = H5S$new("scalar"))
  codes <- as.integer(f) - 1L
  codes[is.na(codes)] <- -1L
  .scx_write_array(g, "codes", codes, .scx_code_type(nlevels(f)))
  .scx_write_string_array(g, "categories", levels(f))
  invisible(g)
}

.scx_write_nullable <- function(parent, name, values, type, dtype) {
  g <- parent$create_group(name)
  .scx_enc(g, type, "0.1.0")
  mask <- is.na(values)
  values[mask] <- if (is.logical(values)) FALSE else 0L
  .scx_write_array(g, "values", values, dtype)
  .scx_write_bool_array(g, "mask", mask)
  invisible(g)
}

# One data.frame column, following spec section 3's type mapping.
.scx_write_column <- function(parent, name, x, what) {
  if (is.factor(x)) {
    lv <- levels(x)
    if (anyNA(lv) || anyDuplicated(lv)) .scx_stop(what, " column '", name, "': factor levels must be unique and non-NA")
    return(.scx_write_categorical(parent, name, x))
  }
  if (is.logical(x)) {
    if (anyNA(x)) {
      .scx_warn(what, " column '", name, "' is logical with NA; writing it as nullable-boolean")
      return(.scx_write_nullable(parent, name, x, "nullable-boolean", H5T_LOGICAL$new(include_NA = FALSE)))
    }
    return(.scx_write_bool_array(parent, name, x))
  }
  if (is.integer(x)) {
    if (anyNA(x)) {
      .scx_warn(what, " column '", name, "' is integer with NA; writing it as nullable-integer")
      return(.scx_write_nullable(parent, name, x, "nullable-integer", h5types$H5T_NATIVE_INT32))
    }
    return(.scx_write_array(parent, name, x, h5types$H5T_NATIVE_INT32))
  }
  if (is.numeric(x)) return(.scx_write_array(parent, name, as.double(x), h5types$H5T_NATIVE_DOUBLE))
  if (!is.character(x)) {
    .scx_warn(what, " column '", name, "' has class '", class(x)[1], "'; writing it as strings")
    x <- as.character(x)
  }
  if (anyNA(x)) {
    .scx_warn(what, " column '", name, "' is a string column with missing values; ",
              "writing it as a categorical (missing = code -1)")
    lv <- sort(unique(x[!is.na(x)]), method = "radix")
    return(.scx_write_categorical(parent, name, factor(x, levels = lv)))
  }
  .scx_write_string_array(parent, name, x)
}

.scx_write_dataframe <- function(parent, name, df, index, what) {
  g <- parent$create_group(name)
  .scx_enc(g, "dataframe", "0.2.0")
  .scx_attr_str(g, "_index", "_index")
  cols <- colnames(df)
  if (any(cols == "_index")) .scx_stop(what, " has a column named '_index'")
  if (anyDuplicated(cols)) .scx_stop(what, " has duplicated column names")
  .scx_attr_str_vec(g, "column-order", cols)
  .scx_write_string_array(g, "_index", index)
  for (cn in cols) .scx_write_column(g, cn, df[[cn]], what)
  invisible(g)
}

# CSR from arrays: indptr / indices / data. Index arrays int32 (R vectors are
# < 2^31 long); data dtype chosen by caller.
.scx_write_csr <- function(parent, name, indptr, indices, data, shape, data_dtype) {
  g <- parent$create_group(name)
  .scx_enc(g, "csr_matrix", "0.1.0")
  g$create_attr("shape", as.double(shape), dtype = h5types$H5T_NATIVE_INT64,
                space = H5S$new(dims = 2, maxdims = 2))
  .scx_dataset(g, "data", data, data_dtype)
  .scx_dataset(g, "indices", as.integer(indices), h5types$H5T_NATIVE_INT32)
  ip_type <- if (length(indptr) && max(indptr) > .SCX_INT32_MAX) h5types$H5T_NATIVE_INT64 else h5types$H5T_NATIVE_INT32
  .scx_dataset(g, "indptr", indptr, ip_type)
  invisible(g)
}

# genes x cells dgCMatrix -> cells x genes CSR (same arrays)
.scx_write_seurat_sparse <- function(parent, name, m, data_dtype, data = m@x) {
  .scx_write_csr(parent, name, m@p, m@i, data, c(ncol(m), nrow(m)), data_dtype)
}

# cells x dims (or genes x dims) R matrix -> dense float32 array of that shape
.scx_write_dense2d <- function(parent, name, mat) {
  mat <- as.matrix(mat)
  storage.mode(mat) <- "double"
  dimnames(mat) <- NULL
  .scx_write_array(parent, name, t(mat), h5types$H5T_NATIVE_FLOAT)
}

.scx_empty_dict <- function(parent, name) .scx_enc(parent$create_group(name), "dict", "0.1.0")

# uns values: named list -> dict, scalar string -> string, string vector ->
# string-array, numeric/integer/logical scalar -> numeric-scalar, vectors -> array
.scx_write_uns <- function(parent, name, x) {
  if (is.list(x)) {
    g <- .scx_empty_dict(parent, name)
    nm <- names(x)
    if (length(x) && (is.null(nm) || any(nm == "") || anyDuplicated(nm))) .scx_stop("uns '", name, "' must be a uniquely named list")
    for (k in nm) .scx_write_uns(g, k, x[[k]])
    return(invisible(g))
  }
  if (is.factor(x)) x <- as.character(x)
  if (is.character(x)) {
    if (length(x) == 1 && is.null(dim(x))) {
      d <- parent$create_dataset(name, .scx_utf8(x), dtype = .scx_str_type(),
                                 space = H5S$new("scalar"), chunk_dims = NULL)
      return(invisible(.scx_enc(d, "string", "0.2.0")))
    }
    return(.scx_write_string_array(parent, name, x))
  }
  dt <- if (is.logical(x)) H5T_LOGICAL$new(include_NA = FALSE)
        else if (is.integer(x)) h5types$H5T_NATIVE_INT64
        else h5types$H5T_NATIVE_DOUBLE
  if (length(x) == 1 && is.null(dim(x))) {
    d <- parent$create_dataset(name, x, dtype = dt, space = H5S$new("scalar"), chunk_dims = NULL)
    return(invisible(.scx_enc(d, "numeric-scalar", "0.2.0")))
  }
  if (!is.null(dim(x))) x <- t(x)
  .scx_write_array(parent, name, x, dt)
}

# --------------------------------------------------------------------------
# Seurat accessors
# --------------------------------------------------------------------------

.scx_layer <- function(obj, assay, layer) {
  a <- obj[[assay]]
  lys <- SeuratObject::Layers(a)
  if (!(layer %in% lys)) {
    split <- grep(paste0("^", layer, "\\."), lys, value = TRUE)
    if (length(split)) .scx_stop("assay '", assay, "' has split '", layer, "' layers (",
                                 paste(split, collapse = ", "), "); run JoinLayers() first")
    return(NULL)
  }
  m <- SeuratObject::LayerData(obj, assay = assay, layer = layer)
  if (is.null(m) || length(m) == 0 || any(dim(m) == 0)) return(NULL)
  m
}

.scx_feature_meta <- function(obj, assay) {
  fm <- tryCatch(obj[[assay]][[]], error = function(e) NULL)
  if (is.null(fm) || nrow(fm) != nrow(obj[[assay]])) return(NULL)
  fm
}

.scx_reduction_key <- function(name) {
  switch(name,
         pca = "PC_", harmony = "harmony_", umap = "UMAP_", {
           k <- gsub("[^A-Za-z0-9]", "", toupper(name))
           if (!nzchar(k) || !grepl("^[A-Za-z]", k)) k <- paste0("R", k)
           paste0(k, "_")
         })
}

.scx_obsm_name <- function(reduction) paste0("X_", reduction)

# --------------------------------------------------------------------------
# write_exchange
# --------------------------------------------------------------------------

#' Write a Seurat object as an exchange file (spec v1.0).
#'
#' mode "full": X = data layer (all genes, float32; counts if no data layer),
#'   layers["counts"] (int32/int64), varm["PCs"], obsm, obsp, counts_checksum.
#' mode "slim": X = data layer for VariableFeatures() only, no layers or varm,
#'   source_counts_checksum = checksum of the counts layer.
#' reductions: NULL = all, character() = none, or a vector of reduction names.
#' graphs: TRUE writes <assay>_snn -> obsp["connectivities"] and
#'   <assay>_nn -> obsp["distances"] when they exist.
write_exchange <- function(obj, path, mode = c("full", "slim"), run_id = "none",
                           stage = "unknown", config_hash = "none", assay = "RNA",
                           reductions = NULL, graphs = TRUE) {
  mode <- match.arg(mode)
  if (!inherits(obj, "Seurat")) .scx_stop("obj must be a Seurat object")
  if (!(assay %in% SeuratObject::Assays(obj))) .scx_stop("assay '", assay, "' not in object")
  cells <- colnames(obj)
  genes <- rownames(obj[[assay]])
  if (anyDuplicated(cells)) .scx_stop("cell barcodes are not unique")
  if (anyDuplicated(genes)) .scx_stop("feature names are not unique")
  n_cells <- length(cells)
  notes <- character()

  counts <- .scx_layer(obj, assay, "counts")
  data <- .scx_layer(obj, assay, "data")
  if (!is.null(counts)) counts <- .scx_as_dgc(counts)
  if (!is.null(data)) data <- .scx_as_dgc(data)
  for (m in list(counts, data)) {
    if (!is.null(m) && (!identical(colnames(m), cells) || !identical(rownames(m), genes)))
      .scx_stop("layer dimnames do not match the object's cells/features")
  }
  if (is.null(counts)) .scx_stop("assay '", assay, "' has no counts layer (needed for the counts checksum)")
  checksum <- sc_counts_checksum(counts)

  fm <- .scx_feature_meta(obj, assay)
  sym <- if (!is.null(fm) && "gene_symbol_original" %in% colnames(fm)) as.character(fm$gene_symbol_original) else genes
  sym[is.na(sym) | !nzchar(sym)] <- genes[is.na(sym) | !nzchar(sym)]
  gid <- if (!is.null(fm) && "gene_id" %in% colnames(fm)) as.character(fm$gene_id) else rep("", length(genes))
  gid[is.na(gid)] <- ""
  hv_names <- SeuratObject::VariableFeatures(obj, assay = assay)
  hv <- genes %in% hv_names
  var <- data.frame(gene_symbol_original = sym, gene_id = gid, highly_variable = hv,
                    stringsAsFactors = FALSE, check.names = FALSE)

  if (mode == "slim") {
    if (!any(hv)) .scx_stop("slim mode needs VariableFeatures(); none are set for assay '", assay, "'")
    if (is.null(data)) .scx_stop("slim mode needs a log-normalised data layer (run NormalizeData)")
    keep <- which(hv)
    X <- data[keep, , drop = FALSE]
    var <- var[keep, , drop = FALSE]
    var_names <- genes[keep]
  } else {
    var_names <- genes
    if (is.null(data)) {
      .scx_warn("no data layer in assay '", assay, "': X holds the raw counts as float32")
      notes <- c(notes, "X holds raw counts (no data layer)")
      X <- counts
    } else {
      X <- data
    }
  }

  # meta.data -> obs
  md <- obj[[]]
  md <- md[match(cells, rownames(md)), , drop = FALSE]

  # reductions -> obsm
  all_red <- SeuratObject::Reductions(obj)
  if (is.null(reductions)) reductions <- all_red
  missing_red <- setdiff(reductions, all_red)
  if (length(missing_red)) .scx_stop("reductions not in object: ", paste(missing_red, collapse = ", "))
  obsm <- list()
  for (r in reductions) {
    e <- SeuratObject::Embeddings(obj, reduction = r)
    if (!all(cells %in% rownames(e))) {
      .scx_warn("reduction '", r, "' does not cover all cells; not written")
      next
    }
    obsm[[.scx_obsm_name(r)]] <- e[cells, , drop = FALSE]
  }

  varm <- list()
  if (mode == "full" && "pca" %in% all_red) {
    L <- tryCatch(SeuratObject::Loadings(obj, reduction = "pca"), error = function(e) NULL)
    if (!is.null(L) && length(L) && nrow(L) > 0) {
      full <- matrix(0, nrow = length(genes), ncol = ncol(L))
      idx <- match(rownames(L), genes)
      ok <- !is.na(idx)
      full[idx[ok], ] <- L[ok, , drop = FALSE]
      varm[["PCs"]] <- full
    }
  }

  obsp <- list()
  if (isTRUE(graphs)) {
    gmap <- c(connectivities = paste0(assay, "_snn"), distances = paste0(assay, "_nn"))
    for (k in names(gmap)) {
      gname <- gmap[[k]]
      if (!(gname %in% SeuratObject::Graphs(obj))) next
      G <- obj[[gname]]
      if (!setequal(rownames(G), cells) || !setequal(colnames(G), cells)) {
        .scx_warn("graph '", gname, "' does not cover exactly the object's cells; not written")
        next
      }
      G <- .scx_as_dgc(as(G[cells, cells], "CsparseMatrix"))
      obsp[[k]] <- .scx_as_dgc(Matrix::t(G))  # CSC of t(G) == CSR of G
    }
  }

  ex <- list(schema_version = .SCX_SCHEMA_VERSION, mode = mode,
             run_id = as.character(run_id), stage = as.character(stage), writer = "R",
             created_utc = .scx_utc_now(), config_hash = as.character(config_hash),
             n_cells = as.integer(n_cells), n_genes = as.integer(length(var_names)),
             cells_sha256 = .scx_names_sha256(cells), genes_sha256 = .scx_names_sha256(var_names))
  if (mode == "full") ex$counts_checksum <- checksum else ex$source_counts_checksum <- checksum
  ex$packages <- .scx_packages()
  if (length(notes)) ex$notes <- paste(notes, collapse = "; ")

  # Write to a temp file next to the target, then rename (no half-written files).
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp-", Sys.getpid())
  if (file.exists(tmp)) unlink(tmp)
  h5 <- H5File$new(tmp, mode = "w")
  ok <- FALSE
  on.exit({ try(h5$close_all(), silent = TRUE); if (!ok) unlink(tmp) }, add = TRUE)

  .scx_enc(h5, "anndata", "0.1.0")
  .scx_write_dataframe(h5, "obs", md, cells, "obs")
  .scx_write_dataframe(h5, "var", var, var_names, "var")
  .scx_write_seurat_sparse(h5, "X", X, h5types$H5T_NATIVE_FLOAT)

  lay <- .scx_empty_dict(h5, "layers")
  if (mode == "full") {
    cm <- Matrix::drop0(counts)
    x <- cm@x
    if (length(x) && any(!is.finite(x) | x != round(x))) .scx_stop("counts layer contains non-integer values")
    fits <- !length(x) || (min(x) >= -.SCX_INT32_MAX - 1 && max(x) <= .SCX_INT32_MAX)
    if (fits) .scx_write_seurat_sparse(lay, "counts", cm, h5types$H5T_NATIVE_INT32, data = as.integer(x))
    else .scx_write_seurat_sparse(lay, "counts", cm, h5types$H5T_NATIVE_INT64, data = x)
  }
  om <- .scx_empty_dict(h5, "obsm")
  for (k in names(obsm)) .scx_write_dense2d(om, k, obsm[[k]])
  vm <- .scx_empty_dict(h5, "varm")
  for (k in names(varm)) .scx_write_dense2d(vm, k, varm[[k]])
  op <- .scx_empty_dict(h5, "obsp")
  for (k in names(obsp)) .scx_write_seurat_sparse(op, k, obsp[[k]], h5types$H5T_NATIVE_FLOAT)
  .scx_empty_dict(h5, "varp")
  un <- .scx_empty_dict(h5, "uns")
  .scx_write_uns(un, "exchange", ex)

  h5$close_all()
  if (!file.rename(tmp, path)) {
    if (!file.copy(tmp, path, overwrite = TRUE)) .scx_stop("could not move ", tmp, " to ", path)
    unlink(tmp)
  }
  ok <- TRUE
  invisible(path)
}

# --------------------------------------------------------------------------
# HDF5 reading primitives
# --------------------------------------------------------------------------

.scx_has <- function(node, path) {
  parts <- strsplit(path, "/", fixed = TRUE)[[1]]
  parts <- parts[nzchar(parts)]
  cur <- node
  for (p in parts) {
    if (!inherits(cur, "H5Group") && !inherits(cur, "H5File")) return(FALSE)
    if (!isTRUE(tryCatch(cur$exists(p), error = function(e) FALSE))) return(FALSE)
    cur <- cur[[p]]
  }
  TRUE
}

.scx_attr <- function(node, name, default = NULL) {
  if (!isTRUE(tryCatch(node$attr_exists(name), error = function(e) FALSE))) return(default)
  a <- node$attr_open(name)
  sp <- a$get_space()
  if (as.character(sp$get_simple_extent_type()) == "H5S_SIMPLE" && length(sp$dims) && prod(sp$dims) == 0) return(character(0))
  v <- .scx_num(a$read())
  if (is.character(v)) Encoding(v) <- "UTF-8"
  v
}

.scx_is_group <- function(node) inherits(node, "H5Group") || inherits(node, "H5File")

.scx_read_dataset <- function(d) {
  dims <- d$dims
  if (length(dims) && prod(dims) == 0) {
    cls <- as.character(d$get_type()$get_class())
    v <- switch(cls, H5T_STRING = character(0), H5T_INTEGER = integer(0),
                H5T_ENUM = logical(0), double(0))
    if (length(dims) == 2) v <- matrix(v, nrow = dims[[1]], ncol = dims[[2]])
    return(v)
  }
  ext <- as.character(d$get_space()$get_simple_extent_type())
  if (ext == "H5S_NULL") return(NULL)            # AnnData encoding-type "null" (Python None)
  v <- if (ext == "H5S_SCALAR") d$read_low_level() else d$read()
  v <- .scx_num(v)
  if (is.character(v)) Encoding(v) <- "UTF-8"
  if (is.factor(v)) v <- as.logical(as.character(v))  # enum other than h5py bool
  v
}

# 2-D dense array in AnnData orientation (rows x cols as h5py sees it)
.scx_read_dense <- function(d) {
  v <- .scx_read_dataset(d)
  if (length(d$dims) == 2) {
    if (is.null(dim(v))) v <- matrix(v, nrow = d$dims[[1]], ncol = d$dims[[2]])
    return(t(v))
  }
  v
}

# Sparse group -> dgCMatrix in AnnData orientation is t() of what we build;
# .scx_read_sparse_t returns the TRANSPOSE of the stored matrix (i.e. Seurat
# orientation for X/layers), without densifying.
.scx_read_sparse_t <- function(g) {
  enc <- .scx_attr(g, "encoding-type")
  shape <- as.numeric(.scx_attr(g, "shape"))
  ip <- as.numeric(.scx_read_dataset(g[["indptr"]]))
  ix <- as.integer(.scx_read_dataset(g[["indices"]]))
  x <- as.double(.scx_read_dataset(g[["data"]]))
  if (identical(enc, "csr_matrix")) {
    # CSR of A (r x c) == CSC of t(A) (c x r)
    dims <- c(shape[[2]], shape[[1]])
    m <- Matrix::sparseMatrix(i = ix, p = ip, x = x, dims = dims, index1 = FALSE, repr = "C")
  } else if (identical(enc, "csc_matrix")) {
    m <- Matrix::sparseMatrix(i = ix, p = ip, x = x, dims = shape, index1 = FALSE, repr = "C")
    m <- Matrix::t(m)
  } else .scx_stop("not a sparse matrix encoding: ", enc)
  .scx_as_dgc(m)
}

# Element shape as AnnData sees it (without loading data).
.scx_elem_shape <- function(node) {
  enc <- .scx_attr(node, "encoding-type", "")
  if (enc %in% c("csr_matrix", "csc_matrix")) return(as.numeric(.scx_attr(node, "shape")))
  if (enc == "categorical") return(as.numeric(node[["codes"]]$dims))
  if (enc %in% c("nullable-integer", "nullable-boolean", "nullable-string-array"))
    return(as.numeric(node[["values"]]$dims))
  if (enc == "dataframe") {
    idx <- .scx_attr(node, "_index", "_index")
    return(as.numeric(node[[idx]]$dims))
  }
  if (!.scx_is_group(node)) return(rev(as.numeric(node$dims)))  # hdf5r dims are R (reversed) order
  NULL
}

.scx_read_nullable <- function(g) {
  v <- .scx_read_dataset(g[["values"]])
  mask <- as.logical(.scx_read_dataset(g[["mask"]]))
  v[mask] <- NA
  v
}

.scx_read_column <- function(node) {
  enc <- .scx_attr(node, "encoding-type", "")
  switch(enc,
    categorical = {
      codes <- as.integer(.scx_read_dataset(node[["codes"]]))
      cats <- as.character(.scx_read_elem(node[["categories"]]))
      ord <- isTRUE(as.logical(.scx_attr(node, "ordered", FALSE)))
      if (any(codes < -1L | codes >= length(cats), na.rm = TRUE))
        stop("categorical codes must be between -1 and length(categories) - 1")
      codes[codes < 0] <- NA
      factor(cats[codes + 1L], levels = cats, ordered = ord)
    },
    "nullable-integer" = , "nullable-boolean" = , "nullable-string-array" = .scx_read_nullable(node),
    .scx_read_elem(node))
}

.scx_read_dataframe <- function(g) {
  idx_name <- .scx_attr(g, "_index", "_index")
  index <- as.character(.scx_read_column(g[[idx_name]]))
  cols <- as.character(.scx_attr(g, "column-order", character(0)))
  out <- lapply(cols, function(cn) {
    v <- .scx_read_column(g[[cn]])
    if (!is.null(dim(v))) .scx_stop("dataframe column '", cn, "' is not 1-D")
    v
  })
  names(out) <- cols
  df <- as.data.frame(out, stringsAsFactors = FALSE, optional = TRUE)
  if (length(cols) == 0) df <- data.frame(row.names = seq_along(index))[, FALSE, drop = FALSE]
  colnames(df) <- cols
  if (anyDuplicated(index)) {
    attr(df, "duplicated_index") <- TRUE
    rownames(df) <- make.unique(index)
  } else {
    rownames(df) <- index
  }
  df
}

# Generic element reader. Matrices come back in AnnData orientation except
# sparse ones, which callers handle explicitly.
.scx_read_elem <- function(node) {
  enc <- .scx_attr(node, "encoding-type", "")
  if (.scx_is_group(node)) {
    switch(enc,
      dataframe = return(.scx_read_dataframe(node)),
      csr_matrix = , csc_matrix = return(Matrix::t(.scx_read_sparse_t(node))),
      categorical = , "nullable-integer" = , "nullable-boolean" = ,
      "nullable-string-array" = return(.scx_read_column(node)),
      {
        out <- list()
        for (k in names(node)) out[k] <- list(.scx_read_elem(node[[k]]))
        return(out)
      })
  }
  if (enc == "array") return(.scx_read_dense(node))
  v <- .scx_read_dataset(node)
  if (length(node$dims) == 2) v <- .scx_read_dense(node)
  v
}

.scx_open <- function(path) {
  if (!file.exists(path)) .scx_stop("file not found: ", path)
  tryCatch(H5File$new(path, mode = "r"), error = function(e) .scx_stop("not a readable HDF5 file: ", path))
}

# --------------------------------------------------------------------------
# read_exchange
# --------------------------------------------------------------------------

#' Read an exchange (.h5ad) file into a plain list, Seurat orientation.
#'
#' Returns list(obs, var, X, layers, obsm, varm, obsp, uns). X and layers are
#' genes x cells (X is NULL when the file has 0 genes); obsm are cells x dims,
#' varm genes x dims, obsp cells x cells dgCMatrix; all carry dimnames.
read_exchange <- function(path) {
  h5 <- .scx_open(path)
  on.exit(h5$close_all(), add = TRUE)
  ex_major <- NULL
  if (.scx_has(h5, "uns/exchange/schema_version")) {
    sv <- as.character(.scx_read_elem(h5[["uns/exchange/schema_version"]]))
    ex_major <- suppressWarnings(as.integer(strsplit(sv, ".", fixed = TRUE)[[1]][1]))
    if (!identical(ex_major, .SCX_SUPPORTED_MAJOR))
      .scx_stop("unsupported schema_version '", sv, "' (supported major: ", .SCX_SUPPORTED_MAJOR, ")")
  }
  obs <- .scx_read_dataframe(h5[["obs"]])
  var <- .scx_read_dataframe(h5[["var"]])
  if (isTRUE(attr(obs, "duplicated_index"))) .scx_stop("obs_names (cell barcodes) are not unique")
  if (isTRUE(attr(var, "duplicated_index"))) .scx_stop("var_names are not unique")
  cells <- rownames(obs); genes <- rownames(var)
  if (nrow(var) == 0) genes <- character(0)

  as_seurat_mat <- function(node) {
    enc <- .scx_attr(node, "encoding-type", "")
    if (enc %in% c("csr_matrix", "csc_matrix")) m <- .scx_read_sparse_t(node)
    else { m <- .scx_read_elem(node); m <- t(as.matrix(m)) }  # dense cells x genes -> genes x cells
    if (!identical(dim(m), c(length(genes), length(cells))))
      .scx_stop("matrix shape ", paste(rev(dim(m)), collapse = "x"), " does not match obs/var")
    dimnames(m) <- list(genes, cells)
    m
  }

  X <- NULL
  if (.scx_has(h5, "X") && length(genes) > 0) X <- as_seurat_mat(h5[["X"]])

  rd_dict <- function(name, fun) {
    out <- list()
    if (.scx_has(h5, name)) {
      g <- h5[[name]]
      for (k in names(g)) out[[k]] <- fun(g[[k]], k)
    }
    out
  }
  layers <- rd_dict("layers", function(n, k) as_seurat_mat(n))
  obsm <- rd_dict("obsm", function(n, k) {
    v <- .scx_read_elem(n)
    if (is.data.frame(v)) return(v)
    v <- as.matrix(v)
    if (nrow(v) != length(cells)) .scx_stop("obsm['", k, "'] rows do not match obs")
    rownames(v) <- cells
    v
  })
  varm <- rd_dict("varm", function(n, k) {
    v <- .scx_read_elem(n)
    if (is.data.frame(v)) return(v)
    v <- as.matrix(v)
    if (nrow(v) != length(genes)) .scx_stop("varm['", k, "'] rows do not match var")
    rownames(v) <- genes
    v
  })
  obsp <- rd_dict("obsp", function(n, k) {
    m <- .scx_as_dgc(Matrix::t(.scx_read_sparse_t(n)))
    if (!identical(dim(m), c(length(cells), length(cells)))) .scx_stop("obsp['", k, "'] is not cells x cells")
    dimnames(m) <- list(cells, cells)
    m
  })
  uns <- if (.scx_has(h5, "uns")) .scx_read_elem(h5[["uns"]]) else list()
  list(obs = obs, var = var, X = X, layers = layers, obsm = obsm, varm = varm,
       obsp = obsp, uns = uns)
}

# --------------------------------------------------------------------------
# Seurat construction helpers
# --------------------------------------------------------------------------

.scx_seurat_names <- function(x) gsub("[_|]", "-", x)

.scx_reduction_name <- function(obsm_key) {
  if (startsWith(obsm_key, "X_")) substring(obsm_key, 3) else obsm_key
}

.scx_make_dimreduc <- function(red_name, emb, assay, loadings = NULL) {
  key <- .scx_reduction_key(red_name)
  emb <- as.matrix(emb)
  storage.mode(emb) <- "double"
  colnames(emb) <- paste0(key, seq_len(ncol(emb)))
  args <- list(embeddings = emb, key = key, assay = assay)
  if (!is.null(loadings)) {
    loadings <- as.matrix(loadings)
    storage.mode(loadings) <- "double"
    colnames(loadings) <- colnames(emb)
    args$loadings <- loadings
  }
  do.call(SeuratObject::CreateDimReducObject, args)
}

.scx_make_graph <- function(m, assay) {
  g <- SeuratObject::as.Graph(m)
  SeuratObject::DefaultAssay(g) <- assay
  g
}

.scx_graph_name <- function(obsp_key, assay) {
  switch(obsp_key, connectivities = paste0(assay, "_snn"), distances = paste0(assay, "_nn"),
         paste0(assay, "_", obsp_key))
}

# --------------------------------------------------------------------------
# read_exchange_seurat
# --------------------------------------------------------------------------

#' Full-mode exchange file -> Seurat v5 object.
#'
#' counts + data layers, meta.data from obs, feature metadata from var
#' (gene_symbol_original, gene_id, ...), VariableFeatures from
#' var$highly_variable, reductions from obsm, graphs from obsp. Feature names
#' are made Seurat-safe ('_' and '|' -> '-'); the originals stay in
#' gene_symbol_original.
read_exchange_seurat <- function(path, assay = "RNA") {
  probs <- validate_exchange(path)
  if (length(probs)) .scx_stop("invalid exchange file ", path, ":\n  ", paste(probs, collapse = "\n  "))
  ex <- read_exchange(path)
  mode <- ex$uns$exchange$mode
  if (!identical(mode, "full")) .scx_stop("read_exchange_seurat needs a full-mode file (got '", mode, "')")
  counts <- ex$layers[["counts"]]
  if (is.null(counts)) .scx_stop("file has no layers['counts']")
  orig <- rownames(ex$var)
  sn <- .scx_seurat_names(orig)
  if (anyDuplicated(sn)) .scx_stop("feature names collide after Seurat renaming ('_'/'|' -> '-'): ",
                                   paste(head(unique(sn[duplicated(sn)]), 5), collapse = ", "))
  cells <- rownames(ex$obs)
  to_dgc <- function(m) { m <- .scx_as_dgc(m); dimnames(m) <- list(sn, cells); m }
  counts <- to_dgc(counts)
  data <- if (is.null(ex$X)) NULL else to_dgc(ex$X)

  a5 <- if (is.null(data)) SeuratObject::CreateAssay5Object(counts = counts)
        else SeuratObject::CreateAssay5Object(counts = counts, data = data)
  obj <- SeuratObject::CreateSeuratObject(a5, assay = assay)
  if (ncol(ex$obs) > 0) obj <- SeuratObject::AddMetaData(obj, metadata = ex$obs)

  fm <- ex$var
  rownames(fm) <- sn
  if (!("gene_symbol_original" %in% colnames(fm))) fm$gene_symbol_original <- orig
  if (!("gene_id" %in% colnames(fm))) fm$gene_id <- rep("", nrow(fm))
  hv <- if ("highly_variable" %in% colnames(fm)) as.logical(fm$highly_variable) else NULL
  fm$highly_variable <- NULL
  obj[[assay]] <- SeuratObject::AddMetaData(obj[[assay]], metadata = fm)
  if (!is.null(hv) && any(hv, na.rm = TRUE))
    SeuratObject::VariableFeatures(obj, assay = assay) <- sn[which(hv)]

  pcs <- ex$varm[["PCs"]]
  for (k in names(ex$obsm)) {
    emb <- ex$obsm[[k]]
    if (is.data.frame(emb)) next
    rn <- .scx_reduction_name(k)
    ld <- NULL
    if (rn == "pca" && !is.null(pcs) && is.matrix(pcs) && ncol(pcs) == ncol(emb)) {
      ld <- pcs; rownames(ld) <- sn
    }
    obj[[rn]] <- .scx_make_dimreduc(rn, emb, assay, ld)
  }
  for (k in names(ex$obsp)) obj[[.scx_graph_name(k, assay)]] <- .scx_make_graph(ex$obsp[[k]], assay)
  obj
}

# --------------------------------------------------------------------------
# validate_exchange (spec section 8)
# --------------------------------------------------------------------------

#' Check a file against spec v1.0. Returns a character vector of problems
#' (empty = valid).
validate_exchange <- function(path) {
  if (!file.exists(path)) return(paste0("file not found: ", path))
  h5 <- tryCatch(H5File$new(path, mode = "r"), error = function(e) NULL)
  if (is.null(h5)) return(paste0("not a readable HDF5 file: ", path))
  on.exit(h5$close_all(), add = TRUE)
  res <- tryCatch(.scx_validate(h5), error = function(e) paste0("could not validate: ", conditionMessage(e)))
  res
}

.scx_validate <- function(h5) {
  p <- character()
  add <- function(...) p <<- c(p, paste0(...))
  if (!identical(.scx_attr(h5, "encoding-type"), "anndata")) add("root group encoding-type is not 'anndata'")
  if (!identical(.scx_attr(h5, "encoding-version"), "0.1.0")) add("root group encoding-version is not '0.1.0'")
  for (k in c("obs", "var")) {
    if (!.scx_has(h5, k)) add("missing /", k)
    else if (!identical(.scx_attr(h5[[k]], "encoding-type"), "dataframe")) add("/", k, " encoding-type is not 'dataframe'")
  }
  if (!.scx_has(h5, "uns/exchange") || !.scx_is_group(h5[["uns/exchange"]]))
    return(c(p, "missing uns['exchange'] (or not a mapping)"))
  if (length(p)) return(p)

  ex <- .scx_read_elem(h5[["uns/exchange"]])
  sv <- ex$schema_version
  if (!is.character(sv) || length(sv) != 1) add("uns['exchange']['schema_version'] missing")
  else {
    major <- suppressWarnings(as.integer(strsplit(sv, ".", fixed = TRUE)[[1]][1]))
    if (!identical(major, .SCX_SUPPORTED_MAJOR))
      add("unsupported schema_version '", sv, "' (supported major: ", .SCX_SUPPORTED_MAJOR, ")")
  }
  mode <- ex$mode
  mode_ok <- is.character(mode) && length(mode) == 1 && mode %in% .SCX_MODES
  if (!mode_ok) add("uns['exchange']['mode'] is '", paste(mode, collapse = ","), "', expected one of full/slim/results")
  if (!is.null(ex$created_utc) && !grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$", ex$created_utc[1]))
    add("uns['exchange']['created_utc'] must be YYYY-MM-DDTHH:MM:SSZ")
  required <- c(.SCX_COMMON_KEYS, if (mode_ok) .SCX_MODE_KEYS[[mode]])
  miss <- setdiff(required, names(ex))
  if (length(miss)) add("uns['exchange'] missing required keys: ", paste(miss, collapse = ", "))
  for (k in intersect(.SCX_STRING_KEYS, names(ex))) {
    v <- ex[[k]]
    if (!is.character(v) || length(v) != 1) add("uns['exchange']['", k, "'] must be a string")
  }
  if (!is.null(ex$writer) && !(identical(ex$writer, "R") || identical(ex$writer, "python")))
    add("uns['exchange']['writer'] is '", ex$writer, "', expected 'R' or 'python'")
  if (!is.null(ex$packages) && !(is.list(ex$packages) && all(vapply(ex$packages, is.character, TRUE))))
    add("uns['exchange']['packages'] must be a mapping string -> string")

  obs_g <- h5[["obs"]]; var_g <- h5[["var"]]
  idx_o <- .scx_attr(obs_g, "_index", "_index"); idx_v <- .scx_attr(var_g, "_index", "_index")
  if (!.scx_has(obs_g, idx_o)) return(c(p, "obs index dataset missing"))
  if (!.scx_has(var_g, idx_v)) return(c(p, "var index dataset missing"))
  cells <- as.character(.scx_read_column(obs_g[[idx_o]]))
  genes <- as.character(.scx_read_column(var_g[[idx_v]]))
  n <- length(cells); g <- length(genes)

  for (k in c("n_cells", "n_genes")) {
    if (is.null(ex[[k]])) next
    v <- ex[[k]]; actual <- if (k == "n_cells") n else g
    if (!is.integer(v) && !(is.numeric(v) && inherits(v, "integer64")))
      add("uns['exchange']['", k, "'] must be an integer")
    else if (length(v) != 1 || is.na(v))
      add("uns['exchange']['", k, "'] must be an integer")
    else if (v != actual) add("uns['exchange']['", k, "'] = ", v, " but file has ", actual)
  }
  if (is.character(ex$cells_sha256) && !identical(ex$cells_sha256, .scx_names_sha256(cells)))
    add("cells_sha256 does not match obs_names")
  if (is.character(ex$genes_sha256) && !identical(ex$genes_sha256, .scx_names_sha256(genes)))
    add("genes_sha256 does not match var_names")
  if (anyDuplicated(cells)) add("obs_names (cell barcodes) are not unique")
  if (anyDuplicated(genes)) add("var_names are not unique")

  for (spec in list(list(obs_g, "obs", n), list(var_g, "var", g))) {
    cols <- as.character(.scx_attr(spec[[1]], "column-order", character(0)))
    for (cn in cols) {
      if (!.scx_has(spec[[1]], cn)) { add(spec[[2]], " column '", cn, "' listed in column-order but missing"); next }
      sh <- .scx_elem_shape(spec[[1]][[cn]])
      if (length(sh) != 1 || sh[[1]] != spec[[3]]) add(spec[[2]], " column '", cn, "' has length ", paste(sh, collapse = "x"), ", expected ", spec[[3]])
    }
  }

  shp <- function(node) .scx_elem_shape(node)
  fmt <- function(s) paste0("(", paste(s, collapse = ", "), ")")
  if (!.scx_has(h5, "X")) add("X is missing")
  else {
    s <- shp(h5[["X"]])
    if (!identical(as.numeric(s), as.numeric(c(n, g)))) add("X shape ", fmt(s), " != ", fmt(c(n, g)))
  }
  if (mode_ok && mode == "results" && g != 0) add("results mode X must have 0 columns, has ", g)
  each <- function(grp, fun) if (.scx_has(h5, grp)) for (k in names(h5[[grp]])) fun(k, h5[[grp]][[k]])
  each("layers", function(k, nd) { s <- shp(nd); if (!identical(as.numeric(s), as.numeric(c(n, g)))) add("layers['", k, "'] shape ", fmt(s), " != ", fmt(c(n, g))) })
  each("obsm", function(k, nd) { s <- shp(nd); if (!length(s) || s[[1]] != n) add("obsm['", k, "'] has ", s[1], " rows, expected ", n) })
  each("varm", function(k, nd) { s <- shp(nd); if (!length(s) || s[[1]] != g) add("varm['", k, "'] has ", s[1], " rows, expected ", g) })
  each("obsp", function(k, nd) { s <- shp(nd); if (!identical(as.numeric(s), as.numeric(c(n, n)))) add("obsp['", k, "'] shape ", fmt(s), " != ", fmt(c(n, n))) })

  if (mode_ok && mode == "full") {
    if (!.scx_has(h5, "layers/counts")) add("full mode file has no layers['counts']")
    else if (is.character(ex$counts_checksum) && length(p) == 0) {
      nd <- h5[["layers/counts"]]
      enc <- .scx_attr(nd, "encoding-type", "")
      cm <- if (enc %in% c("csr_matrix", "csc_matrix")) .scx_read_sparse_t(nd) else t(as.matrix(.scx_read_elem(nd)))
      if (!identical(sc_counts_checksum(cm), ex$counts_checksum)) add("counts_checksum does not match layers['counts']")
    }
  }
  if (mode_ok && mode == "results" && is.character(ex$cluster_key) && length(ex$cluster_key) == 1) {
    ck <- ex$cluster_key
    cols <- as.character(.scx_attr(obs_g, "column-order", character(0)))
    if (!(ck %in% cols)) add("cluster_key '", ck, "' is not an obs column")
    else if (!identical(.scx_attr(obs_g[[ck]], "encoding-type"), "categorical")) add("obs['", ck, "'] must be categorical")
  }
  p
}

# --------------------------------------------------------------------------
# merge_exchange_results (spec section 7)
# --------------------------------------------------------------------------

#' Merge a results-mode file back into the Seurat object it came from.
#'
#' Refuses when the file is invalid, is not mode "results", or its
#' source_counts_checksum differs from the object's counts. Cells are matched
#' by barcode; barcodes unknown to the object are always an error, object
#' cells missing from the results are an error unless allow_subset = TRUE
#' (their cluster label is NA and they are absent from the reductions and
#' graphs). Adds DimReducs (pca/harmony/umap with keys PC_/harmony_/UMAP_,
#' other X_<name> -> <name> with key toupper(name)_), Graphs <assay>_snn /
#' <assay>_nn, and the cluster column (uns exchange cluster_key) as a factor in
#' meta.data under that name and as seurat_clusters, and sets Idents().
#' Existing slots of the same names are replaced only with overwrite = TRUE.
merge_exchange_results <- function(obj, path, overwrite = FALSE, allow_subset = FALSE, assay = "RNA") {
  if (!inherits(obj, "Seurat")) .scx_stop("obj must be a Seurat object")
  probs <- validate_exchange(path)
  if (length(probs)) .scx_stop("invalid exchange file ", path, ":\n  ", paste(probs, collapse = "\n  "))
  res <- read_exchange(path)
  ex <- res$uns$exchange
  if (!identical(ex$mode, "results")) .scx_stop("merge needs a results-mode file (got mode '", ex$mode, "')")

  counts <- .scx_layer(obj, assay, "counts")
  if (is.null(counts)) .scx_stop("object has no counts layer in assay '", assay, "'")
  cs <- sc_counts_checksum(counts)
  obj_cells <- colnames(obj)
  res_cells <- rownames(res$obs)
  if (!identical(cs, ex$source_counts_checksum) && setequal(obj_cells, res_cells) &&
      length(obj_cells) == length(res_cells)) {
    # The checksum depends on cell order; an object whose cells were only
    # reordered since the slim file was written still holds the same counts.
    cs_perm <- sc_counts_checksum(counts[, match(res_cells, obj_cells), drop = FALSE])
    if (identical(cs_perm, ex$source_counts_checksum)) cs <- cs_perm
  }
  if (!identical(cs, ex$source_counts_checksum))
    .scx_stop("source_counts_checksum mismatch: results were computed from a different object ",
              "(file ", ex$source_counts_checksum, ", object ", cs, "); merge refused")

  unknown <- setdiff(res_cells, obj_cells)
  if (length(unknown)) .scx_stop(length(unknown), " barcode(s) in the results are not in the object, e.g. ",
                                 paste(head(unknown, 5), collapse = ", "))
  missing <- setdiff(obj_cells, res_cells)
  if (length(missing) && !isTRUE(allow_subset))
    .scx_stop(length(missing), " object cell(s) are missing from the results, e.g. ",
              paste(head(missing, 5), collapse = ", "), " (pass allow_subset = TRUE to accept)")
  present <- obj_cells[obj_cells %in% res_cells]   # object order
  if (length(missing))
    .scx_warn(length(missing), " object cell(s) absent from the results: NA cluster label, ",
              "not in the new reductions/graphs")

  ck <- ex$cluster_key
  if (!(ck %in% colnames(res$obs))) .scx_stop("cluster_key '", ck, "' is not an obs column")
  red_names <- vapply(names(res$obsm), .scx_reduction_name, "")
  graph_names <- vapply(names(res$obsp), .scx_graph_name, "", assay = assay)
  meta_cols <- unique(c(ck, "seurat_clusters"))
  if (!isTRUE(overwrite)) {
    clash <- c(intersect(red_names, SeuratObject::Reductions(obj)),
               intersect(graph_names, SeuratObject::Graphs(obj)),
               intersect(meta_cols, colnames(obj[[]])))
    if (length(clash)) .scx_stop("object already has: ", paste(clash, collapse = ", "),
                                 "; pass overwrite = TRUE to replace")
  }

  # Seurat warns when a DimReduc/Graph covers fewer cells than an existing
  # slot; with allow_subset that is expected and reported once above.
  quiet <- function(expr) withCallingHandlers(expr, warning = function(w) {
    if (length(missing) && grepl("Number of cells changing|Different cells from existing", conditionMessage(w)))
      invokeRestart("muffleWarning")
  })
  for (k in names(res$obsm)) {
    emb <- res$obsm[[k]]
    if (is.data.frame(emb)) next
    rn <- .scx_reduction_name(k)
    quiet(obj[[rn]] <- .scx_make_dimreduc(rn, emb[present, , drop = FALSE], assay))
  }
  for (k in names(res$obsp)) {
    G <- res$obsp[[k]][present, present, drop = FALSE]
    quiet(obj[[.scx_graph_name(k, assay)]] <- .scx_make_graph(.scx_as_dgc(G), assay))
  }
  cl <- res$obs[[ck]]
  if (!is.factor(cl)) cl <- factor(as.character(cl))
  full_cl <- factor(as.character(cl)[match(obj_cells, res_cells)], levels = levels(cl))
  names(full_cl) <- obj_cells
  for (mc in meta_cols) obj[[mc]] <- full_cl
  SeuratObject::Idents(obj) <- full_cl

  rec <- list(path = normalizePath(path, mustWork = FALSE), run_id = ex$run_id, stage = ex$stage,
              writer = ex$writer, created_utc = ex$created_utc,
              source_counts_checksum = ex$source_counts_checksum, cluster_key = ck,
              n_cells = length(present), merged_utc = .scx_utc_now())
  obj@misc$cellwarden_exchange_merges <- c(obj@misc$cellwarden_exchange_merges, list(rec))
  obj
}
