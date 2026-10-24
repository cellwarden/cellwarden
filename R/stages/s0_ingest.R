# cellwarden stage S0: ingest (docs/stages-spec.md section 3.1).
#
# Reads every sample in samples.csv, checks counts are raw integers, prefixes
# barcodes with <sample_id>_, harmonises genes across samples (Ensembl ID key
# when every sample has IDs, else symbols), merges into one Seurat v5 object
# with a single counts layer, and writes S0/object.rds + S0/ingest_report.json.
#
# Per-sample readers return a common structure:
#   list(counts = dgCMatrix genes x cells (colnames = raw barcodes),
#        symbol = character (one per row), gene_id = character ("" unknown),
#        meta = data.frame (rows = cells) or NULL, notes = character(),
#        optional: warnings = character(), orientation = list (tables))
# Raw 10x matrices are filtered for empty droplets per sample before
# prefixing (config ingest.empty_drops: auto | on | off).

.S0_FORMATS <- c("10x_h5", "10x_mtx", "h5ad", "seurat", "loom", "table")
.S0_RESERVED_META <- c("orig.ident", "nCount_RNA", "nFeature_RNA", "sample_id", "ident",
                       "seurat_clusters", "nCount_SCT", "nFeature_SCT")
.S0_ENSG_RE <- "^ENSG[0-9]{6,}(\\.[0-9]+)?$"

.s0_seurat_safe <- function(x) gsub("[_|]", "-", x)

.s0_first_existing <- function(dir, names) {
  for (n in names) {
    f <- file.path(dir, n)
    if (file.exists(f)) return(f)
  }
  NA_character_
}

# --------------------------------------------------------------------------
# Format detection
# --------------------------------------------------------------------------

s0_detect_format <- function(path) {
  if (dir.exists(path)) {
    if (!is.na(.s0_first_existing(path, c("matrix.mtx", "matrix.mtx.gz")))) return("10x_mtx")
    scp_stop("directory ", path, " has no matrix.mtx(.gz); cannot detect its format")
  }
  if (!file.exists(path)) scp_stop("input not found: ", path)
  p <- tolower(path)
  if (grepl("\\.h5ad$", p)) return("h5ad")
  if (grepl("\\.(rds|qs2|qs)$", p)) return("seurat")
  if (grepl("\\.loom$", p)) return("loom")
  if (grepl("\\.(csv|tsv|txt)(\\.gz)?$", p)) return("table")
  if (grepl("matrix\\.mtx(\\.gz)?$", p)) return("10x_mtx")
  if (grepl("\\.(h5|hdf5)$", p)) {
    h5 <- tryCatch(hdf5r::H5File$new(path, mode = "r"), error = function(e) NULL)
    if (is.null(h5)) scp_stop("cannot open ", path, " as HDF5")
    on.exit(h5$close_all(), add = TRUE)
    top <- names(h5)
    if ("matrix" %in% top) return("10x_h5")
    if (all(c("obs", "var") %in% top)) return("h5ad")
    for (g in top) {
      if (inherits(h5[[g]], "H5Group") && all(c("barcodes", "data", "indptr") %in% names(h5[[g]])))
        return("10x_h5")   # Cell Ranger v2 layout: /<genome>/...
    }
    scp_stop(path, " is an HDF5 file without a 10x 'matrix' group or AnnData obs/var")
  }
  scp_stop("cannot detect the format of ", path, " (set the 'format' column)")
}

# --------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------

.s0_read_lines_table <- function(f) {
  con <- if (grepl("\\.gz$", f)) gzfile(f) else file(f)
  utils::read.delim(con, header = FALSE, colClasses = "character", quote = "",
                    comment.char = "", na.strings = character())
}

# 10x MatrixMarket directory (v2 genes.tsv or v3 features.tsv, each optionally
# gzipped). Seurat::Read10X expects all v3 files gzipped, so this reader takes
# any mix; columns follow Read10X: 1 = Ensembl ID, 2 = symbol, 3 = feature type.
s0_read_10x_mtx <- function(path) {
  dir <- if (dir.exists(path)) path else dirname(path)
  mf <- .s0_first_existing(dir, c("matrix.mtx.gz", "matrix.mtx"))
  ff <- .s0_first_existing(dir, c("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"))
  bf <- .s0_first_existing(dir, c("barcodes.tsv.gz", "barcodes.tsv"))
  miss <- c(matrix = mf, features = ff, barcodes = bf)
  if (anyNA(miss)) scp_stop("10x directory ", dir, " is missing: ", paste(names(miss)[is.na(miss)], collapse = ", "))
  m <- Matrix::readMM(if (grepl("\\.gz$", mf)) gzfile(mf) else mf)
  m <- as(as(as(m, "CsparseMatrix"), "generalMatrix"), "dMatrix")
  feats <- .s0_read_lines_table(ff)
  bcs <- .s0_read_lines_table(bf)[[1]]
  if (nrow(feats) != nrow(m)) scp_stop(ff, " has ", nrow(feats), " rows but the matrix has ", nrow(m), " genes")
  if (length(bcs) != ncol(m)) scp_stop(bf, " has ", length(bcs), " rows but the matrix has ", ncol(m), " cells")
  notes <- character()
  if (ncol(feats) >= 2) { gid <- feats[[1]]; sym <- feats[[2]] }
  else { sym <- feats[[1]]; gid <- ifelse(grepl(.S0_ENSG_RE, sym), sym, "") }
  if (ncol(feats) >= 3) {
    ft <- feats[[3]]
    if (any(ft == "Gene Expression") && any(ft != "Gene Expression")) {
      keep <- ft == "Gene Expression"
      notes <- c(notes, sprintf("dropped %d non-'Gene Expression' features (%s)", sum(!keep),
                                paste(unique(ft[!keep]), collapse = ", ")))
      m <- m[keep, , drop = FALSE]; sym <- sym[keep]; gid <- gid[keep]
    }
  }
  sym[is.na(sym) | !nzchar(sym)] <- gid[is.na(sym) | !nzchar(sym)]
  dimnames(m) <- list(NULL, bcs)
  list(counts = m, symbol = sym, gene_id = gid, meta = NULL, notes = notes)
}

# 10x HDF5 (Cell Ranger v3 'matrix' group, or v2 per-genome groups) via
# Seurat::Read10X_h5 with IDs as row names; symbols and feature types from hdf5r.
s0_read_10x_h5 <- function(path) {
  h5 <- hdf5r::H5File$new(path, mode = "r")
  feat <- tryCatch({
    if ("matrix" %in% names(h5)) {
      f <- h5[["matrix/features"]]
      list(list(id = f[["id"]][], name = f[["name"]][],
                type = if ("feature_type" %in% names(f)) f[["feature_type"]][] else NULL))
    } else {
      lapply(names(h5), function(g) list(id = h5[[g]][["genes"]][], name = h5[[g]][["gene_names"]][], type = NULL))
    }
  }, finally = h5$close_all())
  ids_all <- unlist(lapply(feat, `[[`, "id"))
  names_all <- unlist(lapply(feat, `[[`, "name"))
  types_all <- unlist(lapply(feat, function(x) if (is.null(x$type)) rep("Gene Expression", length(x$id)) else x$type))
  m <- suppressWarnings(Seurat::Read10X_h5(path, use.names = FALSE, unique.features = TRUE))
  notes <- character()
  if (is.list(m)) {
    if ("Gene Expression" %in% names(m)) {
      dropped <- setdiff(names(m), "Gene Expression")
      if (length(dropped)) notes <- c(notes, paste0("dropped feature types: ", paste(dropped, collapse = ", ")))
      m <- m[["Gene Expression"]]
    } else {
      notes <- c(notes, paste0("several genomes (", paste(names(m), collapse = ", "), "); their genes are combined"))
      m <- do.call(rbind, unname(m))
    }
  }
  m <- as(as(as(m, "CsparseMatrix"), "generalMatrix"), "dMatrix")
  ids <- rownames(m)
  # Read10X_h5 makes duplicate IDs unique; map back through make.unique().
  idx <- match(ids, make.unique(ids_all))
  if (anyNA(idx)) idx[is.na(idx)] <- match(ids[is.na(idx)], ids_all)
  sym <- names_all[idx]
  gid <- ids_all[idx]
  gid[is.na(gid)] <- ids[is.na(gid)]
  sym[is.na(sym) | !nzchar(sym)] <- gid[is.na(sym) | !nzchar(sym)]
  bcs <- colnames(m)
  dimnames(m) <- list(NULL, bcs)
  list(counts = m, symbol = sym, gene_id = gid, meta = NULL, notes = notes)
}

# Gene symbols / IDs from an AnnData var data.frame (index + common columns).
.s0_var_genes <- function(var) {
  idx <- rownames(var)
  pick <- function(cands) { k <- intersect(cands, colnames(var)); if (length(k)) as.character(var[[k[1]]]) else NULL }
  sym_col <- pick(c("gene_symbol_original", "gene_symbols", "gene_symbol", "feature_name", "symbol", "gene_name", "Symbol"))
  id_col <- pick(c("gene_id", "gene_ids", "ensembl_id", "feature_id", "Accession", "ID"))
  idx_is_id <- length(idx) > 0 && mean(grepl(.S0_ENSG_RE, idx)) > 0.9
  sym <- if (!is.null(sym_col)) sym_col else idx
  gid <- if (!is.null(id_col)) id_col else if (idx_is_id) idx else rep("", length(idx))
  if (is.null(sym_col) && idx_is_id) sym <- idx
  sym[is.na(sym) | !nzchar(sym)] <- idx[is.na(sym) | !nzchar(sym)]
  gid[is.na(gid)] <- ""
  list(symbol = sym, gene_id = gid)
}

.s0_meta_from_obs <- function(obs) {
  if (is.null(obs) || ncol(obs) == 0) return(NULL)
  keep <- vapply(obs, function(v) is.atomic(v) && is.null(dim(v)), TRUE)
  keep <- keep & !(colnames(obs) %in% .S0_RESERVED_META) & !grepl("^(nCount|nFeature)_", colnames(obs))
  if (!any(keep)) return(NULL)
  obs[, keep, drop = FALSE]
}

.s0_is_integer_mat <- function(m) {
  x <- if (is(m, "sparseMatrix")) m@x else as.vector(m)
  !length(x) || all(is.finite(x) & x == round(x))
}

# AnnData: exchange files through read_exchange(); other files: counts from
# layers/counts, raw/X, or integer X (in that order).
s0_read_h5ad <- function(path) {
  h5 <- .scx_open(path)
  is_exchange <- .scx_has(h5, "uns/exchange/schema_version")
  h5$close_all()
  if (is_exchange) {
    ex <- read_exchange(path)
    mode <- ex$uns$exchange$mode
    if (!identical(mode, "full") || is.null(ex$layers[["counts"]]))
      scp_stop(path, " is an exchange file in '", mode, "' mode, which carries no raw counts (need mode 'full')")
    g <- .s0_var_genes(ex$var)
    m <- .scx_as_dgc(ex$layers[["counts"]])
    dimnames(m) <- list(NULL, rownames(ex$obs))
    return(list(counts = m, symbol = g$symbol, gene_id = g$gene_id,
                meta = .s0_meta_from_obs(ex$obs), notes = "read as cellwarden exchange file (layers['counts'])"))
  }
  h5 <- .scx_open(path)
  on.exit(h5$close_all(), add = TRUE)
  rd <- function(node) {
    enc <- .scx_attr(node, "encoding-type", "")
    if (enc %in% c("csr_matrix", "csc_matrix")) return(.scx_read_sparse_t(node))
    if (!.scx_is_group(node)) {
      v <- .scx_read_dense(node)                 # cells x genes
      return(Matrix::Matrix(t(v), sparse = TRUE))
    }
    scp_stop("unsupported matrix encoding '", enc, "' in ", path)
  }
  obs <- tryCatch(.scx_read_dataframe(h5[["obs"]]), error = function(e)
    scp_stop("cannot read obs of ", path, " (AnnData >= 0.7 on-disk format needed): ", conditionMessage(e)))
  var <- .scx_read_dataframe(h5[["var"]])
  src <- NULL; m <- NULL; v <- var
  if (.scx_has(h5, "layers/counts")) {
    m <- rd(h5[["layers/counts"]]); src <- "layers['counts']"
  } else if (.scx_has(h5, "raw/X")) {
    m <- rd(h5[["raw/X"]]); src <- "raw/X"
    v <- .scx_read_dataframe(h5[["raw/var"]])
  } else if (.scx_has(h5, "X")) {
    m <- rd(h5[["X"]]); src <- "X"
    if (!.s0_is_integer_mat(m))
      scp_stop("h5ad ", path, ": no layers['counts'] or raw/X, and X is not integer counts")
  } else scp_stop("h5ad ", path, " has no X, raw/X or layers['counts']")
  m <- as(as(as(m, "CsparseMatrix"), "generalMatrix"), "dMatrix")
  if (nrow(m) != nrow(v) || ncol(m) != nrow(obs)) scp_stop("h5ad ", path, ": ", src, " shape does not match obs/var")
  g <- .s0_var_genes(v)
  dimnames(m) <- list(NULL, rownames(obs))
  list(counts = m, symbol = g$symbol, gene_id = g$gene_id, meta = .s0_meta_from_obs(obs),
       notes = paste0("counts from ", src))
}

# Seurat (.rds/.qs2): v4 assays converted to Assay5, split layers joined.
# SingleCellExperiment objects are accepted too (assay 'counts').
s0_read_seurat <- function(path) {
  x <- scp_read_object_file(path)
  notes <- character()
  if (inherits(x, "SingleCellExperiment")) {
    m <- SummarizedExperiment::assay(x, "counts")
    rd <- as.data.frame(SummarizedExperiment::rowData(x))
    rownames(rd) <- rownames(x)
    g <- .s0_var_genes(rd)
    meta <- .s0_meta_from_obs(as.data.frame(SummarizedExperiment::colData(x)))
    m <- scp_as_dgc(m)
    bcs <- colnames(x)
    dimnames(m) <- list(NULL, bcs)
    return(list(counts = m, symbol = g$symbol, gene_id = g$gene_id, meta = meta,
                notes = "SingleCellExperiment, assay 'counts'"))
  }
  if (!inherits(x, "Seurat")) scp_stop(path, " does not hold a Seurat or SingleCellExperiment object (class ", class(x)[1], ")")
  assay <- SeuratObject::DefaultAssay(x)
  if (assay %in% c("SCT", "integrated") && "RNA" %in% SeuratObject::Assays(x)) assay <- "RNA"
  a <- x[[assay]]
  if (!inherits(a, "Assay5")) {
    a5 <- tryCatch(as(a, "Assay5"), error = function(e) NULL)
    if (is.null(a5)) {
      notes <- c(notes, paste0("assay '", assay, "' is v3/v4-style and could not be converted; counts read directly"))
      m <- SeuratObject::GetAssayData(x, assay = assay, slot = "counts")
      fm <- tryCatch(a[[]], error = function(e) NULL)
    } else {
      notes <- c(notes, paste0("v3/v4 assay '", assay, "' converted to Assay5"))
      a <- a5
    }
  }
  if (inherits(a, "Assay5")) {
    lys <- SeuratObject::Layers(a)
    if (!("counts" %in% lys)) {
      if (any(grepl("^counts\\.", lys))) {
        a <- SeuratObject::JoinLayers(a, layers = "counts")
        notes <- c(notes, "split counts layers joined")
      } else scp_stop("Seurat object ", path, ": assay '", assay, "' has no counts layer")
    }
    m <- SeuratObject::LayerData(a, layer = "counts")
    fm <- tryCatch(a[[]], error = function(e) NULL)
  }
  m <- scp_as_dgc(m)
  feats <- rownames(m)
  if (is.null(feats)) feats <- rownames(x[[assay]])
  sym <- feats; gid <- rep("", length(feats))
  if (!is.null(fm) && nrow(fm) == length(feats)) {
    fm <- fm[match(feats, rownames(fm)), , drop = FALSE]
    rownames(fm) <- feats
    g <- .s0_var_genes(fm)
    sym <- g$symbol; gid <- g$gene_id
  } else {
    gid[grepl(.S0_ENSG_RE, feats)] <- feats[grepl(.S0_ENSG_RE, feats)]
  }
  bcs <- colnames(m)
  md <- x[[]]
  md <- md[match(bcs, rownames(md)), , drop = FALSE]
  dimnames(m) <- list(NULL, bcs)
  list(counts = m, symbol = sym, gene_id = gid, meta = .s0_meta_from_obs(md), notes = notes)
}

# Loom: /matrix is genes x cells in loompy terms (h5py shape (genes, cells));
# hdf5r reverses dimensions, so blocks read here are cells x genes.
s0_read_loom <- function(path) {
  h5 <- hdf5r::H5File$new(path, mode = "r")
  on.exit(h5$close_all(), add = TRUE)
  if (!h5$exists("matrix")) scp_stop("loom ", path, " has no /matrix")
  d <- h5[["matrix"]]
  dims <- d$dims                         # R order: (cells, genes)
  n_cells <- dims[[1]]; n_genes <- dims[[2]]
  ra <- if (h5$exists("row_attrs")) h5[["row_attrs"]] else NULL
  ca <- if (h5$exists("col_attrs")) h5[["col_attrs"]] else NULL
  getattr <- function(grp, cands) {
    if (is.null(grp)) return(NULL)
    for (k in cands) if (grp$exists(k)) return(as.character(grp[[k]][]))
    NULL
  }
  sym <- getattr(ra, c("Gene", "gene_symbols", "gene_name", "var_names"))
  gid <- getattr(ra, c("Accession", "gene_ids", "gene_id", "ensembl_id"))
  bcs <- getattr(ca, c("CellID", "obs_names", "cell_names", "barcode"))
  if (is.null(sym) && is.null(gid)) scp_stop("loom ", path, " has no /row_attrs/Gene")
  if (is.null(sym)) sym <- gid
  if (is.null(gid)) gid <- ifelse(grepl(.S0_ENSG_RE, sym), sym, "")
  if (is.null(bcs)) scp_stop("loom ", path, " has no /col_attrs/CellID")
  if (length(sym) != n_genes || length(bcs) != n_cells) scp_stop("loom ", path, ": attribute lengths do not match /matrix")
  block <- max(1L, as.integer(floor(5e7 / max(1, n_cells))))
  parts <- list()
  s <- 1L
  while (s <= n_genes) {
    e <- min(n_genes, s + block - 1L)
    v <- d[, s:e, drop = FALSE]          # cells x genes block
    parts[[length(parts) + 1L]] <- Matrix::Matrix(t(v), sparse = TRUE)
    s <- e + 1L
  }
  m <- do.call(rbind, parts)
  m <- as(as(as(m, "CsparseMatrix"), "generalMatrix"), "dMatrix")
  dimnames(m) <- list(NULL, bcs)
  list(counts = m, symbol = sym, gene_id = gid, meta = NULL, notes = character())
}

# Fraction of names that look like genes: known human symbols (case-insensitive,
# so mouse-style Cd3e counts) or Ensembl gene IDs of any species. Samples at
# most n names to stay fast on wide tables.
.s0_gene_like_fraction <- function(x, repo, n = 2000L) {
  x <- as.character(x)
  x <- x[!is.na(x) & nzchar(x)]
  if (!length(x)) return(0)
  if (length(x) > n) x <- x[unique(round(seq(1, length(x), length.out = n)))]
  hs <- toupper(.s0_human_symbols(repo))
  mean(toupper(x) %in% hs | grepl("^ENS[A-Z]*G[0-9]", x, ignore.case = TRUE))
}

# Table orientation from the header names vs the first-column values: the side
# that looks like genes (>= 5% gene-like and >= 5x the other side) holds genes.
.s0_table_orientation <- function(header, first_col, repo) {
  fh <- .s0_gene_like_fraction(header, repo)
  fc <- .s0_gene_like_fraction(first_col, repo)
  o <- if (fh >= 0.05 && fc < fh / 5) "cells_x_genes"
       else if (fc >= 0.05 && fh < fc / 5) "genes_x_cells" else "ambiguous"
  list(orientation = o, header_gene_fraction = round(fh, 4), first_column_gene_fraction = round(fc, 4))
}

# Dense table: genes x cells (header row of cell names; first column gene
# names, or first two columns gene ID + symbol when both are non-numeric) or
# cells x genes (header row of gene names; first column cell names), detected
# by .s0_table_orientation; ambiguous tables are read as genes x cells.
s0_read_table <- function(path, repo = ".") {
  sep <- if (grepl("\\.csv(\\.gz)?$", tolower(path))) "," else "\t"
  if (scp_has_pkg("data.table") && (!grepl("\\.gz$", path) || scp_has_pkg("R.utils"))) {
    df <- data.table::fread(path, sep = sep, header = TRUE, data.table = FALSE, check.names = FALSE)
  } else {
    con <- if (grepl("\\.gz$", path)) gzfile(path) else file(path)
    df <- utils::read.table(con, sep = sep, header = TRUE, check.names = FALSE, quote = "\"",
                            comment.char = "", stringsAsFactors = FALSE)
  }
  if (ncol(df) < 2) scp_stop("table ", path, " needs a gene column and at least one cell column")
  n_lead <- 1L
  if (ncol(df) >= 3 && !is.numeric(df[[2]])) n_lead <- 2L
  lead <- lapply(df[seq_len(n_lead)], as.character)
  vals <- df[-seq_len(n_lead)]
  nonnum <- !vapply(vals, is.numeric, TRUE)
  if (any(nonnum)) scp_stop("table ", path, ": non-numeric cell columns, e.g. ", paste(head(names(vals)[nonnum], 3), collapse = ", "))
  notes <- character(); warns <- character()
  if (n_lead == 2L) {
    ori <- list(orientation = "genes_x_cells", reason = "two leading gene columns (ID + symbol)")
  } else {
    ori <- .s0_table_orientation(colnames(vals), lead[[1]], repo)
    ori$reason <- sprintf("gene-like names: %.0f%% of the header, %.0f%% of the first column",
                          100 * ori$header_gene_fraction, 100 * ori$first_column_gene_fraction)
  }
  if (identical(ori$orientation, "cells_x_genes")) {
    m <- Matrix::Matrix(t(as.matrix(vals)), sparse = TRUE)
    m <- as(as(as(m, "CsparseMatrix"), "generalMatrix"), "dMatrix")
    sym <- colnames(vals)
    gid <- ifelse(grepl(.S0_ENSG_RE, sym), sym, "")
    dimnames(m) <- list(NULL, lead[[1]])
    notes <- c(notes, paste0("table read as cells x genes and transposed (", ori$reason, ")"))
    return(list(counts = m, symbol = sym, gene_id = gid, meta = NULL, notes = notes, warnings = warns,
                orientation = ori))
  }
  if (identical(ori$orientation, "ambiguous")) {
    warns <- c(warns, paste0("table orientation is ambiguous (", ori$reason, "); read as genes x cells"))
    ori$orientation <- "genes_x_cells"; ori$ambiguous <- TRUE
  }
  notes <- c(notes, paste0("table read as genes x cells (", ori$reason, ")"))
  m <- Matrix::Matrix(as.matrix(vals), sparse = TRUE)
  m <- as(as(as(m, "CsparseMatrix"), "generalMatrix"), "dMatrix")
  if (n_lead == 2L) {
    a <- lead[[1]]; b <- lead[[2]]
    if (mean(grepl(.S0_ENSG_RE, a)) >= mean(grepl(.S0_ENSG_RE, b))) { gid <- a; sym <- b } else { gid <- b; sym <- a }
  } else {
    sym <- lead[[1]]
    gid <- ifelse(grepl(.S0_ENSG_RE, sym), sym, "")
  }
  dimnames(m) <- list(NULL, colnames(vals))
  list(counts = m, symbol = sym, gene_id = gid, meta = NULL, notes = notes, warnings = warns, orientation = ori)
}

s0_read_sample <- function(path, format, repo = ".") {
  switch(format,
         "10x_h5" = s0_read_10x_h5(path),
         "10x_mtx" = s0_read_10x_mtx(path),
         "h5ad" = s0_read_h5ad(path),
         "seurat" = s0_read_seurat(path),
         "loom" = s0_read_loom(path),
         "table" = s0_read_table(path, repo),
         scp_stop("unknown format '", format, "' (allowed: ", paste(.S0_FORMATS, collapse = ", "), ")"))
}

.s0_raw_hint <- function(format) {
  switch(format,
         h5ad = "in AnnData files raw counts usually live in layers['counts'] or raw.X",
         seurat = "in Seurat objects raw counts live in the 'counts' layer of the RNA assay (not 'data' or SCT)",
         "10x_h5" = , "10x_mtx" = "use Cell Ranger's filtered_feature_bc_matrix (raw UMI counts)",
         loom = "the loom /matrix must hold raw counts (check /layers for a counts layer)",
         table = "the table must hold raw UMI counts, not normalised or log values",
         "check that the input holds raw UMI counts")
}

# --------------------------------------------------------------------------
# Sample sheet
# --------------------------------------------------------------------------

s0_read_samples <- function(file, base_dir) {
  df <- utils::read.csv(file, colClasses = "character", check.names = FALSE, na.strings = character(),
                        strip.white = TRUE)
  p <- character()
  for (k in c("sample_id", "path")) if (!(k %in% colnames(df))) p <- c(p, paste0("missing required column '", k, "'"))
  if (length(p)) scp_stop("samples sheet ", file, ": ", paste(p, collapse = "; "))
  if (nrow(df) == 0) scp_stop("samples sheet ", file, " has no rows")
  if (!("format" %in% colnames(df))) df$format <- ""
  bad_id <- which(!nzchar(df$sample_id))
  if (length(bad_id)) p <- c(p, paste0("empty sample_id in row(s) ", paste(bad_id + 1, collapse = ", ")))
  dup <- unique(df$sample_id[duplicated(df$sample_id)])
  if (length(dup)) p <- c(p, paste0("duplicated sample_id: ", paste(dup, collapse = ", ")))
  if (any(!nzchar(df$path))) p <- c(p, paste0("empty path for sample(s) ", paste(df$sample_id[!nzchar(df$path)], collapse = ", ")))
  badf <- nzchar(df$format) & !(df$format %in% .S0_FORMATS)
  if (any(badf)) p <- c(p, paste0("unknown format for sample(s) ", paste(df$sample_id[badf], collapse = ", "),
                                  " (allowed: ", paste(.S0_FORMATS, collapse = ", "), ")"))
  if (length(p)) scp_stop("samples sheet ", file, ":\n  ", paste(p, collapse = "\n  "))
  abs_path <- function(x) {
    x <- path.expand(x)
    x <- gsub("\\$\\{?USER\\}?", Sys.getenv("USER"), x)
    if (grepl("^(/|[A-Za-z]:[\\\\/])", x)) x else file.path(base_dir, x)
  }
  df$path <- vapply(df$path, function(x) normalizePath(abs_path(x), mustWork = FALSE), "")
  missing <- !file.exists(df$path)
  if (any(missing)) scp_stop("input path(s) not found:\n  ",
                             paste(sprintf("%s: %s", df$sample_id[missing], df$path[missing]), collapse = "\n  "))
  df
}

# --------------------------------------------------------------------------
# Metadata binding
# --------------------------------------------------------------------------

# rbind a list of data.frames with possibly different columns; factors kept
# as factors (levels in order of first appearance), mixed types -> character.
.s0_bind_meta <- function(dfs, n_rows) {
  cols <- unique(unlist(lapply(dfs, function(d) if (is.null(d)) character() else colnames(d))))
  out <- list()
  for (cn in cols) {
    pieces <- lapply(seq_along(dfs), function(i) {
      d <- dfs[[i]]
      if (!is.null(d) && cn %in% colnames(d)) d[[cn]] else rep(NA, n_rows[[i]])
    })
    present <- Filter(function(v) !all(is.na(v)) || !is.logical(v), pieces)
    kinds <- unique(vapply(present, function(v) if (is.factor(v)) "factor" else if (is.logical(v)) "logical"
                           else if (is.numeric(v)) "numeric" else "character", ""))
    if (length(kinds) == 1 && kinds %in% c("numeric", "logical")) {
      out[[cn]] <- unlist(lapply(pieces, function(v) if (kinds == "numeric") as.numeric(v) else as.logical(v)), use.names = FALSE)
    } else {
      ch <- unlist(lapply(pieces, as.character), use.names = FALSE)
      lv <- unique(unlist(lapply(pieces, function(v) if (is.factor(v)) levels(v) else unique(as.character(v)))))
      lv <- lv[!is.na(lv)]
      out[[cn]] <- if (length(kinds) == 1 && kinds == "character") ch else factor(ch, levels = unique(c(lv, unique(ch[!is.na(ch)]))))
    }
  }
  as.data.frame(out, stringsAsFactors = FALSE, optional = TRUE, check.names = FALSE)
}

# --------------------------------------------------------------------------
# Organism check
# --------------------------------------------------------------------------

.s0_human_symbols <- local({
  cache <- NULL
  function(repo) {
    if (is.null(cache)) {
      f <- file.path(repo, "R", "stages", "data", "human_symbols.txt")
      l <- if (file.exists(f)) readLines(f, warn = FALSE) else character()
      cache <<- l[nzchar(l) & !startsWith(l, "#")]
    }
    cache
  }
})

# Fraction of the top-n most-expressed genes that look human.
s0_human_fraction <- function(totals, symbol, gene_id, repo, n_top = 1000L) {
  ok <- totals > 0
  if (!any(ok)) return(list(fraction = NA_real_, n_top = 0L))
  o <- order(-totals)
  o <- o[ok[o]][seq_len(min(n_top, sum(ok)))]
  hs <- .s0_human_symbols(repo)
  human <- grepl(.S0_ENSG_RE, gene_id[o]) | (symbol[o] %in% hs) | grepl("^MT-", symbol[o])
  list(fraction = round(mean(human), 4), n_top = length(o))
}

# --------------------------------------------------------------------------
# Empty droplets (raw 10x matrices)
# --------------------------------------------------------------------------

# ingest.empty_drops: auto | on | off (missing -> auto). YAML 1.1 reads bare
# on/off as booleans, so TRUE/FALSE map to on/off.
.s0_empty_drops_mode <- function(cfg) {
  v <- cfg$ingest$empty_drops %||% "auto"
  if (is.logical(v) && length(v) == 1 && !is.na(v)) v <- if (v) "on" else "off"
  v <- tolower(as.character(v))
  if (length(v) != 1 || !(v %in% c("auto", "on", "off")))
    scp_stop("config ingest.empty_drops must be one of auto, on, off (got '", paste(v, collapse = ","), "')")
  v
}

# Barcode-rank knee on the column totals. The curve is log10(UMI) vs
# log10(rank) for barcodes with UMI >= floor_umi (plus the first barcode below
# the floor, clamped to it, so a clean gap still ends in a cliff), resampled on
# a log-spaced rank grid and smoothed with a 5-point moving average.
#   knee:      kneedle point, max distance above the chord from the first
#              point to the floor point (end of the cell plateau);
#   threshold: midpoint of the steepest segment at or past the knee (the
#              cells -> empties cliff, cf. DropletUtils' inflection), capped
#              at the knee UMI and never below min_umi.
# Returns NULL when there are too few barcodes above the floor for a curve.
s0_barcode_knee <- function(tot, floor_umi = 10, min_umi = 50, n_grid = 1000L) {
  s <- sort(as.numeric(tot), decreasing = TRUE)
  n <- sum(s >= floor_umi)
  if (n < 20) return(NULL)
  n_pts <- min(length(s), n + 1L)
  ranks <- unique(round(10^seq(0, log10(n_pts), length.out = n_grid)))
  if (ranks[length(ranks)] != n_pts) ranks <- c(ranks, n_pts)
  x <- log10(ranks)
  y <- log10(pmax(s[ranks], floor_umi))
  ys <- y
  if (length(y) >= 5) {
    f <- stats::filter(y, rep(1 / 5, 5), sides = 2)
    ys[!is.na(f)] <- f[!is.na(f)]
  }
  k <- length(x)
  # distance above the chord (first point -> last point), up to a constant factor
  d <- (ys - ys[1]) * (x[k] - x[1]) - (x - x[1]) * (ys[k] - ys[1])
  ik <- which.max(d)
  knee_umi <- s[ranks[ik]]
  if (ik >= k) return(list(knee_umi = knee_umi, threshold = max(min_umi, knee_umi), knee_rank = ranks[ik]))
  seg <- ik:(k - 1L)
  slope <- (ys[seg + 1L] - ys[seg]) / pmax(x[seg + 1L] - x[seg], 1e-12)
  j <- seg[which.min(slope)]
  thr <- 10^((ys[j] + ys[j + 1L]) / 2)
  thr <- max(min_umi, min(thr, knee_umi))
  list(knee_umi = knee_umi, threshold = ceiling(thr), knee_rank = ranks[ik], steepest_slope = round(min(slope), 3))
}

# Decide and apply empty-droplet filtering for one 10x sample (before barcode
# prefixing). Works on column sums, then subsets columns.
s0_empty_drops <- function(m, path, format, mode) {
  info <- list(applied = FALSE, mode = mode, reason = NULL, threshold_umi = NULL,
               kept = ncol(m), dropped = 0L)
  if (!(format %in% c("10x_h5", "10x_mtx"))) return(NULL)
  if (mode == "off") { info$reason <- "ingest.empty_drops is off"; return(list(counts = m, info = info)) }
  tot <- Matrix::colSums(m)
  raw_name <- grepl("raw_feature_bc_matrix|raw_gene_bc_matrices", path, ignore.case = TRUE)
  med <- if (length(tot)) stats::median(tot) else 0
  if (mode == "auto") {
    if (raw_name) info$reason <- "auto: path names a raw 10x matrix"
    else if (length(tot) > 1e5 && med < 10)
      info$reason <- sprintf("auto: %d barcodes with median %g UMIs looks unfiltered", length(tot), med)
    else { info$reason <- "auto: matrix looks filtered (no raw_* path, <= 100,000 barcodes or median UMI >= 10)"
           return(list(counts = m, info = info)) }
  } else info$reason <- "ingest.empty_drops is on"
  kn <- s0_barcode_knee(tot)
  if (is.null(kn)) {
    info$reason <- paste0(info$reason, "; fewer than 20 barcodes with >= 10 UMIs, nothing filtered")
    return(list(counts = m, info = info))
  }
  keep <- tot >= kn$threshold
  if (sum(keep) < 10) {
    info$reason <- paste0(info$reason, sprintf("; only %d barcodes above the knee (%g UMIs), nothing filtered",
                                               sum(keep), kn$threshold))
    return(list(counts = m, info = info))
  }
  info$applied <- TRUE
  info$threshold_umi <- kn$threshold
  info$knee_umi <- kn$knee_umi
  info$kept <- sum(keep); info$dropped <- sum(!keep)
  list(counts = m[, keep, drop = FALSE], info = info)
}

# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------

run_s0 <- function(cfg, repo) {
  out_dir <- scp_stage_dir(cfg, "S0")
  samples_file <- scp_config_file(cfg, "samples", "samples.csv")
  base_dir <- if (is.character(cfg$config_dir) && nzchar(cfg$config_dir %||% "")) cfg$config_dir else dirname(samples_file)
  sheet <- s0_read_samples(samples_file, base_dir)
  scp_log("S0: ", nrow(sheet), " sample(s) from ", samples_file)
  extra_cols <- setdiff(colnames(sheet), c("sample_id", "path", "format"))
  ed_mode <- .s0_empty_drops_mode(cfg)
  timings <- list()

  samples <- list()
  reports <- list()
  for (i in seq_len(nrow(sheet))) {
    sid <- sheet$sample_id[i]
    fmt <- sheet$format[i]
    detected <- !nzchar(fmt)
    if (detected) fmt <- tryCatch(s0_detect_format(sheet$path[i]),
                                  error = function(e) scp_stop("sample ", sid, ": ", conditionMessage(e)))
    t0 <- proc.time()[["elapsed"]]
    s <- tryCatch(s0_read_sample(sheet$path[i], fmt, repo),
                  error = function(e) scp_stop("sample ", sid, " (", fmt, ", ", sheet$path[i], "): ", conditionMessage(e)))
    timings[[sid]] <- round(proc.time()[["elapsed"]] - t0, 3)
    m <- s$counts
    warn <- character()
    if (length(m@x) && any(m@x < 0)) scp_stop("sample ", sid, ": counts contain negative values; S0 needs raw UMI counts (",
                                             .s0_raw_hint(fmt), ")")
    if (!.s0_is_integer_mat(m)) {
      ex <- m@x[which(m@x != round(m@x))[1]]
      scp_stop("sample ", sid, ": counts are not integers (e.g. ", signif(ex, 6),
               "); S0 needs raw UMI counts, not normalised data. Hint: ", .s0_raw_hint(fmt))
    }
    m <- Matrix::drop0(m)
    notes <- s$notes
    ed <- s0_empty_drops(m, sheet$path[i], fmt, ed_mode)
    if (!is.null(ed)) {
      m <- ed$counts
      e <- ed$info
      notes <- c(notes, if (e$applied)
        sprintf("empty droplets removed (%s): kept %d of %d barcodes with >= %g UMIs (knee %g UMIs)",
                e$reason, e$kept, e$kept + e$dropped, e$threshold_umi, e$knee_umi)
        else paste0("no empty-droplet filtering (", e$reason, ")"))
      if (e$applied) scp_log(sprintf("sample %s: empty droplets: kept %d, dropped %d barcodes (threshold %g UMIs)",
                                     sid, e$kept, e$dropped, e$threshold_umi))
    }
    bcs <- colnames(m)
    if (anyDuplicated(bcs)) {
      warn <- c(warn, sprintf("%d duplicated barcodes made unique", sum(duplicated(bcs))))
      bcs <- make.unique(bcs)
    }
    colnames(m) <- paste0(sid, "_", bcs)
    sym <- as.character(s$symbol); gid <- as.character(s$gene_id)
    gid[is.na(gid)] <- ""
    sym[is.na(sym) | !nzchar(sym)] <- ifelse(nzchar(gid[is.na(sym) | !nzchar(sym)]), gid[is.na(sym) | !nzchar(sym)], "unknown")
    dup_sym <- unique(sym[duplicated(sym)])
    has_ids <- length(gid) > 0 && all(nzchar(gid)) && !anyDuplicated(gid)
    if (length(s$notes)) warn <- c(warn, s$notes[grepl("dropped|could not|several", s$notes)])
    warn <- c(warn, as.character(s$warnings))
    tot <- Matrix::rowSums(m)
    hf <- s0_human_fraction(tot, sym, gid, repo)
    if (!is.na(hf$fraction) && hf$fraction < 0.5)
      warn <- c(warn, sprintf("only %.0f%% of the top %d expressed genes look human; the data may not be human",
                              100 * hf$fraction, hf$n_top))
    for (w in warn) scp_warn_log("sample ", sid, ": ", w)
    scp_log(sprintf("sample %s: %s%s, %d cells x %d genes, %.0f UMIs%s", sid, fmt, if (detected) " (detected)" else "",
                    ncol(m), nrow(m), sum(m@x), if (length(dup_sym)) sprintf(", %d duplicated symbols", length(dup_sym)) else ""))
    samples[[sid]] <- list(counts = m, symbol = sym, gene_id = gid, has_ids = has_ids, meta = s$meta)
    reports[[sid]] <- list(sample_id = sid, path = sheet$path[i], format = fmt, format_detected = detected,
                           cells = ncol(m), genes = nrow(m), total_umis = sum(m@x),
                           has_gene_ids = has_ids,
                           duplicate_symbols = list(n = length(dup_sym), examples = head(dup_sym, 10)),
                           human_fraction_top_genes = hf$fraction,
                           notes = as.list(notes), warnings = as.list(warn),
                           read_seconds = timings[[sid]])
    if (!is.null(s$orientation)) reports[[sid]]$orientation <- s$orientation
    if (!is.null(ed)) reports[[sid]]$empty_drops <- ed$info
  }

  # ---- gene harmonisation
  use_ids <- all(vapply(samples, `[[`, TRUE, "has_ids"))
  key_name <- if (use_ids) "gene_id" else "gene_symbol"
  key_reason <- if (use_ids) "every sample has unique Ensembl/gene IDs" else {
    no_ids <- names(samples)[!vapply(samples, `[[`, TRUE, "has_ids")]
    paste0("sample(s) without complete gene IDs: ", paste(no_ids, collapse = ", "))
  }
  global_warn <- character()
  keys <- list()
  for (sid in names(samples)) {
    s <- samples[[sid]]
    k <- if (use_ids) s$gene_id else make.unique(s$symbol)
    if (!use_ids && length(reports[[sid]]$duplicate_symbols$examples))
      scp_log("sample ", sid, ": ", reports[[sid]]$duplicate_symbols$n, " duplicated symbol(s) made unique with make.unique, e.g. ",
              paste(head(reports[[sid]]$duplicate_symbols$examples, 5), collapse = ", "))
    keys[[sid]] <- k
  }
  union_keys <- unique(unlist(keys, use.names = FALSE))
  sym_u <- rep(NA_character_, length(union_keys)); gid_u <- rep("", length(union_keys))
  sym_conflicts <- 0L
  for (sid in names(samples)) {
    idx <- match(keys[[sid]], union_keys)
    s <- samples[[sid]]
    fresh <- is.na(sym_u[idx])
    sym_conflicts <- sym_conflicts + sum(!fresh & sym_u[idx] != s$symbol)
    sym_u[idx[fresh]] <- s$symbol[fresh]
    gfresh <- !nzchar(gid_u[idx]) & nzchar(s$gene_id)
    gid_u[idx[gfresh]] <- s$gene_id[gfresh]
  }
  if (sym_conflicts > 0) {
    w <- sprintf("%d gene ID(s) have different symbols across samples; the first sample's symbol is kept", sym_conflicts)
    scp_warn_log(w); global_warn <- c(global_warn, w)
  }
  base_names <- .s0_seurat_safe(if (use_ids) sym_u else union_keys)
  feat_names <- make.unique(base_names)
  n_renamed <- sum(feat_names != base_names)
  dup_union <- unique(base_names[duplicated(base_names)])
  if (n_renamed > 0) scp_log(n_renamed, " feature name(s) made unique across the merged gene set, e.g. ",
                             paste(head(dup_union, 5), collapse = ", "))

  # ---- merge (sparse, rows re-indexed to the union)
  n_union <- length(union_keys)
  mats <- lapply(names(samples), function(sid) {
    m <- samples[[sid]]$counts
    idx <- match(keys[[sid]], union_keys)
    out <- Matrix::sparseMatrix(i = idx[m@i + 1L], p = m@p, x = m@x, dims = c(n_union, ncol(m)),
                                dimnames = list(feat_names, colnames(m)), index1 = TRUE)
    as(out, "CsparseMatrix")
  })
  counts <- if (length(mats) == 1) mats[[1]] else do.call(cbind, mats)
  counts <- scp_as_dgc(counts)
  rm(mats)
  for (sid in names(samples)) samples[[sid]]$counts <- NULL
  if (anyDuplicated(colnames(counts))) scp_stop("cell barcodes are not unique after prefixing with sample_id")

  # ---- metadata
  n_per <- vapply(names(samples), function(sid) reports[[sid]]$cells, 1)
  sid_vec <- rep(names(samples), times = n_per)
  md <- data.frame(orig.ident = factor(sid_vec, levels = sheet$sample_id),
                   sample_id = factor(sid_vec, levels = sheet$sample_id),
                   row.names = colnames(counts), check.names = FALSE)
  for (cn in extra_cols) {
    v <- sheet[[cn]]
    v[!nzchar(v)] <- NA
    lv <- unique(v[!is.na(v)])
    md[[cn]] <- factor(rep(v, times = n_per), levels = lv)
  }
  carried <- lapply(names(samples), function(sid) samples[[sid]]$meta)
  if (any(!vapply(carried, is.null, TRUE))) {
    cm <- .s0_bind_meta(carried, n_per)
    clash <- intersect(colnames(cm), colnames(md))
    if (length(clash)) {
      w <- paste0("input metadata column(s) also in the sample sheet; sheet values kept: ", paste(clash, collapse = ", "))
      scp_warn_log(w); global_warn <- c(global_warn, w)
      cm <- cm[, setdiff(colnames(cm), clash), drop = FALSE]
    }
    if (ncol(cm)) {
      rownames(cm) <- colnames(counts)
      md <- cbind(md, cm)
      scp_log("carried input metadata column(s): ", paste(colnames(cm), collapse = ", "))
    }
  }

  # ---- object
  options(Seurat.object.assay.version = "v5")
  a5 <- SeuratObject::CreateAssay5Object(counts = counts)
  obj <- SeuratObject::CreateSeuratObject(a5, assay = "RNA", meta.data = md)
  fmeta <- data.frame(gene_symbol_original = sym_u, gene_id = gid_u, row.names = feat_names,
                      stringsAsFactors = FALSE)
  obj[["RNA"]] <- SeuratObject::AddMetaData(obj[["RNA"]], metadata = fmeta)
  obj@misc$cellwarden <- list(stage = "S0", run_id = cfg$run_id, config_hash = cfg$config_hash,
                          gene_key = key_name, created_utc = .scp_utc())

  # ---- organism check on the merged data
  tot <- Matrix::rowSums(counts)
  hf <- s0_human_fraction(tot, sym_u, gid_u, repo)
  human_warn <- NULL
  if (!is.na(hf$fraction) && hf$fraction < 0.5) {
    human_warn <- sprintf("only %.0f%% of the top %d expressed genes look human (Ensembl ENSG IDs or known human symbols); the data may not be human",
                          100 * hf$fraction, hf$n_top)
    scp_warn_log(human_warn); global_warn <- c(global_warn, human_warn)
  }

  scp_timed("save object", scp_save_object(obj, out_dir, cfg$object_format))
  report <- list(
    stage = "S0", run_id = cfg$run_id, config_hash = cfg$config_hash, created_utc = .scp_utc(),
    samples_file = samples_file,
    gene_key = key_name, gene_key_reason = key_reason,
    samples = unname(reports),
    totals = list(samples = length(samples), cells = ncol(counts), genes = nrow(counts),
                  total_umis = sum(counts@x)),
    feature_names = list(made_unique = n_renamed, examples = head(dup_union, 10),
                         symbol_conflicts_across_samples = sym_conflicts),
    sample_metadata_columns = as.list(extra_cols),
    human_check = list(fraction_top_genes = hf$fraction, n_top = hf$n_top, threshold = 0.5,
                       warning = human_warn),
    warnings = as.list(c(global_warn, unlist(lapply(reports, function(r)
      if (length(r$warnings)) paste0(r$sample_id, ": ", unlist(r$warnings)) else NULL)))),
    object = basename(file.path(out_dir, paste0("object.", if (identical(cfg$object_format, "qs2") && scp_has_pkg("qs2")) "qs2" else "rds"))),
    peak_memory_gb = scp_peak_mem_gb(),
    packages = list(R = paste(R.version$major, R.version$minor, sep = "."),
                    Seurat = scp_pkg_version("Seurat"), SeuratObject = scp_pkg_version("SeuratObject"),
                    BPCells = scp_pkg_version("BPCells"), qs2 = scp_pkg_version("qs2"))
  )
  scp_write_json(report, file.path(out_dir, "ingest_report.json"))
  scp_log(sprintf("S0 done: %d cells x %d genes (key: %s)", ncol(counts), nrow(counts), key_name))
  invisible(report)
}
