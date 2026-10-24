# cellwarden stage S3 (R half): per-cluster evidence (docs/agents-spec.md section 4).
#
# Input:   S2/object.rds (or .qs2), clusters in `seurat_clusters`
#          config/samples.csv (sample sheet; only for the metadata hint)
# Outputs: S3/markers.tsv.gz        cluster, gene, avg_log2fc, pct_in, pct_out, p_adj, rank
#                                   (top 50 positive markers per cluster; `gene` is the
#                                   original symbol, gene_symbol_original)
#          S3/cluster_summary.json  {"<cluster>": {n_cells, pct_cells, per_sample, median_genes,
#                                   median_umis, median_pct_mito, top_markers:[20 genes]}}
#          S3/signatures.tsv.gz     cluster, signature, category, cell_type, mean_score,
#                                   median_score, pct_positive, n_genes_set, n_genes_found
#          S3/cell_clusters.tsv.gz  cell_id, cluster, sample_id (per cell; used by the Python
#                                   side to aggregate CellTypist votes; not part of the bundle)
#          S3/s3_r_summary.json     method notes, signature-set coverage, sample metadata
#                                   hint (sheet extra columns, unique values only), timings
#          S3/exchange/counts.h5ad  slim counts for CellTypist, written only when
#                                   annotation.celltypist != off and Python can import
#                                   celltypist (deleted by s3_celltypist.py when done)
#
# Markers: presto::wilcoxauc when presto is installed, otherwise
# Seurat::FindAllMarkers(logfc.threshold = 0.25, min.pct = 0.1, only.pos = TRUE)
# on the `data` layer. Both paths keep genes with p_adj < 0.05 (if a cluster has
# fewer than 5 such genes, its best 50 by p_adj are kept instead and the cluster is
# listed under `lenient_clusters`), ranked by avg_log2fc (descending), top 50.
# presto's logFC (difference of mean log1p expression) is divided by log(2) so it
# is on a log2 scale like Seurat's avg_log2FC; the two are close but not identical.
#
# Signatures: Seurat::AddModuleScore per set with nbin/ctrl scaled down for small
# gene universes; if it fails for a set, the fallback score is the mean
# log-normalised expression of the set's genes minus the cell's mean over all
# genes (`method` per set in s3_r_summary.json). Sets with no gene in the data
# get NA scores.

.S3_TOP_MARKERS <- 50L
.S3_TOP_SUMMARY <- 20L
.S3_LOGFC_MIN <- 0.25
.S3_MINPCT <- 0.1
.S3_PADJ <- 0.05

.S3_ANNOTATION_DEFAULTS <- list(celltypist_model = "Immune_All_Low.pkl", celltypist = "auto",
                                min_confidence = 0.6, models_dir = NULL)

s3_annotation_cfg <- function(cfg) {
  a <- cfg$annotation
  if (!is.list(a)) a <- list()
  for (k in names(.S3_ANNOTATION_DEFAULTS)) if (is.null(a[[k]])) a[[k]] <- .S3_ANNOTATION_DEFAULTS[[k]]
  if (!(a$celltypist %in% c("auto", "off"))) scp_stop("annotation.celltypist must be auto or off (got '", a$celltypist, "')")
  a
}

# --------------------------------------------------------------------------
# Signatures
# --------------------------------------------------------------------------

#' Read the signature YAML (config `signatures` overrides the built-in file).
#' Returns a named list: name -> list(genes, category, cell_type).
s3_read_signatures <- function(cfg, repo) {
  path <- cfg$signatures
  if (!(is.character(path) && length(path) == 1 && nzchar(path))) path <- file.path(repo, "R", "stages", "data", "signatures.yaml")
  if (!file.exists(path)) scp_stop("signature file not found: ", path)
  y <- yaml::read_yaml(path)
  sets <- y$sets %||% y
  if (!is.list(sets) || !length(sets)) scp_stop("signature file ", path, " has no 'sets'")
  out <- list()
  for (nm in names(sets)) {
    s <- sets[[nm]]
    genes <- if (is.list(s)) s$genes else s
    genes <- unique(as.character(unlist(genes)))
    if (!length(genes)) scp_stop("signature '", nm, "' has no genes")
    out[[nm]] <- list(genes = genes, category = as.character((if (is.list(s)) s$category else NULL) %||% "lineage"),
                      cell_type = as.character((if (is.list(s)) s$cell_type else NULL) %||% NA_character_))
  }
  attr(out, "path") <- normalizePath(path)
  out
}

# Per-cell scores for one set; returns list(score = numeric(ncells) | NULL, method, n_found).
.s3_score_set <- function(obj, features, seed, nbin, ctrl) {
  if (!length(features)) return(list(score = NULL, method = "none", n_found = 0L))
  res <- tryCatch({
    set.seed(seed)
    o <- suppressWarnings(Seurat::AddModuleScore(obj, features = list(features), name = "scp_sig_", nbin = nbin,
                                                 ctrl = ctrl, seed = seed, search = FALSE))
    list(score = as.numeric(o[["scp_sig_1", drop = TRUE]]), method = "AddModuleScore")
  }, error = function(e) NULL)
  if (!is.null(res)) return(c(res, list(n_found = length(features))))
  x <- SeuratObject::LayerData(obj, layer = "data")
  s <- Matrix::colMeans(x[features, , drop = FALSE]) - Matrix::colMeans(x)
  list(score = as.numeric(s), method = "mean_expression_fallback", n_found = length(features))
}

s3_signature_scores <- function(obj, sigs, symbols, seed) {
  n_genes <- nrow(obj)
  nbin <- max(2L, min(24L, n_genes %/% 40L))
  ctrl <- max(5L, min(100L, (n_genes %/% nbin) %/% 2L))
  cl <- as.character(obj$seurat_clusters)
  cl_levels <- levels(obj$seurat_clusters) %||% sort(unique(cl))
  rows <- list()
  meta <- list()
  for (nm in names(sigs)) {
    s <- sigs[[nm]]
    feats <- rownames(obj)[match(s$genes, symbols)]
    feats <- feats[!is.na(feats)]
    sc <- .s3_score_set(obj, feats, seed, nbin, ctrl)
    meta[[nm]] <- list(category = s$category, cell_type = s$cell_type, n_genes_set = length(s$genes),
                       n_genes_found = length(feats), genes_found = feats, method = sc$method)
    for (k in cl_levels) {
      v <- if (is.null(sc$score)) numeric() else sc$score[cl == k]
      rows[[length(rows) + 1L]] <- data.frame(
        cluster = k, signature = nm, category = s$category, cell_type = s$cell_type,
        mean_score = if (length(v)) mean(v) else NA_real_,
        median_score = if (length(v)) stats::median(v) else NA_real_,
        pct_positive = if (length(v)) 100 * mean(v > 0) else NA_real_,
        n_genes_set = length(s$genes), n_genes_found = length(feats), stringsAsFactors = FALSE)
    }
  }
  df <- do.call(rbind, rows)
  df$mean_score <- round(df$mean_score, 6); df$median_score <- round(df$median_score, 6); df$pct_positive <- round(df$pct_positive, 3)
  list(table = df, meta = meta, nbin = nbin, ctrl = ctrl)
}

# --------------------------------------------------------------------------
# Markers
# --------------------------------------------------------------------------

#' Markers with presto (if installed) or FindAllMarkers. Returns a data frame
#' cluster, gene (feature name), avg_log2fc, pct_in, pct_out, p_adj (all genes
#' passing the logFC / min.pct filters, every cluster), plus attr "method".
s3_all_markers <- function(obj, seed) {
  cl <- obj$seurat_clusters
  if (scp_has_pkg("presto")) {
    x <- SeuratObject::LayerData(obj, layer = "data")
    r <- presto::wilcoxauc(X = x, y = as.character(cl))
    df <- data.frame(cluster = as.character(r$group), gene = as.character(r$feature),
                     avg_log2fc = r$logFC / log(2), pct_in = r$pct_in, pct_out = r$pct_out,
                     p_adj = r$padj, stringsAsFactors = FALSE)
    df <- df[df$avg_log2fc >= .S3_LOGFC_MIN & pmax(df$pct_in, df$pct_out) >= 100 * .S3_MINPCT, , drop = FALSE]
    attr(df, "method") <- paste0("presto::wilcoxauc ", scp_pkg_version("presto"), " (logFC / log 2)")
    return(df)
  }
  SeuratObject::Idents(obj) <- cl
  set.seed(seed)
  m <- suppressWarnings(Seurat::FindAllMarkers(obj, logfc.threshold = .S3_LOGFC_MIN, min.pct = .S3_MINPCT,
                                                only.pos = TRUE, verbose = FALSE))
  if (!nrow(m)) {
    df <- data.frame(cluster = character(), gene = character(), avg_log2fc = numeric(), pct_in = numeric(),
                     pct_out = numeric(), p_adj = numeric(), stringsAsFactors = FALSE)
  } else {
    df <- data.frame(cluster = as.character(m$cluster), gene = as.character(m$gene), avg_log2fc = m$avg_log2FC,
                     pct_in = 100 * m$pct.1, pct_out = 100 * m$pct.2, p_adj = m$p_val_adj, stringsAsFactors = FALSE)
  }
  attr(df, "method") <- paste0("Seurat::FindAllMarkers(wilcox, logfc.threshold = ", .S3_LOGFC_MIN, ", min.pct = ",
                               .S3_MINPCT, ", only.pos = TRUE) Seurat ", scp_pkg_version("Seurat"))
  df
}

#' Top N per cluster: p_adj < 0.05 (lenient when fewer than 5), ranked by avg_log2fc.
s3_top_markers <- function(df, cl_levels, n = .S3_TOP_MARKERS) {
  out <- list(); lenient <- character()
  for (k in cl_levels) {
    d <- df[df$cluster == k & df$avg_log2fc > 0, , drop = FALSE]
    sig <- d[!is.na(d$p_adj) & d$p_adj < .S3_PADJ, , drop = FALSE]
    if (nrow(sig) < 5 && nrow(d) > nrow(sig)) {
      lenient <- c(lenient, k)
      d <- d[order(d$p_adj, -d$avg_log2fc), , drop = FALSE]
      d <- head(d, n)
    } else d <- sig
    d <- d[order(-d$avg_log2fc, d$p_adj, d$gene), , drop = FALSE]
    d <- head(d, n)
    if (nrow(d)) d$rank <- seq_len(nrow(d))
    else d$rank <- integer()
    out[[k]] <- d
  }
  res <- do.call(rbind, c(out, list(make.row.names = FALSE)))
  if (is.null(res)) res <- data.frame(cluster = character(), gene = character(), avg_log2fc = numeric(), pct_in = numeric(),
                                      pct_out = numeric(), p_adj = numeric(), rank = integer(), stringsAsFactors = FALSE)
  rownames(res) <- NULL
  attr(res, "lenient") <- lenient
  res
}

# --------------------------------------------------------------------------
# Sample metadata hint: sheet columns beyond sample_id/path/format, unique values only.
# --------------------------------------------------------------------------

s3_sample_metadata_hint <- function(cfg) {
  f <- file.path(cfg$run_dir, "config", "samples.csv")
  if (!file.exists(f)) return(structure(list(), names = character()))
  df <- tryCatch(utils::read.csv(f, colClasses = "character", check.names = FALSE, na.strings = character()),
                 error = function(e) NULL)
  if (is.null(df)) return(structure(list(), names = character()))
  extra <- setdiff(colnames(df), c("sample_id", "path", "format"))
  hint <- list()
  for (cn in extra) {
    v <- unique(trimws(as.character(df[[cn]])))
    v <- v[nzchar(v)]
    hint[[cn]] <- I(sort(v))
  }
  if (!length(hint)) return(structure(list(), names = character()))
  hint
}

# --------------------------------------------------------------------------
# CellTypist input (only when the Python side can use it)
# --------------------------------------------------------------------------

.s3_write_celltypist_input <- function(obj, cfg, repo, out_dir, ann) {
  if (identical(ann$celltypist, "off")) return(list(written = FALSE, reason = "annotation.celltypist is off"))
  if (!s2_python_has(repo, "celltypist")) return(list(written = FALSE, reason = "celltypist is not importable by python"))
  ex_dir <- file.path(out_dir, "exchange")
  dir.create(ex_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(ex_dir, "counts.h5ad")
  unlink(path)
  write_exchange(obj, path, mode = "full", run_id = cfg$run_id, stage = "S3", config_hash = cfg$config_hash,
                 reductions = character(), graphs = FALSE)
  list(written = TRUE, path = path)
}

# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------

run_s3 <- function(cfg, repo) {
  out_dir <- scp_stage_dir(cfg, "S3")
  seed <- cfg$seed
  set.seed(seed)
  ann <- s3_annotation_cfg(cfg)
  timings <- list()
  tm <- function(label, expr) {
    r <- scp_timed(label, expr)
    timings[[label]] <<- r$seconds
    r$value
  }

  obj <- tm("load", scp_load_object(file.path(cfg$run_dir, "S2")))
  if (!("seurat_clusters" %in% colnames(obj[[]]))) scp_stop("S2 object has no seurat_clusters column")
  SeuratObject::DefaultAssay(obj) <- "RNA"
  if (!("data" %in% SeuratObject::Layers(obj[["RNA"]]))) scp_stop("S2 object has no 'data' layer in assay RNA")
  cl <- obj$seurat_clusters
  if (!is.factor(cl)) cl <- factor(as.character(cl), levels = unique(as.character(cl)))
  obj$seurat_clusters <- cl
  cl_levels <- levels(cl)
  cl_chr <- as.character(cl)
  n_cells <- ncol(obj)
  symbols <- scp_symbols(obj)
  scp_log(sprintf("S3: %d cells, %d genes, %d clusters", n_cells, nrow(obj), length(cl_levels)))

  # markers
  all_mk <- tm("markers", s3_all_markers(obj, seed))
  method <- attr(all_mk, "method")
  scp_log("markers: ", method, "; ", nrow(all_mk), " cluster x gene rows pass the filters")
  top <- s3_top_markers(all_mk, cl_levels)
  lenient <- attr(top, "lenient")
  top$gene <- symbols[match(top$gene, rownames(obj))]
  top$avg_log2fc <- round(top$avg_log2fc, 5); top$pct_in <- round(top$pct_in, 3); top$pct_out <- round(top$pct_out, 3)
  top$p_adj <- signif(top$p_adj, 6)
  top <- top[, c("cluster", "gene", "avg_log2fc", "pct_in", "pct_out", "p_adj", "rank")]
  scp_write_tsv_gz(top, file.path(out_dir, "markers.tsv.gz"))

  # cluster summary
  counts <- scp_counts(obj)
  mito <- scp_gene_flags(symbols)$mito
  umis <- Matrix::colSums(counts)
  pct_mito <- if (any(mito)) 100 * Matrix::colSums(counts[mito, , drop = FALSE]) / pmax(umis, 1) else rep(0, n_cells)
  n_genes_cell <- if ("nFeature_RNA" %in% colnames(obj[[]])) as.numeric(obj$nFeature_RNA) else Matrix::colSums(counts > 0)
  n_umi_cell <- if ("nCount_RNA" %in% colnames(obj[[]])) as.numeric(obj$nCount_RNA) else as.numeric(umis)
  sample <- if ("sample_id" %in% colnames(obj[[]])) as.character(obj$sample_id) else rep("sample", n_cells)
  summary <- list()
  for (k in cl_levels) {
    i <- cl_chr == k
    ps <- table(sample[i])
    summary[[k]] <- list(
      n_cells = sum(i), pct_cells = round(100 * sum(i) / n_cells, 4),
      per_sample = as.list(stats::setNames(as.integer(ps), names(ps))),
      median_genes = stats::median(n_genes_cell[i]), median_umis = stats::median(n_umi_cell[i]),
      median_pct_mito = round(stats::median(pct_mito[i]), 4),
      top_markers = I(head(top$gene[top$cluster == k], .S3_TOP_SUMMARY)))
  }
  scp_write_json(summary, file.path(out_dir, "cluster_summary.json"))
  # per-cell cluster map (never part of the bundle): lets the Python side aggregate CellTypist per cluster
  scp_write_tsv_gz(data.frame(cell_id = colnames(obj), cluster = cl_chr, sample_id = sample, stringsAsFactors = FALSE),
                   file.path(out_dir, "cell_clusters.tsv.gz"))

  # signatures
  sigs <- s3_read_signatures(cfg, repo)
  sg <- tm("signatures", s3_signature_scores(obj, sigs, symbols, seed))
  scp_write_tsv_gz(sg$table, file.path(out_dir, "signatures.tsv.gz"))
  missing_sets <- names(sg$meta)[vapply(sg$meta, function(m) m$n_genes_found == 0L, logical(1))]
  if (length(missing_sets)) scp_warn_log("signature set(s) with no gene in the data: ", paste(missing_sets, collapse = ", "))

  # celltypist input
  ct <- tm("celltypist_input", .s3_write_celltypist_input(obj, cfg, repo, out_dir, ann))
  scp_log("celltypist input: ", if (isTRUE(ct$written)) ct$path else paste0("not written (", ct$reason, ")"))

  r_summary <- list(
    stage = "S3", run_id = cfg$run_id, config_hash = cfg$config_hash, created_utc = .scp_utc(),
    n_cells = n_cells, n_genes = nrow(obj), n_clusters = length(cl_levels), clusters = I(cl_levels),
    samples = I(sort(unique(sample))),
    markers = list(method = method, top_n = .S3_TOP_MARKERS, logfc_threshold = .S3_LOGFC_MIN, min_pct = .S3_MINPCT,
                   p_adj_max = .S3_PADJ, lenient_clusters = I(lenient), rows_written = nrow(top)),
    signatures = list(file = attr(sigs, "path"), nbin = sg$nbin, ctrl = sg$ctrl, sets = sg$meta),
    celltypist_input = ct,
    sample_metadata_hint = s3_sample_metadata_hint(cfg),
    timings_s = timings, total_s = round(sum(unlist(timings)), 3),
    peak_memory = list(r_peak_rss_gb = scp_peak_mem_gb()),
    packages = list(R = paste(R.version$major, R.version$minor, sep = "."), Seurat = scp_pkg_version("Seurat"),
                    presto = scp_pkg_version("presto"))
  )
  scp_write_json(r_summary, file.path(out_dir, "s3_r_summary.json"))
  scp_log(sprintf("S3 (R) done: %d marker rows, %d signature sets, %d clusters", nrow(top), length(sigs), length(cl_levels)))
  invisible(r_summary)
}
