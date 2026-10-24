# cellwarden stage S4: differential expression (docs/agents-spec.md section 6,
# DESIGN.md 6.1). The Open Targets half of S4 is Python
# (python/cellwarden/stages/s4_opentargets.py) and runs after this step.
#
# Two engines (dge.method):
# * findmarkers (default): Seurat::FindMarkers between two sets of cells, with
#   dge.test (MAST by default; wilcox, wilcox_limma, bimod, t, negbinom,
#   poisson, LR, DESeq2). Each contrast names a cell metadata column (any
#   samples.csv column, cell_type_broad / cell_type_fine, seurat_clusters, ...),
#   the values on each side (numerator, denominator; a list, or "rest" for every
#   other value), and `within`: the column whose groups are tested separately
#   (default dge.group_by; "none" tests all cells together). Covariates become
#   latent.vars for MAST, LR, negbinom and poisson. With no contrasts and
#   dge.markers auto/true, the markers of each group_by group against the rest
#   of the cells are found (contrast "markers"), so a one-sample study still gets
#   DE genes and Open Targets evidence.
# * pseudobulk (or the older values deseq2 / limma_voom): counts summed per
#   group x sample and tested with DESeq2 or limma-voom (dge.pseudobulk_method);
#   the contrast column must be constant within each sample (condition, donor,
#   sex, ...), and each side needs min_replicates samples.
#
# FindMarkers tables keep the pseudobulk columns: baseMean is the mean
# normalised expression of the group's cells, lfcSE is empty, log2fc is
# avg_log2FC, padj is p_val_adj (Bonferroni, Seurat's), plus pct_1 / pct_2
# (fraction of cells expressing the gene on each side).
#
# Inputs:  S2/object.rds (or .qs2); optional gates/G3/decision.json or
#          S3/annotation.tsv (cell-type labels)
# Outputs: S3/annotation.tsv                      (written from gates/G3/decision.json when it exists)
#          S4/gene_map.tsv                        gene, symbol, ensembl_id for every feature
#          S4/pseudobulk/<contrast>/counts.tsv.gz gene x pseudobulk unit (group x sample) counts
#          S4/pseudobulk/<contrast>/samples.tsv   one row per unit: pb_id, group, sample, condition,
#                                                 covariates, n_cells, included, exclude_reason
#          S4/dge/<contrast>__<group>.tsv.gz      gene, baseMean, log2fc, lfcSE, pvalue, padj, direction
#          S4/dge_summary.json                    per contrast x group: n_up, n_down, n_tested,
#                                                 skipped_reason, top 10 up/down, and the choices made
#
# Choices not fixed by the spec (documented here):
# * Labels: if gates/G3/decision.json exists its clusters are written to
#   S3/annotation.tsv and added to the object as cell_type_broad / cell_type_fine.
#   Without it an existing S3/annotation.tsv is used; without either, group_by
#   falls back to seurat_clusters with a note in dge_summary.json.
#   `group_by: cluster` means seurat_clusters.
# * Pseudobulk unit = group_by x sample_col. The two columns are combined into
#   one temporary metadata column before Seurat::AggregateExpression so that
#   Seurat's renaming of group names (underscores to dashes, a "g" prefix for
#   names starting with a digit) can never merge or mislabel units; the
#   grouping is still group_by x sample_col. If AggregateExpression refuses
#   (e.g. a single unit) the sums are computed directly, which gives the same
#   matrix.
# * Units with fewer than min_cells_per_sample cells are dropped before
#   counting replicates. A group needs min_replicates units on each side of
#   the contrast, otherwise it is skipped with skipped_reason.
# * Design: ~ covariates + condition, condition levels (denominator,
#   numerator). A covariate that has one level within a group, or that makes
#   the design rank deficient or leaves no residual degrees of freedom, is
#   dropped for that group (recorded in covariates_used / notes).
# * Genes with fewer than 10 summed counts across the group's units are not
#   tested (n_tested counts the rest); they are absent from the DGE table.
# * method deseq2: DESeq2 with Wald test, results(contrast = c(condition,
#   numerator, denominator)). method limma_voom (also the fallback when
#   DESeq2 is not installed, or when DESeq2 fails on a group): limma::voom with
#   effective library sizes from median-of-ratios size factors (computed here;
#   no edgeR TMM factors, edgeR is not required), lmFit + eBayes; baseMean is
#   the mean size-factor-normalised count, lfcSE = sqrt(s2.post) * stdev.unscaled.
# * direction: up if padj < dge.padj and log2fc >= dge.lfc; down if padj < dge.padj
#   and log2fc <= -dge.lfc; otherwise ns (NA padj counts as ns).

.S4_DEFAULTS <- list(
  sample_col = "sample_id", condition_col = "condition", group_by = "cell_type_broad",
  contrasts = list(), min_cells_per_sample = 20L, min_replicates = 2L,
  method = "findmarkers", test = "MAST", pseudobulk_method = "deseq2", markers = "auto",
  min_pct = 0.1, logfc_threshold = 0.25, min_cells_per_side = 10L, max_cells_per_side = NA,
  padj = 0.05, lfc = 0.5
)
.S4_METHODS <- c("findmarkers", "pseudobulk", "deseq2", "limma_voom")
.S4_TESTS <- c("MAST", "wilcox", "wilcox_limma", "bimod", "t", "negbinom", "poisson", "LR", "DESeq2")
.S4_LATENT_TESTS <- c("MAST", "LR", "negbinom", "poisson")
.S4_MIN_GENE_COUNTS <- 10
.S4_TOP_N <- 10L
.S4_DGE_COLS <- c("gene", "baseMean", "log2fc", "lfcSE", "pvalue", "padj", "direction")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

s4_config <- function(cfg) {
  d <- .scp_fill(cfg$dge %||% list(), .S4_DEFAULTS)
  if (is.null(d$contrasts)) d$contrasts <- list()
  p <- character()
  chk <- function(ok, msg) if (!isTRUE(ok)) p <<- c(p, msg)
  str1 <- function(v) is.character(v) && length(v) == 1 && nzchar(v)
  num1 <- function(v) is.numeric(v) && length(v) == 1 && !is.na(v)
  for (k in c("sample_col", "condition_col", "group_by")) chk(str1(d[[k]]), paste0("dge.", k, " must be a non-empty string"))
  chk(str1(d$method) && d$method %in% .S4_METHODS, paste0("dge.method must be one of ", paste(.S4_METHODS, collapse = ", ")))
  chk(str1(d$test) && d$test %in% .S4_TESTS, paste0("dge.test must be one of ", paste(.S4_TESTS, collapse = ", ")))
  chk(str1(d$pseudobulk_method) && d$pseudobulk_method %in% c("deseq2", "limma_voom"),
      "dge.pseudobulk_method must be deseq2 or limma_voom")
  mk <- d$markers
  chk((is.logical(mk) && length(mk) == 1 && !is.na(mk)) || identical(mk, "auto"), "dge.markers must be auto, true or false")
  chk(num1(d$min_pct) && d$min_pct >= 0 && d$min_pct <= 1, "dge.min_pct must be in [0, 1]")
  chk(num1(d$logfc_threshold) && d$logfc_threshold >= 0, "dge.logfc_threshold must be >= 0")
  chk(num1(d$min_cells_per_side) && d$min_cells_per_side >= 1, "dge.min_cells_per_side must be >= 1")
  chk(is.null(d$max_cells_per_side) || is.na(d$max_cells_per_side) || (num1(d$max_cells_per_side) && d$max_cells_per_side >= 1),
      "dge.max_cells_per_side must be a number >= 1 or null")
  chk(num1(d$padj) && d$padj > 0 && d$padj <= 1, "dge.padj must be in (0, 1]")
  chk(num1(d$lfc) && d$lfc >= 0, "dge.lfc must be >= 0")
  chk(num1(d$min_cells_per_sample) && d$min_cells_per_sample >= 1, "dge.min_cells_per_sample must be >= 1")
  chk(num1(d$min_replicates) && d$min_replicates >= 1, "dge.min_replicates must be >= 1")
  chk(is.list(d$contrasts), "dge.contrasts must be a list")
  if (is.list(d$contrasts)) {
    for (i in seq_along(d$contrasts)) {
      ct <- d$contrasts[[i]]
      w <- paste0("dge.contrasts[", i, "]")
      if (!is.list(ct)) { chk(FALSE, paste0(w, " must be a mapping")); next }
      chk(str1(ct$name), paste0(w, ".name must be a non-empty string"))
      for (k in c("numerator", "denominator")) {
        v <- unlist(ct[[k]])
        chk(is.character(v) && length(v) >= 1 && all(nzchar(v)), paste0(w, ".", k, " must be a value or a list of values"))
        d$contrasts[[i]][[k]] <- as.character(v)
      }
      if (identical(d$contrasts[[i]]$numerator, "rest")) chk(FALSE, paste0(w, ".numerator cannot be 'rest' (use it for the denominator)"))
      for (k in c("column", "within")) if (!is.null(ct[[k]])) chk(str1(ct[[k]]), paste0(w, ".", k, " must be a column name"))
      if (str1(ct$name)) chk(grepl("^[A-Za-z0-9][A-Za-z0-9._-]*$", ct$name), paste0(w, ".name '", ct$name, "' has characters not allowed in file names"))
      if (!is.null(ct$covariates)) {
        cv <- unlist(ct$covariates)
        chk(is.character(cv) && all(nzchar(cv)), paste0(w, ".covariates must be a list of column names"))
        d$contrasts[[i]]$covariates <- as.character(cv)
      } else d$contrasts[[i]]$covariates <- character()
    }
    nm <- vapply(d$contrasts, function(ct) as.character(ct$name %||% ""), "")
    dup <- unique(nm[duplicated(nm) & nzchar(nm)])
    if (length(dup)) chk(FALSE, paste0("dge.contrasts: duplicated name(s) ", paste(dup, collapse = ", ")))
  }
  if (length(p)) scp_stop("invalid dge config:\n  ", paste(p, collapse = "\n  "))
  d$min_cells_per_sample <- as.integer(d$min_cells_per_sample)
  d$min_replicates <- as.integer(d$min_replicates)
  d$min_cells_per_side <- as.integer(d$min_cells_per_side)
  # engine: findmarkers, or pseudobulk with deseq2 / limma_voom (method deseq2 | limma_voom name the
  # pseudobulk method directly, as before)
  d$engine <- if (d$method == "findmarkers") "findmarkers" else "pseudobulk"
  if (d$method %in% c("deseq2", "limma_voom")) d$pseudobulk_method <- d$method
  for (i in seq_along(d$contrasts)) {
    d$contrasts[[i]]$column <- d$contrasts[[i]]$column %||% d$condition_col
    d$contrasts[[i]]$within <- d$contrasts[[i]]$within %||% d$group_by
  }
  d
}

#' Side of each cell / unit: "num", "den" or NA from its value of the contrast column.
s4_sides <- function(values, numerator, denominator) {
  v <- as.character(values)
  side <- rep(NA_character_, length(v))
  side[v %in% numerator] <- "num"
  if (identical(denominator, "rest")) side[is.na(side) & !is.na(v)] <- "den"
  else side[v %in% denominator] <- "den"
  side
}

.s4_label <- function(x) paste(x, collapse = "+")

#' Check that a contrast's values exist in its column; stop with the available values otherwise.
s4_check_values <- function(ct, values) {
  lv <- sort(unique(as.character(values[!is.na(values)])))
  for (k in c("numerator", "denominator")) {
    want <- ct[[k]]
    if (k == "denominator" && identical(want, "rest")) next
    bad <- setdiff(want, lv)
    if (length(bad)) scp_stop("dge.contrasts '", ct$name, "': ", k, " '", paste(bad, collapse = "', '"), "' is not a value of '",
                              ct$column, "' (values: ", paste(utils::head(lv, 30), collapse = ", "),
                              if (length(lv) > 30) ", ..." else "", ")")
  }
  if (!length(setdiff(ct$numerator, ct$denominator)) && !identical(ct$denominator, "rest"))
    scp_stop("dge.contrasts '", ct$name, "': numerator and denominator are the same")
  lv
}

# --------------------------------------------------------------------------
# Labels (G3 decision -> S3/annotation.tsv -> cell_type_broad / cell_type_fine)
# --------------------------------------------------------------------------

.s4_chr <- function(x, default = NA_character_) {
  if (is.null(x) || length(x) == 0) return(default)
  v <- as.character(x[[1]])
  if (is.na(v)) default else v
}

#' Cluster table from gates/G3/decision.json ({"decision": {"clusters": [...]}}
#' or, tolerated, {"clusters": [...]}); columns cluster, label_broad,
#' label_fine, cl_id, confidence, status.
s4_annotation_from_decision <- function(path) {
  j <- tryCatch(jsonlite::fromJSON(path, simplifyVector = FALSE),
                error = function(e) scp_stop("cannot parse ", path, ": ", conditionMessage(e)))
  cl <- j$decision$clusters %||% j$clusters
  if (!is.list(cl) || !length(cl)) scp_stop(path, " has no decision.clusters entries")
  df <- do.call(rbind, lapply(cl, function(x) data.frame(
    cluster = .s4_chr(x$cluster), label_broad = .s4_chr(x$label_broad), label_fine = .s4_chr(x$label_fine),
    cl_id = .s4_chr(x$cl_id, ""), confidence = suppressWarnings(as.numeric(.s4_chr(x$confidence))),
    status = .s4_chr(x$status, ""), stringsAsFactors = FALSE)))
  if (any(is.na(df$cluster))) scp_stop(path, ": a cluster entry has no 'cluster'")
  if (anyDuplicated(df$cluster)) scp_stop(path, ": duplicated cluster(s) ", paste(unique(df$cluster[duplicated(df$cluster)]), collapse = ", "))
  df$label_broad[is.na(df$label_broad)] <- "unassigned"
  df$label_fine[is.na(df$label_fine)] <- df$label_broad[is.na(df$label_fine)]
  df
}

.s4_write_tsv <- function(df, path) {
  .scp_atomic(path, function(p) utils::write.table(df, p, sep = "\t", quote = FALSE, row.names = FALSE, na = ""))
}

#' Apply labels to the object. Returns list(obj, source, note, n_unassigned).
s4_apply_labels <- function(obj, run_dir) {
  dec <- file.path(run_dir, "gates", "G3", "decision.json")
  ann <- file.path(run_dir, "S3", "annotation.tsv")
  df <- NULL; source <- "none"; note <- NULL
  if (file.exists(dec)) {
    df <- s4_annotation_from_decision(dec)
    .s4_write_tsv(df, ann)
    source <- "gates/G3/decision.json"
    scp_log("labels: wrote S3/annotation.tsv from gates/G3/decision.json (", nrow(df), " clusters)")
  } else if (file.exists(ann)) {
    df <- scp_read_tsv(ann)
    need <- c("cluster", "label_broad")
    miss <- setdiff(need, colnames(df))
    if (length(miss)) scp_stop(ann, " is missing column(s): ", paste(miss, collapse = ", "))
    if (!("label_fine" %in% colnames(df))) df$label_fine <- df$label_broad
    source <- "S3/annotation.tsv"
    scp_log("labels: gates/G3/decision.json not found; using existing S3/annotation.tsv (", nrow(df), " clusters)")
  } else {
    note <- "no gates/G3/decision.json or S3/annotation.tsv: cell-type labels are not available"
    scp_warn_log("labels: ", note)
    return(list(obj = obj, source = source, note = note, n_unassigned = NA_integer_))
  }
  if (!("seurat_clusters" %in% colnames(obj[[]]))) scp_stop("the S2 object has no seurat_clusters column")
  cl <- as.character(obj$seurat_clusters)
  idx <- match(cl, as.character(df$cluster))
  n_un <- sum(is.na(idx))
  broad <- df$label_broad[idx]; fine <- df$label_fine[idx]
  broad[is.na(idx)] <- "unassigned"; fine[is.na(idx)] <- "unassigned"
  if (n_un > 0) {
    note <- sprintf("%d cell(s) in %d cluster(s) have no label in %s; labelled 'unassigned'", n_un,
                    length(unique(cl[is.na(idx)])), source)
    scp_warn_log("labels: ", note)
  }
  obj$cell_type_broad <- factor(broad)
  obj$cell_type_fine <- factor(fine)
  list(obj = obj, source = source, note = note, n_unassigned = as.integer(n_un))
}

# --------------------------------------------------------------------------
# Pseudobulk
# --------------------------------------------------------------------------

#' Sum counts per group x sample unit. Returns list(counts = dgCMatrix genes x
#' units, units = data.frame(pb_id, group, sample, n_cells), method).
s4_pseudobulk <- function(obj, group, sample, assay = "RNA") {
  group <- as.character(group); sample <- as.character(sample)
  if (length(group) != ncol(obj) || length(sample) != ncol(obj)) scp_stop("pseudobulk: group/sample vectors do not match the object")
  key <- paste(group, sample, sep = "\r")
  uk <- unique(key)
  units <- data.frame(pb_id = sprintf("pb%04d", seq_along(uk)),
                      group = group[match(uk, key)], sample = sample[match(uk, key)],
                      n_cells = as.integer(table(factor(key, levels = uk))), stringsAsFactors = FALSE)
  units <- units[order(units$group, units$sample), , drop = FALSE]
  units$pb_id <- sprintf("pb%04d", seq_len(nrow(units)))
  rownames(units) <- NULL
  idx <- match(key, paste(units$group, units$sample, sep = "\r"))
  pb_of_cell <- units$pb_id[idx]
  agg <- NULL; method <- "Seurat::AggregateExpression"
  if (nrow(units) > 1) {
    obj$cellwarden_pb <- factor(pb_of_cell, levels = units$pb_id)
    agg <- tryCatch(suppressMessages(
      # AggregateExpression always sums the counts layer; passing layer= is a
      # duplicate-argument error in Seurat 5.0.x.
      Seurat::AggregateExpression(obj, assays = assay, group.by = "cellwarden_pb", verbose = FALSE)[[assay]]),
      error = function(e) { scp_warn_log("AggregateExpression failed (", conditionMessage(e), "); summing counts directly"); NULL })
    if (!is.null(agg) && (!all(units$pb_id %in% colnames(agg)) || nrow(agg) != nrow(obj[[assay]]))) {
      scp_warn_log("AggregateExpression returned unexpected dimensions; summing counts directly")
      agg <- NULL
    }
  }
  if (is.null(agg)) {
    m <- scp_counts(obj, assay)
    ind <- Matrix::sparseMatrix(i = seq_len(ncol(m)), j = match(pb_of_cell, units$pb_id), x = 1,
                                dims = c(ncol(m), nrow(units)), dimnames = list(colnames(m), units$pb_id))
    agg <- m %*% ind
    method <- "direct sum"
  }
  agg <- scp_as_dgc(agg)[, units$pb_id, drop = FALSE]
  rownames(agg) <- rownames(obj[[assay]])
  list(counts = agg, units = units, method = method)
}

.s4_safe_name <- function(x) {
  s <- gsub("[^A-Za-z0-9.-]+", "-", as.character(x))
  s <- gsub("^-+|-+$", "", s)
  s[!nzchar(s)] <- "group"
  s
}

# --------------------------------------------------------------------------
# DGE methods
# --------------------------------------------------------------------------

#' Build the design for one group. Returns list(formula, covariates, notes,
#' ok, reason).
s4_design <- function(coldata, cond_col, covariates, min_replicates) {
  notes <- character()
  covs <- covariates
  # drop covariates with a single level in this group
  single <- covs[vapply(covs, function(cv) length(unique(coldata[[cv]])) < 2, TRUE)]
  if (length(single)) {
    notes <- c(notes, paste0("covariate(s) ", paste(single, collapse = ", "), " have one level in this group and were dropped"))
    covs <- setdiff(covs, single)
  }
  fml <- function(cv) stats::as.formula(paste("~", paste(c(cv, cond_col), collapse = " + ")))
  repeat {
    X <- stats::model.matrix(fml(covs), coldata)
    rank_ok <- qr(X)$rank == ncol(X)
    df_ok <- nrow(X) - ncol(X) >= 1
    if (rank_ok && df_ok) break
    if (!length(covs)) {
      return(list(ok = FALSE, reason = if (!rank_ok) "design is not full rank" else "no residual degrees of freedom",
                  covariates = covs, notes = notes, formula = fml(covs)))
    }
    dropped <- covs[length(covs)]
    notes <- c(notes, paste0("covariate ", dropped, " dropped: ", if (!rank_ok) "design not full rank" else "no residual degrees of freedom"))
    covs <- covs[-length(covs)]
  }
  list(ok = TRUE, reason = NULL, covariates = covs, notes = notes, formula = fml(covs))
}

s4_run_deseq2 <- function(counts, coldata, formula, cond_col, numerator, denominator, alpha) {
  if (!scp_has_pkg("DESeq2")) scp_stop("DESeq2 is not installed")
  storage.mode(counts) <- "integer"
  dds <- DESeq2::DESeqDataSetFromMatrix(countData = counts, colData = coldata, design = formula)
  dds <- suppressMessages(DESeq2::DESeq(dds, quiet = TRUE))
  res <- DESeq2::results(dds, contrast = c(cond_col, numerator, denominator), alpha = alpha)
  data.frame(gene = rownames(res), baseMean = as.numeric(res$baseMean), log2fc = as.numeric(res$log2FoldChange),
             lfcSE = as.numeric(res$lfcSE), pvalue = as.numeric(res$pvalue), padj = as.numeric(res$padj),
             stringsAsFactors = FALSE)
}

#' Median-of-ratios size factors (DESeq2's estimateSizeFactors, written out so
#' the limma path needs neither DESeq2 nor edgeR). Robust to a few very highly
#' expressed genes shifting the library composition, which plain library-size
#' scaling is not. Falls back to library size when fewer than 10 genes are
#' non-zero in every sample.
s4_size_factors <- function(counts) {
  lib <- colSums(counts)
  pos <- rowSums(counts > 0) == ncol(counts)
  if (sum(pos) < 10) return(lib / exp(mean(log(lib))))
  lg <- log(counts[pos, , drop = FALSE])
  gm <- rowMeans(lg)
  sf <- apply(lg, 2, function(col) exp(stats::median(col - gm)))
  sf / exp(mean(log(sf)))
}

s4_run_limma_voom <- function(counts, coldata, formula, cond_col, numerator, denominator) {
  if (!scp_has_pkg("limma")) scp_stop("limma is not installed")
  sf <- s4_size_factors(counts)
  lib <- sf * mean(colSums(counts))      # effective library sizes (geometric mean of sf is 1)
  design <- stats::model.matrix(formula, coldata)
  coef <- paste0(cond_col, numerator)
  if (!(coef %in% colnames(design))) scp_stop("limma: coefficient ", coef, " not in the design (", paste(colnames(design), collapse = ", "), ")")
  v <- limma::voom(counts, design, lib.size = lib)
  fit <- limma::eBayes(limma::lmFit(v, design))
  tt <- limma::topTable(fit, coef = coef, number = Inf, sort.by = "none")
  lfcSE <- sqrt(fit$s2.post) * fit$stdev.unscaled[, coef]
  norm <- sweep(counts, 2, sf, "/")
  data.frame(gene = rownames(tt), baseMean = as.numeric(rowMeans(norm)[rownames(tt)]), log2fc = as.numeric(tt$logFC),
             lfcSE = as.numeric(lfcSE[rownames(tt)]), pvalue = as.numeric(tt$P.Value), padj = as.numeric(tt$adj.P.Val),
             stringsAsFactors = FALSE)
}

s4_direction <- function(log2fc, padj, padj_thr, lfc_thr) {
  d <- rep("ns", length(log2fc))
  sig <- !is.na(padj) & !is.na(log2fc) & padj < padj_thr
  d[sig & log2fc >= lfc_thr] <- "up"
  d[sig & log2fc <= -lfc_thr] <- "down"
  d
}

#' One contrast x group. Returns list(table (or NULL), method_used, notes, skipped_reason, n_tested).
s4_dge_group <- function(counts, coldata, cond_col, numerator, denominator, covariates, method, padj_thr, lfc_thr, min_replicates) {
  notes <- character()
  coldata[[cond_col]] <- factor(as.character(coldata[[cond_col]]), levels = c(denominator, numerator))
  for (cv in covariates) coldata[[cv]] <- factor(as.character(coldata[[cv]]))
  des <- s4_design(coldata, cond_col, covariates, min_replicates)
  notes <- c(notes, des$notes)
  if (!isTRUE(des$ok)) return(list(table = NULL, method_used = NULL, notes = notes, skipped_reason = des$reason, n_tested = 0L, covariates_used = des$covariates))
  keep <- Matrix::rowSums(counts) >= .S4_MIN_GENE_COUNTS
  m <- as.matrix(counts[keep, , drop = FALSE])
  n_tested <- nrow(m)
  if (n_tested < 2) return(list(table = NULL, method_used = NULL, notes = notes, skipped_reason = sprintf("only %d gene(s) with >= %d counts", n_tested, .S4_MIN_GENE_COUNTS), n_tested = n_tested, covariates_used = des$covariates))
  method_used <- method
  tab <- NULL
  if (method == "deseq2") {
    if (!scp_has_pkg("DESeq2")) {
      notes <- c(notes, "DESeq2 is not installed; limma_voom used instead")
      method_used <- "limma_voom"
    } else {
      tab <- tryCatch(s4_run_deseq2(m, coldata, des$formula, cond_col, numerator, denominator, padj_thr),
                      error = function(e) { notes <<- c(notes, paste0("DESeq2 failed (", conditionMessage(e), "); limma_voom used instead")); NULL })
      if (is.null(tab)) method_used <- "limma_voom"
    }
  }
  if (is.null(tab)) tab <- s4_run_limma_voom(m, coldata, des$formula, cond_col, numerator, denominator)
  tab$direction <- s4_direction(tab$log2fc, tab$padj, padj_thr, lfc_thr)
  o <- order(is.na(tab$padj), tab$padj, tab$pvalue, -abs(tab$log2fc))
  tab <- tab[o, .S4_DGE_COLS, drop = FALSE]
  rownames(tab) <- NULL
  list(table = tab, method_used = method_used, notes = notes, skipped_reason = NULL, n_tested = n_tested, covariates_used = des$covariates)
}

# --------------------------------------------------------------------------
# FindMarkers (single-cell tests)
# --------------------------------------------------------------------------

#' The object ready for FindMarkers: RNA layers joined, a normalised data layer present.
s4_prepare_findmarkers <- function(obj) {
  Seurat::DefaultAssay(obj) <- "RNA"
  if (inherits(obj[["RNA"]], "Assay5") && length(SeuratObject::Layers(obj[["RNA"]], search = "data")) > 1)
    obj[["RNA"]] <- SeuratObject::JoinLayers(obj[["RNA"]])
  has_data <- tryCatch(length(SeuratObject::Layers(obj[["RNA"]], search = "data")) > 0, error = function(e) FALSE)
  if (!has_data) {
    scp_log("findmarkers: no normalised data layer in the S2 object; running NormalizeData (LogNormalize)")
    obj <- Seurat::NormalizeData(obj, normalization.method = "LogNormalize", verbose = FALSE)
  }
  obj
}

#' One FindMarkers test between the "num" and "den" cells of `obj`. Returns the same list as
#' s4_dge_group (table, method_used, notes, skipped_reason, n_tested, covariates_used).
s4_findmarkers_group <- function(obj, side, test, covariates, d, seed) {
  notes <- character()
  test_used <- test
  if (test == "MAST" && !scp_has_pkg("MAST")) {
    notes <- c(notes, "MAST is not installed; wilcox used instead")
    test_used <- "wilcox"
  }
  if (test == "DESeq2" && !scp_has_pkg("DESeq2")) {
    notes <- c(notes, "DESeq2 is not installed; wilcox used instead")
    test_used <- "wilcox"
  }
  latent <- NULL
  if (length(covariates)) {
    if (test_used %in% .S4_LATENT_TESTS) {
      single <- covariates[vapply(covariates, function(cv) length(unique(obj[[cv]][[1]])) < 2, TRUE)]
      if (length(single)) notes <- c(notes, paste0("covariate(s) ", paste(single, collapse = ", "), " have one level here and were dropped"))
      latent <- setdiff(covariates, single)
      if (!length(latent)) latent <- NULL
    } else {
      notes <- c(notes, paste0("covariates are used by the MAST, LR, negbinom and poisson tests only; ignored for ", test_used))
    }
  }
  SeuratObject::Idents(obj) <- factor(side, levels = c("den", "num"))
  maxc <- d$max_cells_per_side
  maxc <- if (is.null(maxc) || is.na(maxc)) Inf else as.numeric(maxc)
  fm <- tryCatch(suppressWarnings(suppressMessages(Seurat::FindMarkers(
    obj, ident.1 = "num", ident.2 = "den", test.use = test_used, logfc.threshold = d$logfc_threshold,
    min.pct = d$min_pct, max.cells.per.ident = maxc, latent.vars = latent, random.seed = seed, verbose = FALSE))),
    error = function(e) e)
  if (inherits(fm, "error"))
    return(list(table = NULL, method_used = NULL, notes = notes, n_tested = 0L, covariates_used = latent %||% character(),
                skipped_reason = paste0("FindMarkers (", test_used, ") failed: ", conditionMessage(fm))))
  if (!nrow(fm))
    return(list(table = NULL, method_used = NULL, notes = notes, n_tested = 0L, covariates_used = latent %||% character(),
                skipped_reason = sprintf("no gene passed min.pct %g and logfc.threshold %g", d$min_pct, d$logfc_threshold)))
  genes <- rownames(fm)
  dat <- SeuratObject::LayerData(obj, assay = "RNA", layer = "data")[genes, , drop = FALSE]
  tab <- data.frame(gene = genes, baseMean = as.numeric(Matrix::rowMeans(dat)), log2fc = as.numeric(fm$avg_log2FC),
                    lfcSE = NA_real_, pvalue = as.numeric(fm$p_val), padj = as.numeric(fm$p_val_adj),
                    stringsAsFactors = FALSE)
  tab$direction <- s4_direction(tab$log2fc, tab$padj, d$padj, d$lfc)
  tab$pct_1 <- as.numeric(fm$pct.1); tab$pct_2 <- as.numeric(fm$pct.2)
  o <- order(is.na(tab$padj), tab$padj, tab$pvalue, -abs(tab$log2fc))
  tab <- tab[o, c(.S4_DGE_COLS, "pct_1", "pct_2"), drop = FALSE]
  rownames(tab) <- NULL
  list(table = tab, method_used = paste0("findmarkers:", test_used), notes = notes, skipped_reason = NULL,
       n_tested = nrow(tab), covariates_used = latent %||% character())
}

.s4_top <- function(tab, dir, n = .S4_TOP_N) {
  t <- tab[tab$direction == dir, , drop = FALSE]
  t <- t[order(t$padj, -abs(t$log2fc)), , drop = FALSE]
  t <- utils::head(t, n)
  lapply(seq_len(nrow(t)), function(i) list(gene = t$gene[i], log2fc = round(t$log2fc[i], 4), padj = signif(t$padj[i], 4)))
}

.s4_col <- function(x) if (identical(x, "cluster")) "seurat_clusters" else x

#' Parallel FindMarkers workers: CELLWARDEN_THREADS (the runner's resources.threads), at least 1.
s4_workers <- function() {
  n <- suppressWarnings(as.integer(Sys.getenv("CELLWARDEN_THREADS", "1")))
  if (is.na(n) || n < 1) 1L else n
}

.s4_group_entry <- function(g, numerator, denominator, n_samples, n_cells, covariates) {
  list(group = g, file = NULL, numerator = numerator, denominator = denominator, n_samples = n_samples, n_cells = n_cells,
       covariates_used = as.list(covariates), method_used = NULL, n_tested = 0L, n_up = 0L, n_down = 0L,
       skipped_reason = NULL, notes = list(), top_up = list(), top_down = list())
}

.s4_store <- function(entry, r, f) {
  entry$notes <- as.list(r$notes); entry$covariates_used <- as.list(r$covariates_used); entry$n_tested <- as.integer(r$n_tested)
  if (is.null(r$table)) { entry$skipped_reason <- r$skipped_reason; return(entry) }
  scp_write_tsv_gz(r$table, f)
  entry$file <- file.path("S4", "dge", basename(f))
  entry$method_used <- r$method_used
  entry$n_up <- sum(r$table$direction == "up"); entry$n_down <- sum(r$table$direction == "down")
  entry$top_up <- .s4_top(r$table, "up"); entry$top_down <- .s4_top(r$table, "down")
  entry
}

#' One contrast with FindMarkers: per group of `within` (or all cells), or each group of the
#' column against the rest (`each`). Returns list(entry, methods).
s4_contrast_findmarkers <- function(obj, md, ct, col, within, each, d, seed, dge_dir) {
  vals <- as.character(md[[col]])
  samp <- if (d$sample_col %in% colnames(md)) as.character(md[[d$sample_col]]) else rep("all", nrow(md))
  num_l <- if (each) "each group" else .s4_label(ct$numerator)
  den_l <- if (each) "the other cells" else .s4_label(ct$denominator)
  groups <- if (each) sort(unique(vals[!is.na(vals)])) else if (is.null(within)) "all cells" else
    sort(unique(as.character(md[[within]][!is.na(md[[within]])])))
  side_all <- if (each) NULL else s4_sides(vals, ct$numerator, ct$denominator)
  fnames <- make.unique(paste0(ct$name, "__", .s4_safe_name(groups)), sep = ".")
  one <- function(gi) {
    g <- groups[[gi]]
    if (each) {
      side <- ifelse(vals == g, "num", "den"); keep <- !is.na(vals)
    } else {
      side <- side_all
      keep <- !is.na(side) & (if (is.null(within)) TRUE else (as.character(md[[within]]) == g & !is.na(md[[within]])))
    }
    idx <- which(keep); sd <- side[idx]
    n1 <- sum(sd == "num"); n2 <- sum(sd == "den")
    entry <- .s4_group_entry(g, if (each) g else num_l, if (each) "rest" else den_l,
                             list(numerator = length(unique(samp[idx][sd == "num"])), denominator = length(unique(samp[idx][sd == "den"]))),
                             list(numerator = n1, denominator = n2), ct$covariates)
    if (n1 < d$min_cells_per_side || n2 < d$min_cells_per_side) {
      entry$skipped_reason <- sprintf("%d %s and %d %s cell(s); min_cells_per_side is %d", n1, entry$numerator, n2,
                                      entry$denominator, d$min_cells_per_side)
      scp_log("  ", g, ": skipped (", entry$skipped_reason, ")")
      return(list(entry = entry, method = NULL))
    }
    r <- s4_findmarkers_group(obj[, idx], sd, d$test, ct$covariates, d, seed + gi)
    entry <- .s4_store(entry, r, file.path(dge_dir, paste0(fnames[[gi]], ".tsv.gz")))
    if (is.null(entry$skipped_reason)) {
      scp_log(sprintf("  %s: %s, %d tested, %d up, %d down (%d vs %d cells)", g, r$method_used, entry$n_tested,
                      entry$n_up, entry$n_down, n1, n2))
    } else scp_log("  ", g, ": skipped (", entry$skipped_reason, ")")
    list(entry = entry, method = if (is.null(entry$skipped_reason)) r$method_used else NULL)
  }
  # Groups are independent: test them side by side (forked workers, one BLAS thread each).
  workers <- min(length(groups), s4_workers())
  res <- if (workers > 1 && .Platform$OS.type == "unix") {
    scp_log(sprintf("  testing %d groups with %d workers", length(groups), workers))
    parallel::mclapply(seq_along(groups), function(gi) {
      if (scp_has_pkg("RhpcBLASctl")) try(RhpcBLASctl::blas_set_num_threads(1), silent = TRUE)
      one(gi)
    }, mc.cores = workers, mc.preschedule = FALSE)
  } else lapply(seq_along(groups), one)
  group_out <- list(); methods <- character()
  for (gi in seq_along(groups)) {
    r <- res[[gi]]
    if (inherits(r, "try-error") || is.null(r$entry)) {
      e <- .s4_group_entry(groups[[gi]], num_l, den_l, list(), list(), ct$covariates)
      e$skipped_reason <- paste0("FindMarkers worker failed: ", trimws(as.character(r)))
      scp_warn_log("  ", groups[[gi]], ": ", e$skipped_reason)
      group_out[[gi]] <- e
    } else {
      group_out[[gi]] <- r$entry
      if (!is.null(r$method)) methods <- union(methods, r$method)
    }
  }
  side_cells <- if (each) NULL else side_all
  samples_side <- if (each) list() else list(numerator = sort(unique(samp[which(side_cells == "num")])),
                                             denominator = sort(unique(samp[which(side_cells == "den")])))
  list(methods = methods, entry = list(
    name = ct$name, engine = "findmarkers", column = col, numerator = num_l, denominator = den_l,
    within = within %||% "none", covariates = as.list(ct$covariates), samples = samples_side, groups = group_out))
}

#' One contrast with pseudobulk counts (`pb` from s4_pseudobulk by within-group x sample). The contrast
#' column and the covariates must have one value per sample. Returns list(entry, methods).
s4_contrast_pseudobulk <- function(pb, md, ct, col, d, out_dir, dge_dir) {
  cols <- unique(c(col, ct$covariates))
  sm <- unique(data.frame(sample = as.character(md[[d$sample_col]]), lapply(md[cols], as.character),
                          stringsAsFactors = FALSE, check.names = FALSE))
  if (anyDuplicated(sm$sample)) {
    bad <- unique(sm$sample[duplicated(sm$sample)])
    scp_stop("dge.contrasts '", ct$name, "': ", paste(cols, collapse = "/"), " vary within ", d$sample_col, " ",
             paste(utils::head(bad, 5), collapse = ", "), ". Pseudobulk compares samples, so the column needs one value ",
             "per sample; method findmarkers compares cells and accepts any column")
  }
  rownames(sm) <- sm$sample
  num_l <- .s4_label(ct$numerator); den_l <- .s4_label(ct$denominator)
  u <- pb$units
  u$side <- s4_sides(sm[u$sample, col], ct$numerator, ct$denominator)
  for (cv in ct$covariates) u[[cv]] <- sm[u$sample, cv]
  u <- u[!is.na(u$side), , drop = FALSE]
  u$condition <- ifelse(u$side == "num", num_l, den_l)
  u$included <- u$n_cells >= d$min_cells_per_sample
  u$exclude_reason <- ifelse(u$included, "", sprintf("fewer than %d cells", d$min_cells_per_sample))
  pb_dir <- file.path(out_dir, "pseudobulk", ct$name); dir.create(pb_dir, recursive = TRUE, showWarnings = FALSE)
  inc <- u[u$included, , drop = FALSE]
  cm <- as.matrix(pb$counts[, inc$pb_id, drop = FALSE])
  scp_write_tsv_gz(data.frame(gene = rownames(cm), cm, check.names = FALSE, stringsAsFactors = FALSE), file.path(pb_dir, "counts.tsv.gz"))
  .s4_write_tsv(u, file.path(pb_dir, "samples.tsv"))
  samples_side <- list(numerator = sort(unique(inc$sample[inc$side == "num"])), denominator = sort(unique(inc$sample[inc$side == "den"])))
  groups <- sort(unique(u$group))
  fnames <- make.unique(paste0(ct$name, "__", .s4_safe_name(groups)), sep = ".")
  methods <- character(); group_out <- list()
  for (gi in seq_along(groups)) {
    g <- groups[[gi]]
    ug <- inc[inc$group == g, , drop = FALSE]
    n_num <- sum(ug$side == "num"); n_den <- sum(ug$side == "den")
    entry <- .s4_group_entry(g, num_l, den_l, list(numerator = n_num, denominator = n_den),
                             list(numerator = sum(ug$n_cells[ug$side == "num"]), denominator = sum(ug$n_cells[ug$side == "den"])),
                             ct$covariates)
    if (n_num < d$min_replicates || n_den < d$min_replicates) {
      entry$skipped_reason <- sprintf("%d %s and %d %s pseudobulk sample(s) with >= %d cells; min_replicates is %d per side",
                                      n_num, num_l, n_den, den_l, d$min_cells_per_sample, d$min_replicates)
      scp_log("  group ", g, ": skipped (", entry$skipped_reason, ")")
      group_out[[gi]] <- entry; next
    }
    coldata <- data.frame(sample = ug$sample, side = ug$side, ug[, ct$covariates, drop = FALSE], row.names = ug$pb_id,
                          stringsAsFactors = FALSE, check.names = FALSE)
    r <- s4_dge_group(pb$counts[, ug$pb_id, drop = FALSE], coldata, "side", "num", "den",
                      ct$covariates, d$pseudobulk_method, d$padj, d$lfc, d$min_replicates)
    entry <- .s4_store(entry, r, file.path(dge_dir, paste0(fnames[[gi]], ".tsv.gz")))
    if (is.null(entry$skipped_reason)) {
      methods <- union(methods, r$method_used)
      scp_log(sprintf("  group %s: %s, %d tested, %d up, %d down (%d vs %d samples)", g, r$method_used, entry$n_tested,
                      entry$n_up, entry$n_down, n_num, n_den))
    } else scp_log("  group ", g, ": skipped (", entry$skipped_reason, ")")
    group_out[[gi]] <- entry
  }
  list(methods = methods, entry = list(
    name = ct$name, engine = "pseudobulk", column = col, numerator = num_l, denominator = den_l,
    within = if (identical(unique(u$group), "all cells")) "none" else ct$within, covariates = as.list(ct$covariates),
    samples = samples_side, pseudobulk_dir = file.path("S4", "pseudobulk", ct$name),
    n_units = nrow(u), n_units_excluded = sum(!u$included), groups = group_out))
}

# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------

run_s4 <- function(cfg, repo) {
  out_dir <- scp_stage_dir(cfg, "S4")
  d <- s4_config(cfg)
  set.seed(cfg$seed %||% 42L)
  timings <- list()
  tm <- function(label, expr) { r <- scp_timed(label, expr); timings[[label]] <<- r$seconds; r$value }
  notes <- character()

  obj <- tm("load", scp_load_object(file.path(cfg$run_dir, "S2")))
  lab <- s4_apply_labels(obj, cfg$run_dir)
  obj <- lab$obj
  if (!is.null(lab$note)) notes <- c(notes, lab$note)
  md <- obj[[]]

  # gene map (for the Open Targets step and the report)
  fm <- tryCatch(obj[["RNA"]][[]], error = function(e) NULL)
  genes <- rownames(obj[["RNA"]])
  sym <- scp_symbols(obj)
  gid <- if (!is.null(fm) && "gene_id" %in% colnames(fm)) as.character(fm$gene_id) else rep("", length(genes))
  gid[is.na(gid)] <- ""
  ens <- ifelse(grepl("^ENSG[0-9]+", gid), sub("\\..*$", "", gid), ifelse(grepl("^ENSG[0-9]+", genes), sub("\\..*$", "", genes), ""))
  .s4_write_tsv(data.frame(gene = genes, symbol = sym, ensembl_id = ens, stringsAsFactors = FALSE), file.path(out_dir, "gene_map.tsv"))

  # grouping column
  gb_req <- d$group_by
  gb <- if (identical(gb_req, "cluster")) "seurat_clusters" else gb_req
  gb_note <- NULL
  if (!(gb %in% colnames(md))) {
    gb_note <- sprintf("group_by '%s' is not a cell metadata column%s; grouping by seurat_clusters instead", gb_req,
                       if (lab$source == "none") " (no cell-type labels available)" else "")
    scp_warn_log(gb_note); notes <- c(notes, gb_note)
    gb <- "seurat_clusters"
  }
  if (!(gb %in% colnames(md))) scp_stop("the S2 object has no '", gb, "' column")

  # contrasts: the configured ones; with findmarkers and none configured (or markers: true), the
  # markers of each group_by group against the other cells
  contrasts <- d$contrasts
  if (d$engine == "findmarkers" && (isTRUE(d$markers) || (identical(d$markers, "auto") && !length(contrasts)))) {
    contrasts <- c(contrasts, list(list(name = "markers", column = gb, numerator = ".each", denominator = "rest",
                                        within = "none", covariates = character())))
    scp_log("dge: markers of each ", gb, " group against the other cells (FindMarkers, ", d$test, ")")
  }
  if (!length(contrasts)) {
    notes <- c(notes, "dge.contrasts is empty: no differential expression was run")
    scp_warn_log("dge.contrasts is empty: nothing to test (set dge.contrasts, or method findmarkers for markers per group)")
  }

  dge_dir <- file.path(out_dir, "dge"); dir.create(dge_dir, recursive = TRUE, showWarnings = FALSE)
  unlink(list.files(dge_dir, pattern = "\\.tsv\\.gz$", full.names = TRUE))
  method_used_all <- character()
  contrast_out <- list()
  fobj <- if (d$engine == "findmarkers" && length(contrasts)) tm("prepare", s4_prepare_findmarkers(obj)) else NULL
  pb_cache <- list()
  pb_method <- NULL
  for (ct in contrasts) {
    t0 <- proc.time()[["elapsed"]]
    col <- .s4_col(ct$column)
    within <- if (tolower(ct$within) %in% c("none", "all")) NULL else if (identical(ct$within, gb_req)) gb else .s4_col(ct$within)
    for (cc in c(col, within, ct$covariates))
      if (!(cc %in% colnames(md))) scp_stop("dge.contrasts '", ct$name, "': column '", cc, "' is not in the cell metadata ",
                                            "(samples.csv columns, cell_type_broad, cell_type_fine, seurat_clusters, ...); available: ",
                                            paste(colnames(md), collapse = ", "))
    each <- identical(ct$numerator, ".each")
    if (!each) s4_check_values(ct, md[[col]])
    scp_log("contrast ", ct$name, ": ", if (each) paste0("each ", col, " group vs the rest") else
              paste0(col, " ", .s4_label(ct$numerator), " vs ", .s4_label(ct$denominator)),
            if (is.null(within)) "" else paste0(", within each ", within),
            if (length(ct$covariates)) paste0(" (covariates: ", paste(ct$covariates, collapse = ", "), ")") else "")
    r <- if (d$engine == "findmarkers") {
      s4_contrast_findmarkers(fobj, md, ct, col, within, each, d, cfg$seed %||% 42L, dge_dir)
    } else {
      key <- within %||% ".all"
      if (is.null(pb_cache[[key]])) {
        if (!(d$sample_col %in% colnames(md))) scp_stop("dge: sample_col '", d$sample_col, "' is not in the cell metadata")
        gv <- if (is.null(within)) rep("all cells", ncol(obj)) else as.character(md[[within]])
        pb_cache[[key]] <- tm(paste0("pseudobulk_", key), s4_pseudobulk(obj, gv, as.character(md[[d$sample_col]])))
        pb_method <- pb_cache[[key]]$method
        u <- pb_cache[[key]]$units
        scp_log(sprintf("pseudobulk: %d units (%d groups x %d samples) via %s", nrow(u), length(unique(u$group)),
                        length(unique(u$sample)), pb_cache[[key]]$method))
      }
      s4_contrast_pseudobulk(pb_cache[[key]], md, ct, col, d, out_dir, dge_dir)
    }
    method_used_all <- union(method_used_all, r$methods)
    timings[[paste0("contrast_", ct$name)]] <- round(proc.time()[["elapsed"]] - t0, 3)
    contrast_out[[length(contrast_out) + 1]] <- r$entry
  }
  sample_v <- if (d$sample_col %in% colnames(md)) as.character(md[[d$sample_col]]) else rep("all", ncol(obj))
  group_v <- as.character(md[[gb]])
  cond_levels <- if (d$condition_col %in% colnames(md)) sort(unique(as.character(md[[d$condition_col]]))) else character()

  summary <- list(
    stage = "S4", step = "dge", run_id = cfg$run_id, config_hash = cfg$config_hash, created_utc = .scp_utc(),
    method = list(requested = d$method, engine = d$engine,
                  test = if (d$engine == "findmarkers") d$test else NULL,
                  pseudobulk_method = if (d$engine == "pseudobulk") d$pseudobulk_method else NULL,
                  used = as.list(method_used_all),
                  available = list(DESeq2 = scp_has_pkg("DESeq2"), limma = scp_has_pkg("limma"), MAST = scp_has_pkg("MAST"))),
    group_by = list(requested = gb_req, used = gb, note = gb_note),
    labels = list(source = lab$source, note = lab$note, n_unassigned_cells = lab$n_unassigned),
    sample_col = d$sample_col, condition_col = d$condition_col, condition_levels = as.list(cond_levels),
    thresholds = list(padj = d$padj, lfc = d$lfc, min_cells_per_sample = d$min_cells_per_sample,
                      min_replicates = d$min_replicates, min_gene_counts = .S4_MIN_GENE_COUNTS,
                      min_pct = d$min_pct, logfc_threshold = d$logfc_threshold, min_cells_per_side = d$min_cells_per_side),
    n_cells = ncol(obj), n_genes = nrow(obj), n_groups = length(unique(group_v)), n_samples = length(unique(sample_v)),
    pseudobulk_method = pb_method, gene_map = "S4/gene_map.tsv",
    contrasts = contrast_out, notes = as.list(notes),
    timings_s = timings, total_s = round(sum(unlist(timings)), 3),
    peak_memory = list(r_peak_rss_gb = scp_peak_mem_gb()),
    packages = list(R = paste(R.version$major, R.version$minor, sep = "."), Seurat = scp_pkg_version("Seurat"),
                    DESeq2 = scp_pkg_version("DESeq2"), limma = scp_pkg_version("limma"), MAST = scp_pkg_version("MAST"))
  )
  scp_write_json(summary, file.path(out_dir, "dge_summary.json"))
  n_tab <- sum(vapply(contrast_out, function(c) sum(vapply(c$groups, function(g) !is.null(g$file), TRUE)), 1))
  scp_log(sprintf("S4 DGE done: %d contrast(s), %d DGE table(s), group_by %s", length(contrast_out), n_tab, gb))
  invisible(summary)
}
