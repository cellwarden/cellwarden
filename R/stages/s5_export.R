# Stage S5 (R part): export what the report and explorer need from the Seurat
# object (docs/agents-spec.md section 7).
#
# Inputs:  S2/object.rds (or .qs2); S4/object.* if a later stage saved an
#          annotated object; S3/annotation.tsv (cluster, label_broad,
#          label_fine) when the object lacks cell_type_* columns.
# Outputs: S5/umap.tsv.gz     cell_id, umap_1, umap_2, seurat_clusters, sample_id,
#                             cell_type_broad, cell_type_fine (when known),
#                             plus n_genes/n_umis/pct_mito when in meta.data
#          S5/expr_subset.tsv.gz  sparse long table gene, cell_id, value
#                             (log-normalised `data` layer, non-zero only, 3 dp)
#                             for the top `report.expr_genes` variable genes
#                             x up to `report.expr_max_cells` cells (seeded
#                             subsample), so the explorer can colour by gene
#          S5/export.json     what was exported and why
#
# The Python side (python/cellwarden/stages/s5_report.py) runs this through
# `Rscript R/stages/run_stage.R --stage S5 --run-dir DIR`, then builds the HTML.

s5_find_object <- function(cfg) {
  for (st in c("S4", "S2")) {
    d <- file.path(cfg$run_dir, st)
    cand <- file.path(d, c("object.qs2", "object.rds"))
    cand <- cand[file.exists(cand)]
    if (length(cand)) return(list(path = cand[which.max(file.mtime(cand))], stage = st))
  }
  scp_stop("no S2/object.rds (or .qs2) to export from")
}

# S3/annotation.tsv (final, from the G3 decision) or, before G3 has run, the
# code-only consensus in S3/annotation_consensus.tsv.
s5_annotation_table <- function(cfg) {
  for (f in c("annotation.tsv", "annotation_consensus.tsv")) {
    p <- file.path(cfg$run_dir, "S3", f)
    if (!file.exists(p)) next
    a <- tryCatch(scp_read_tsv(p), error = function(e) NULL)
    if (!is.null(a) && "cluster" %in% names(a)) { attr(a, "source") <- f; return(a) }
  }
  NULL
}

s5_meta_col <- function(md, name) if (name %in% names(md)) as.character(md[[name]]) else NULL

run_s5 <- function(cfg, repo) {
  rep_cfg <- cfg$report %||% list()
  n_genes <- as.integer(rep_cfg$expr_genes %||% 200L)
  max_cells <- as.integer(rep_cfg$expr_max_cells %||% 20000L)
  out_dir <- scp_stage_dir(cfg, "S5")
  src <- s5_find_object(cfg)
  scp_log("S5 export: reading ", src$path)
  obj <- scp_load_object(src$path)
  md <- obj@meta.data
  cells <- colnames(obj)
  notes <- character()

  # --- UMAP / metadata ---------------------------------------------------
  red <- NULL
  for (r in c("umap", "umap.harmony", "UMAP")) if (r %in% SeuratObject::Reductions(obj)) { red <- r; break }
  if (is.null(red)) {
    reds <- SeuratObject::Reductions(obj)
    red <- reds[grepl("umap", reds, ignore.case = TRUE)][1]
  }
  if (is.na(red) || is.null(red)) {
    notes <- c(notes, "no UMAP reduction in the object; umap_1/umap_2 are NA")
    emb <- matrix(NA_real_, nrow = length(cells), ncol = 2)
  } else {
    emb <- SeuratObject::Embeddings(obj, reduction = red)[cells, 1:2, drop = FALSE]
  }
  cl <- s5_meta_col(md, "seurat_clusters")
  if (is.null(cl)) { cl <- as.character(SeuratObject::Idents(obj)); notes <- c(notes, "seurat_clusters missing; used Idents()") }
  sid <- s5_meta_col(md, "sample_id") %||% rep(NA_character_, length(cells))
  umap <- data.frame(cell_id = cells, umap_1 = round(emb[, 1], 4), umap_2 = round(emb[, 2], 4),
                     seurat_clusters = cl, sample_id = sid, stringsAsFactors = FALSE)
  broad <- s5_meta_col(md, "cell_type_broad"); fine <- s5_meta_col(md, "cell_type_fine")
  if (is.null(broad)) {
    ann <- s5_annotation_table(cfg)
    if (!is.null(ann)) {
      i <- match(cl, as.character(ann$cluster))
      if ("label_broad" %in% names(ann)) broad <- as.character(ann$label_broad)[i]
      if ("label_fine" %in% names(ann)) fine <- as.character(ann$label_fine)[i]
      notes <- c(notes, paste0("cell types joined from S3/", attr(ann, "source"), " by cluster"))
    }
  }
  if (!is.null(broad)) umap$cell_type_broad <- broad
  if (!is.null(fine)) umap$cell_type_fine <- fine
  for (k in c("n_genes", "n_umis", "pct_mito", "doublet_score", "nFeature_RNA", "nCount_RNA", "percent.mt")) {
    if (k %in% names(md) && !(k %in% names(umap))) umap[[k]] <- md[[k]]
  }
  # sample metadata columns (condition, batch, ...) become colour options too
  meta_cols <- cfg$sample_columns %||% character()
  for (k in setdiff(meta_cols, c("sample_id", "path", "format", names(umap)))) {
    if (k %in% names(md)) umap[[k]] <- as.character(md[[k]])
  }
  scp_write_tsv_gz(umap, file.path(out_dir, "umap.tsv.gz"))
  scp_log(sprintf("S5 export: umap.tsv.gz with %d cells, %d columns", nrow(umap), ncol(umap)))

  # --- expression subset -------------------------------------------------
  assay <- SeuratObject::DefaultAssay(obj)
  layers <- SeuratObject::Layers(obj, assay = assay)
  layer <- if ("data" %in% layers) "data" else layers[grepl("^data", layers)][1]
  genes <- tryCatch(SeuratObject::VariableFeatures(obj, assay = assay), error = function(e) character())
  if (!length(genes)) {
    notes <- c(notes, "no variable features; using the genes with the highest total expression")
    tot <- Matrix::rowSums(SeuratObject::LayerData(obj, assay = assay, layer = layer))
    genes <- names(sort(tot, decreasing = TRUE))
  }
  genes <- head(genes[genes %in% rownames(obj)], n_genes)
  set.seed(as.integer(cfg$seed %||% 42L))
  sub_cells <- if (length(cells) > max_cells) sort(sample(cells, max_cells)) else cells
  n_nz <- 0L
  if (is.na(layer) || !length(genes)) {
    notes <- c(notes, "no log-normalised layer or no genes; expr_subset.tsv.gz is empty")
    long <- data.frame(gene = character(), cell_id = character(), value = numeric())
  } else {
    x <- SeuratObject::LayerData(obj, assay = assay, layer = layer)[genes, sub_cells, drop = FALSE]
    x <- scp_as_dgc(x)
    tri <- Matrix::summary(x)   # i (gene), j (cell), x
    long <- data.frame(gene = rownames(x)[tri$i], cell_id = colnames(x)[tri$j], value = round(tri$x, 3),
                       stringsAsFactors = FALSE)
    long <- long[long$value != 0, , drop = FALSE]
    n_nz <- nrow(long)
  }
  scp_write_tsv_gz(long, file.path(out_dir, "expr_subset.tsv.gz"))
  scp_log(sprintf("S5 export: expr_subset.tsv.gz with %d genes x %d cells (%d non-zero)", length(genes), length(sub_cells), n_nz))

  scp_write_json(list(
    stage = "S5", run_id = cfg$run_id, created_utc = .scp_utc(), object = src$path, object_stage = src$stage,
    reduction = if (is.null(red) || is.na(red)) NULL else red, cells = length(cells), clusters = length(unique(cl)),
    annotation = !is.null(broad), layer = if (is.na(layer)) NULL else layer, assay = assay,
    expr_genes = length(genes), expr_cells = length(sub_cells), expr_nonzero = n_nz,
    umap_columns = names(umap), notes = as.list(notes)
  ), file.path(out_dir, "export.json"))
  invisible(TRUE)
}
