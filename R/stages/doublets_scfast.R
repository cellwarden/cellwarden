# cellwarden S1a, scfast engine: scrublet-style doublet detection and coarse
# pre-clustering, in process, with scfastR (docs/scfast-integration.md,
# "S1b doublets and pre-clustering"). Used by run_s1a() when
# scp_resolve_engine() gives "scfast"; replaces Python scrublet and the
# cellwarden.gpu_core pre-clustering of S1b, so no S1/metrics_counts.h5ad is written.
#
# Doublets, per sample, on raw counts (genes x cells dgCMatrix):
#   1. simulate 2 * n_cells doublets by summing the counts of random cell pairs;
#   2. total-normalise observed and simulated cells to the observed mean depth, log1p;
#   3. HVGs (binned dispersion on the observed cells; at most 2000 and 15% of the
#      expressed genes, at least 100);
#   4. scfast_pca on the observed cells; simulated cells are projected with the
#      same centring, scaling and loadings (as Scrublet does);
#   5. scfast_knn on observed + simulated cells; doublet_score of an observed cell
#      is the fraction of its neighbours (self excluded) that are simulated, with
#      Scrublet's density correction for the expected doublet rate (6%);
#   6. threshold = the valley of the bimodal distribution of the simulated cells'
#      scores (smoothed histogram, like skimage threshold_minimum used by Scrublet),
#      kept within median + 2 to 2.5 MAD of the observed scores;
#      doublet_call = score >= threshold.
# Samples below SCP_DBL_MIN_CELLS cells are skipped (score NA, call FALSE, warning),
# as in the Python implementation.
#
# Pre-clustering, all cells together: log-normalise (1e4), 2000 HVGs, 20 PCs,
# kNN (k = 15, self included), SNN (prune 1/15), Leiden at resolution 0.3, seed fixed.

SCP_DBL_MIN_CELLS <- 100L
SCP_DBL_SIM_RATIO <- 2
SCP_DBL_EXPECTED_RATE <- 0.06
SCP_DBL_N_HVG <- 2000L
SCP_DBL_N_PCS <- 30L
SCP_PRECLUSTER <- list(n_hvg = 2000L, n_pcs = 20L, k = 15L, resolution = 0.3, prune = 1 / 15)

# log1p(counts scaled to `target` UMIs per cell), same sparsity as `m`.
.dbl_lognorm <- function(m, target = 1e4) {
  tot <- Matrix::colSums(m)
  f <- ifelse(tot > 0, target / tot, 0)
  m@x <- log1p(m@x * rep(f, diff(m@p)))
  m
}

# Top n genes by dispersion (var / mean) z-scored within 20 mean bins, on a log-normalised
# genes x cells dgCMatrix. Returns row indices.
.dbl_hvg <- function(logm, n) {
  nc <- ncol(logm)
  mu <- Matrix::rowMeans(logm)
  ex2 <- Matrix::rowMeans(logm^2)
  v <- pmax(ex2 - mu^2, 0) * nc / max(nc - 1, 1)
  ok <- which(mu > 0 & v > 0)
  if (length(ok) <= n) return(ok)
  disp <- log(v[ok] / mu[ok])
  br <- unique(stats::quantile(mu[ok], seq(0, 1, length.out = 21)))
  bin <- if (length(br) > 2) cut(mu[ok], br, include.lowest = TRUE, labels = FALSE) else rep(1L, length(ok))
  z <- numeric(length(ok))
  for (b in unique(bin)) {
    s <- which(bin == b)
    sdv <- if (length(s) > 1) stats::sd(disp[s]) else 0
    z[s] <- if (is.finite(sdv) && sdv > 0) (disp[s] - mean(disp[s])) / sdv else 0
  }
  ok[order(z, decreasing = TRUE)[seq_len(n)]]
}

# Fraction of simulated neighbours -> doublet score, with Scrublet's correction for the
# density of simulated cells (r = n_sim / n_obs) and the expected doublet rate rho.
# Monotone in the fraction; 0 with no simulated neighbours is lifted by the +1 / +2 prior.
.dbl_adjust <- function(n_sim_nn, k, r, rho) {
  q <- (n_sim_nn + 1) / (k + 2)
  q * rho / (1 - rho - q * (1 - rho - rho / r))
}

# Drop each row's own index from an n x k neighbour matrix (self is first in the exact
# case; if ties put it elsewhere, remove it there; if absent, drop the last column).
.dbl_drop_self <- function(idx) {
  n <- nrow(idx)
  k <- ncol(idx)
  self <- idx == seq_len(n)
  first <- self[, 1]
  out <- idx[, -1, drop = FALSE]
  odd <- which(!first)
  for (i in odd) out[i, ] <- if (any(self[i, ])) idx[i, !self[i, ]] else idx[i, -k]
  out
}

# Threshold at the valley between the two modes of a bimodal score distribution:
# 50-bin histogram on [0, 1], smoothed with a 3-bin mean until it has exactly two peaks
# (skimage.filters.threshold_minimum). NA if it never has two peaks.
.dbl_threshold <- function(sim_scores, nbins = 50L) {
  h <- as.numeric(table(factor(findInterval(pmin(sim_scores, 1 - 1e-9), seq(0, 1, length.out = nbins + 1),
                                            all.inside = TRUE), levels = seq_len(nbins))))
  centers <- (seq_len(nbins) - 0.5) / nbins
  peaks <- function(x) {
    m <- length(x)
    p <- c(0, x, 0)
    which(x > p[seq_len(m)] & x > p[seq_len(m) + 2L])
  }
  for (it in seq_len(10000L)) {
    pk <- peaks(h)
    if (length(pk) == 2L) {
      v <- pk[1] + which.min(h[pk[1]:pk[2]]) - 1L
      return(centers[v])
    }
    if (length(pk) < 2L) return(NA_real_)
    m <- length(h)
    p <- c(h[1], h, h[m])
    h <- (p[seq_len(m)] + p[seq_len(m) + 1L] + p[seq_len(m) + 2L]) / 3
  }
  NA_real_
}

#' Scrublet-style scores for one sample.
#' @param counts genes x cells dgCMatrix of raw counts.
#' @return list(score, call, threshold, n_sim, k, notes)
scp_dbl_one_sample <- function(counts, seed, n_hvg = SCP_DBL_N_HVG, n_pcs = SCP_DBL_N_PCS, device = "cpu",
                               sim_ratio = SCP_DBL_SIM_RATIO, expected_rate = SCP_DBL_EXPECTED_RATE) {
  notes <- character()
  n <- ncol(counts)
  counts <- counts[Matrix::rowSums(counts > 0) >= 3, , drop = FALSE]
  if (nrow(counts) < 10) stop("fewer than 10 genes are expressed in at least 3 cells")
  n_sim <- as.integer(round(sim_ratio * n))
  set.seed(seed)
  a <- sample.int(n, n_sim, replace = TRUE)
  b <- sample.int(n, n_sim, replace = TRUE)
  same <- a == b
  b[same] <- (b[same] %% n) + 1L
  sim <- counts[, a, drop = FALSE] + counts[, b, drop = FALSE]

  target <- mean(Matrix::colSums(counts))
  obs_log <- .dbl_lognorm(counts, target)
  sim_log <- .dbl_lognorm(sim, target)
  # As Scrublet (min_gene_variability_pctl = 85): at most the top 15% of expressed genes.
  hv <- .dbl_hvg(obs_log, min(n_hvg, max(100L, as.integer(round(0.15 * nrow(counts))))))
  if (length(hv) < 10) stop("fewer than 10 variable genes")
  obs_log <- obs_log[hv, , drop = FALSE]
  sim_log <- sim_log[hv, , drop = FALSE]

  npcs <- as.integer(min(n_pcs, length(hv) - 1L, n - 1L))
  pca <- scfastR::scfast_pca(obs_log, npcs = npcs, seed = seed, device = device)
  sd_g <- ifelse(pca$gene_sd > 0, pca$gene_sd, 1)
  vs <- pca$loadings / sd_g
  off <- as.numeric(crossprod(pca$gene_mean / sd_g, pca$loadings))
  e_sim <- as.matrix(Matrix::crossprod(sim_log, vs))
  e_sim <- sweep(e_sim, 2, off, "-")
  emb <- rbind(unname(pca$embeddings), unname(e_sim))

  k <- max(5L, as.integer(round(0.5 * sqrt(n))))
  k_adj <- as.integer(round(k * (1 + n_sim / n)))
  k_adj <- min(k_adj, nrow(emb) - 2L)
  knn <- scfastR::scfast_knn(emb, k = k_adj + 1L, method = "auto", seed = seed, device = device)
  nn <- .dbl_drop_self(knn$idx)
  score_all <- .dbl_adjust(rowSums(nn > n), ncol(nn), n_sim / n, expected_rate)
  score <- score_all[seq_len(n)]
  sim_score <- score_all[n + seq_len(n_sim)]

  thr <- .dbl_threshold(sim_score)
  # Homotypic simulated doublets look like singlets, which pulls the valley around
  # (above the detectable real doublets on some samples, into the singlet bulk on others).
  # Keep it inside [median + 2 MAD, median + 2.5 MAD] of the observed scores.
  md <- stats::median(score)
  mm <- stats::mad(score)
  if (is.finite(thr) && is.finite(mm) && mm > 0) {
    lo <- md + 2 * mm
    hi <- md + 2.5 * mm
    thr2 <- min(max(thr, lo), hi)
    if (thr2 != thr) notes <- c(notes, sprintf("valley threshold %.3f moved to %.3f (observed-score band %.3f-%.3f)", thr, thr2, lo, hi))
    thr <- thr2
  }
  if (!is.finite(thr)) {
    thr <- (stats::median(sim_score) + stats::median(score)) / 2
    notes <- c(notes, sprintf("simulated-doublet score distribution is not bimodal; threshold set to %.3g (midpoint of the medians)", thr))
  }
  list(score = score, call = score >= thr, threshold = thr, n_sim = n_sim, k = k_adj, notes = notes)
}

#' Coarse clusters of all cells (PCA -> kNN -> SNN -> Leiden).
#' @param counts genes x cells dgCMatrix of raw counts.
#' @return character vector of labels ("0", "1", ...; largest cluster first), one per cell.
scp_dbl_precluster <- function(counts, seed, p = SCP_PRECLUSTER, device = "cpu") {
  n <- ncol(counts)
  logm <- .dbl_lognorm(counts, 1e4)
  hv <- .dbl_hvg(logm, p$n_hvg)
  npcs <- as.integer(min(p$n_pcs, length(hv) - 1L, n - 1L))
  if (length(hv) < 3 || npcs < 2L || n < 10L) return(rep("0", n))
  pca <- scfastR::scfast_pca(logm[hv, , drop = FALSE], npcs = npcs, seed = seed, device = device)
  emb <- pca$embeddings
  knn <- scfastR::scfast_knn(emb, k = min(p$k, n), method = "auto", seed = seed, device = device)
  g <- scfastR::scfast_snn(knn$idx, prune = p$prune)
  lab <- scfastR::scfast_leiden(g, resolution = p$resolution, seed = seed, device = device)
  as.character(as.integer(lab))
}

#' Doublet scores/calls and pre-clusters for a whole dataset.
#' @param counts genes x cells dgCMatrix (raw counts), cells named.
#' @param sample_id per-cell sample label (character, same length as ncol(counts)).
#' @return list(table = data.frame(cell_id, doublet_score, doublet_call, pre_cluster)
#'   with score numeric (NA if skipped), call logical, pre_cluster character;
#'   thresholds = named numeric per sample; notes = character).
# device: "cpu" or "cuda" (S1a passes "cuda" on a GPU plan when the engine is scfast).
scp_doublets_scfast <- function(counts, sample_id, seed, device = "cpu") {
  if (!scp_has_pkg("scfastR")) scp_stop("scfastR is not installed")
  counts <- scp_as_dgc(counts)
  n <- ncol(counts)
  score <- rep(NA_real_, n)
  call <- rep(FALSE, n)
  notes <- character()
  thr <- numeric()
  for (s in sort(unique(sample_id))) {
    idx <- which(sample_id == s)
    if (length(idx) < SCP_DBL_MIN_CELLS) {
      notes <- c(notes, sprintf("sample %s: %d cells < %d; doublet detection skipped (score NA, call FALSE)",
                                s, length(idx), SCP_DBL_MIN_CELLS))
      next
    }
    r <- tryCatch(scp_dbl_one_sample(counts[, idx, drop = FALSE], seed, device = device),
                  error = function(e) conditionMessage(e))
    if (is.character(r)) {
      notes <- c(notes, sprintf("sample %s: doublet detection failed (%s); score NA, call FALSE", s, r))
      next
    }
    score[idx] <- r$score
    call[idx] <- r$call
    thr[s] <- r$threshold
    notes <- c(notes, sprintf("sample %s: %d cells, %d simulated, k=%d, threshold %.3f, %d called doublets (%.1f%%)",
                              s, length(idx), r$n_sim, r$k, r$threshold, sum(r$call), 100 * mean(r$call)),
               if (length(r$notes)) paste0("sample ", s, ": ", r$notes))
  }
  pre <- scp_dbl_precluster(counts, seed, device = device)
  list(table = data.frame(cell_id = colnames(counts), doublet_score = score, doublet_call = call,
                          pre_cluster = pre, stringsAsFactors = FALSE),
       thresholds = thr, notes = notes)
}

# Columns as S1b writes them: score "%.6g" or "NA", call TRUE/FALSE.
scp_dbl_format <- function(tab) {
  data.frame(cell_id = tab$cell_id,
             doublet_score = ifelse(is.finite(tab$doublet_score), sprintf("%.6g", tab$doublet_score), "NA"),
             doublet_call = ifelse(tab$doublet_call, "TRUE", "FALSE"),
             pre_cluster = as.character(tab$pre_cluster),
             stringsAsFactors = FALSE)
}
