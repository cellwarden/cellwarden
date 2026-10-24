//! Fused preprocessing: `NormalizeData` + `FindVariableFeatures(vst)` + the gene statistics
//! of `ScaleData`, from one streaming pass over the raw counts (docs/scfast-integration.md,
//! Phase 9).
//!
//! Input: the raw counts as a borrowed CSR (cells x genes), i.e. a Seurat `counts`
//! `dgCMatrix` (genes x cells, CSC) read in place. Output, with the same `indptr` /
//! `indices`:
//!
//! - the log-normalised values `log1p(x / library_size * scale_factor)` (the `data` layer);
//! - the vst statistics of every gene: mean and variance of the raw counts, expected
//!   variance from a loess of `log10(variance)` on `log10(mean)`, and the standardised
//!   variance clipped at `clip` (default `sqrt(n_cells)`);
//! - the HVGs: the top `n_hvg` genes by standardised variance (ties by lower index, like
//!   R's stable `order()`), with the mean and standard deviation of their log-normalised
//!   values for the implicit PCA ([`crate::pca::implicit_scaled_pca`]).
//!
//! # Passes over the data
//!
//! Pass 1 (parallel over fixed row blocks, deterministic tree reduction as in `pca`) reads
//! each cell once: library size, the log-normalised values written straight to the output,
//! and per-gene sums of `x`, `x^2`, `y`, `y^2` (x raw, y log-normalised), stored-entry
//! counts and maxima. Everything per-gene (means, variances, the loess, the ranking) then
//! works on `n_genes`-sized vectors.
//!
//! The clipped standardised variance `sum_i min(clip, (x_ij - mu_j) / sd_j)^2 / (n - 1)`
//! cannot come from those sums alone because `sd_j` (the loess fit) is only known after the
//! pass. Genes whose largest count stays below `mu_j + clip * sd_j` are never clipped, so
//! their value is exactly `variance_j / variance_expected_j`; only the remaining genes
//! need a second read of the counts (pass 2, same row blocks, touching flagged genes only).
//! That pass is skipped entirely when no gene is clipped.
//!
//! # Loess
//!
//! `loess` is written here, as R's default `loess(..., span, degree = 2, family =
//! "gaussian")` that Seurat calls: `surface = "interpolate"`, `cell = 0.2`. R builds a
//! k-d tree over `x`, fits the local quadratic (tricube weights, neighbourhood of
//! `floor(n * span)` points) only at the tree's vertices, keeping the fitted value and
//! slope there, and evaluates every point by cubic Hermite interpolation between the two
//! vertices of its cell ([`loess_interpolate`], a port of `lowesb` / `lowese` in R's
//! `loessf.f`; matches R to ~1e-11). [`loess_direct`] (`surface = "direct"`, the local
//! quadratic at every point) is kept for the R package.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::types::{CsrView, Idx, Real};

/// Upper bound on the number of row blocks for the reductions.
const MAX_BLOCKS: usize = 256;
/// Lower bound on rows per reduction block.
const MIN_BLOCK_ROWS: usize = 64;
/// A leaf should do at least this many times more work than zeroing and merging its
/// per-gene accumulators costs.
const ACC_WORK_RATIO: usize = 16;

/// Counts below this get their log-normalised value from a per-cell table.
const SMALL_COUNTS: usize = 32;

/// Value type of the log-normalised output (`f64` for an R `dgCMatrix`, `f32` for AnnData).
pub trait OutValue: Copy + Send + Sync + 'static {
    fn from_f64(v: f64) -> Self;
}
impl OutValue for f32 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}
impl OutValue for f64 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
}

/// Parameters, with Seurat's defaults.
#[derive(Clone, Debug)]
pub struct PreprocessParams {
    /// Number of variable genes to select (Seurat `nfeatures`, default 2000). Capped at the
    /// number of genes.
    pub n_hvg: usize,
    /// `NormalizeData(scale.factor)`, default 1e4.
    pub scale_factor: f64,
    /// Loess span (`FindVariableFeatures(loess.span)`), default 0.3.
    pub span: f64,
    /// Clip for the standardised values (`clip.max`); `None` means `sqrt(n_cells)`.
    pub clip: Option<f64>,
}

impl Default for PreprocessParams {
    fn default() -> Self {
        Self { n_hvg: 2000, scale_factor: 1e4, span: 0.3, clip: None }
    }
}

/// Output of [`preprocess`].
#[derive(Clone, Debug)]
pub struct PreprocessResult<O: OutValue> {
    /// Per-cell library size (sum of counts).
    pub library_size: Vec<f64>,
    /// `log1p(x / library_size * scale_factor)` for every stored entry, in the order of the
    /// input `data` (same `indptr` and `indices`). Entries of a cell with library size 0
    /// are 0.
    pub data: Vec<O>,
    /// vst: mean of the raw counts per gene (Seurat `vst.mean`).
    pub mean: Vec<f64>,
    /// vst: variance of the raw counts per gene, n - 1 denominator (`vst.variance`).
    pub variance: Vec<f64>,
    /// vst: loess-expected variance (`vst.variance.expected`); 0 for constant genes.
    pub variance_expected: Vec<f64>,
    /// vst: clipped standardised variance (`vst.variance.standardized`); 0 for constant genes.
    pub variance_standardized: Vec<f64>,
    /// The clip actually used.
    pub clip: f64,
    /// Indices of the variable genes, most variable first (Seurat `VariableFeatures()` order).
    pub hvg: Vec<u32>,
    /// Mean of the log-normalised values of each HVG (same order as `hvg`).
    pub hvg_mean: Vec<f64>,
    /// Sample standard deviation (n - 1) of the log-normalised values of each HVG; the
    /// numbers `ScaleData` centres and scales with.
    pub hvg_sd: Vec<f64>,
    /// First cell whose column indices do not strictly increase (`None`: canonical CSR).
    pub unsorted_row: Option<usize>,
}

/// Pass-1 sums of one gene, kept together so a stored entry updates one cache line.
#[derive(Clone, Copy, Default)]
#[repr(align(64))]
struct GeneAcc {
    sum: f64,
    sumsq: f64,
    ysum: f64,
    ysumsq: f64,
    max: f64,
    nnz: u64,
}

/// Per-block accumulators of pass 1.
struct Acc {
    genes: Vec<GeneAcc>,
    /// First row whose column indices do not strictly increase.
    unsorted: Option<usize>,
}

impl Acc {
    fn new(p: usize) -> Self {
        Self { genes: vec![GeneAcc::default(); p], unsorted: None }
    }
    fn merge(&mut self, b: Acc) {
        self.unsorted = self.unsorted.or(b.unsorted);
        for (a, b) in self.genes.iter_mut().zip(&b.genes) {
            a.sum += b.sum;
            a.sumsq += b.sumsq;
            a.ysum += b.ysum;
            a.ysumsq += b.ysumsq;
            a.max = a.max.max(b.max);
            a.nnz += b.nnz;
        }
    }
}

/// Fixed partition of the rows into blocks; depends on the input shape only, never on the
/// thread count, so the reduction order (and the result) is reproducible.
#[derive(Clone, Copy, Debug)]
struct Blocks {
    rows: usize,
    count: usize,
}

impl Blocks {
    fn new(n: usize, p: usize, nnz: usize) -> Self {
        let avg_row_nnz = (nnz / n.max(1)).max(1);
        let for_acc = (ACC_WORK_RATIO * (p + 1)).div_ceil(avg_row_nnz);
        let rows = n.div_ceil(MAX_BLOCKS).max(for_acc).max(MIN_BLOCK_ROWS).min(n.max(1));
        Self { rows, count: n.div_ceil(rows).max(1) }
    }
    #[inline]
    fn range(&self, b: usize, n: usize) -> (usize, usize) {
        (b * self.rows, ((b + 1) * self.rows).min(n))
    }
}

fn validate<I: Idx, T: Real>(x: &CsrView<'_, I, T>, params: &PreprocessParams) -> Result<()> {
    if x.n_rows == 0 || x.n_cols == 0 {
        return Err(Error::InvalidInput(format!("empty matrix ({} cells x {} genes)", x.n_rows, x.n_cols)));
    }
    if params.n_hvg == 0 {
        return Err(Error::InvalidParameter("n_hvg must be >= 1".into()));
    }
    if !params.scale_factor.is_finite() || params.scale_factor <= 0.0 {
        return Err(Error::InvalidParameter(format!("scale_factor must be positive, got {}", params.scale_factor)));
    }
    if !params.span.is_finite() || params.span <= 0.0 {
        return Err(Error::InvalidParameter(format!("span must be positive, got {}", params.span)));
    }
    if let Some(c) = params.clip {
        if c.is_nan() || c <= 0.0 {
            return Err(Error::InvalidParameter(format!("clip must be positive, got {c}")));
        }
    }
    Ok(())
}

/// Fused normalisation and vst statistics of a counts matrix (cells x genes).
///
/// See the module documentation. Errors: `InvalidInput` for malformed CSR structure
/// (including duplicate column indices in a row), negative or non-finite counts, or an empty
/// matrix; `InvalidParameter` for a bad `n_hvg`, `scale_factor`, `span` or `clip`.
pub fn preprocess<I: Idx, T: Real, O: OutValue>(x: CsrView<'_, I, T>, params: &PreprocessParams) -> Result<PreprocessResult<O>> {
    validate(&x, params)?;
    let (n, p, nnz) = (x.n_rows, x.n_cols, x.data.len());
    let blocks = Blocks::new(n, p, nnz);
    let clip = params.clip.unwrap_or_else(|| (n as f64).sqrt());

    // ---- pass 1: library sizes, log-normalised values, per-gene sums
    let mut data: Vec<O> = vec![O::from_f64(0.0); nnz];
    let mut library_size = vec![0.0f64; n];
    let acc = pass1(&x, params.scale_factor, blocks, 0, blocks.count, &mut data, &mut library_size)?;

    let column = |f: fn(&GeneAcc) -> f64| acc.genes.iter().map(f).collect::<Vec<f64>>();
    let gene_nnz: Vec<u64> = acc.genes.iter().map(|g| g.nnz).collect();
    let VstSums { mean, variance, variance_expected, mut variance_standardized, needs_clip, sd } =
        vst_from_sums(&column(|g| g.sum), &column(|g| g.sumsq), &column(|g| g.max), &gene_nnz, n, params.span, clip)?;
    let nf = n as f64;
    let any_clip = needs_clip.iter().any(|&c| c);
    if any_clip {
        let clipped = pass2(&x, &mean, &sd, &needs_clip, clip, blocks)?;
        for j in 0..p {
            if needs_clip[j] {
                let zeros = (n as u64 - acc.genes[j].nnz) as f64;
                let z0 = mean[j] / sd[j];
                variance_standardized[j] = (clipped[j] + zeros * z0 * z0) / (nf - 1.0);
            }
        }
    }

    // ---- ranking: top n_hvg by standardised variance, ties by lower index
    let mut order: Vec<u32> = (0..p as u32).collect();
    order.sort_by(|&a, &b| {
        let (va, vb) = (variance_standardized[a as usize], variance_standardized[b as usize]);
        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
    });
    order.truncate(params.n_hvg.min(p));
    let hvg_mean: Vec<f64> = order.iter().map(|&j| acc.genes[j as usize].ysum / nf).collect();
    let hvg_sd: Vec<f64> = order
        .iter()
        .zip(&hvg_mean)
        .map(|(&j, &m)| {
            let j = j as usize;
            if n > 1 {
                ((acc.genes[j].ysumsq - acc.genes[j].ysum * m).max(0.0) / (nf - 1.0)).sqrt()
            } else {
                0.0
            }
        })
        .collect();

    Ok(PreprocessResult {
        library_size,
        data,
        mean,
        variance,
        variance_expected,
        variance_standardized,
        clip,
        hvg: order,
        hvg_mean,
        hvg_sd,
        unsorted_row: acc.unsorted,
    })
}

/// The per-gene vst statistics of [`preprocess`] from the pass-1 sums of the raw counts
/// (`sum`, `sumsq`, `max`, stored-entry count `nnz` per gene, over `n` cells), shared with
/// the device front end. `variance_standardized` is final except for the genes flagged in
/// `needs_clip`, whose clipped sum `sum_i min(clip, (x_ij - mean_j) / sd_j)^2` over the stored
/// entries must still be added: `(clipped + (n - nnz_j) (mean_j / sd_j)^2) / (n - 1)`.
pub fn vst_from_sums(sum: &[f64], sumsq: &[f64], max: &[f64], nnz: &[u64], n: usize, span: f64, clip: f64) -> Result<VstSums> {
    let p = sum.len();
    let nf = n as f64;
    let mut mean = vec![0.0; p];
    let mut variance = vec![0.0; p];
    for j in 0..p {
        let m = sum[j] / nf;
        mean[j] = m;
        if n > 1 {
            // Seurat: sum((x - mu)^2) / (n - 1), zeros included.
            variance[j] = ((sumsq[j] - sum[j] * m) / (nf - 1.0)).max(0.0);
            // A gene stored in every cell with a single value is constant too.
            if nnz[j] as usize == n && max[j] * max[j] * nf == sumsq[j] && max[j] * nf == sum[j] {
                variance[j] = 0.0;
            }
        }
    }
    let not_const: Vec<usize> = (0..p).filter(|&j| variance[j] > 0.0).collect();
    let mut variance_expected = vec![0.0; p];
    if !not_const.is_empty() {
        let lx: Vec<f64> = not_const.iter().map(|&j| mean[j].log10()).collect();
        let ly: Vec<f64> = not_const.iter().map(|&j| variance[j].log10()).collect();
        let fit = loess_interpolate(&lx, &ly, span)?;
        for (k, &j) in not_const.iter().enumerate() {
            variance_expected[j] = 10f64.powf(fit[k]);
        }
    }

    // Standardised variance: exact from the sums unless the gene has clipped values.
    let mut variance_standardized = vec![0.0; p];
    let mut needs_clip = vec![false; p];
    let sd: Vec<f64> = variance_expected.iter().map(|v| v.sqrt()).collect();
    for j in 0..p {
        if sd[j] == 0.0 {
            continue;
        }
        if max[j] > mean[j] + clip * sd[j] {
            needs_clip[j] = true;
        } else {
            variance_standardized[j] = variance[j] / variance_expected[j];
        }
    }
    Ok(VstSums { mean, variance, variance_expected, variance_standardized, needs_clip, sd })
}

/// Output of [`vst_from_sums`]; `sd` is `sqrt(variance_expected)`.
#[derive(Clone, Debug)]
pub struct VstSums {
    pub mean: Vec<f64>,
    pub variance: Vec<f64>,
    pub variance_expected: Vec<f64>,
    pub variance_standardized: Vec<f64>,
    pub needs_clip: Vec<bool>,
    pub sd: Vec<f64>,
}

/// Pass 1 over the row blocks `lo..hi` (recursive binary split, `rayon::join` on the two
/// halves, accumulators merged in a fixed order). `data` and `library_size` are the output
/// slices for exactly these rows.
fn pass1<I: Idx, T: Real, O: OutValue>(
    x: &CsrView<'_, I, T>,
    scale_factor: f64,
    blocks: Blocks,
    lo: usize,
    hi: usize,
    data: &mut [O],
    library_size: &mut [f64],
) -> Result<Acc> {
    let (n, p, nnz) = (x.n_rows, x.n_cols, x.data.len());
    if hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        let (r0, _) = blocks.range(lo, n);
        let (rm, _) = blocks.range(mid, n);
        let s0 = x.indptr[r0].to_usize();
        let sm = x.indptr[rm].to_usize();
        if sm < s0 || sm - s0 > data.len() || rm - r0 > library_size.len() {
            return Err(Error::InvalidInput("indptr is not non-decreasing or exceeds nnz".into()));
        }
        let (da, db) = data.split_at_mut(sm - s0);
        let (la, lb) = library_size.split_at_mut(rm - r0);
        let (a, b) = rayon::join(
            || pass1(x, scale_factor, blocks, lo, mid, da, la),
            || pass1(x, scale_factor, blocks, mid, hi, db, lb),
        );
        let mut a = a?;
        a.merge(b?);
        return Ok(a);
    }
    let (r0, r1) = blocks.range(lo, n);
    let base = x.indptr[r0].to_usize();
    let mut acc = Acc::new(p);
    // Row stamp per gene to find duplicate column indices in rows that are not strictly
    // increasing (a strictly increasing row has none).
    let mut seen: Vec<u32> = Vec::new();
    let mut small = [f64::NAN; SMALL_COUNTS];
    for i in r0..r1 {
        let (s, e) = (x.indptr[i].to_usize(), x.indptr[i + 1].to_usize());
        if s > e || e > nnz || s < base || e - base > data.len() {
            return Err(Error::InvalidInput(format!("indptr is invalid at row {i}")));
        }
        let stamp = (i - r0) as u32;
        let idx = &x.indices[s..e];
        let val = &x.data[s..e];
        let mut lib = 0.0f64;
        let mut prev = None;
        let mut increasing = true;
        for (j, v) in idx.iter().zip(val) {
            let j = j.to_usize();
            increasing &= prev.is_none_or(|q| j > q);
            prev = Some(j);
            if j >= p {
                return Err(Error::InvalidInput(format!("column index {j} out of range for {p} columns")));
            }
            let v = v.to_f64();
            if !v.is_finite() || v < 0.0 {
                return Err(Error::InvalidInput(format!("counts must be finite and non-negative; row {i}, column {j} is {v}")));
            }
            lib += v;
        }
        if !increasing {
            acc.unsorted.get_or_insert(i);
            seen.resize(p, u32::MAX);
            for j in idx {
                let j = j.to_usize();
                if seen[j] == stamp {
                    return Err(Error::InvalidInput(format!("duplicate column index {j} in row {i}; sum duplicates first")));
                }
                seen[j] = stamp;
            }
        }
        library_size[i - r0] = lib;
        let factor = if lib > 0.0 { scale_factor / lib } else { 0.0 };
        let out = &mut data[s - base..e - base];
        small.fill(f64::NAN);
        for ((j, v), o) in idx.iter().zip(val).zip(out.iter_mut()) {
            let j = j.to_usize();
            let v = v.to_f64();
            // Most counts are small integers: their log1p is computed once per cell.
            let y = match small.get_mut(v as usize) {
                Some(t) if v == (v as usize) as f64 => {
                    if t.is_nan() {
                        *t = (v * factor).ln_1p();
                    }
                    *t
                }
                _ => (v * factor).ln_1p(),
            };
            *o = O::from_f64(y);
            let g = &mut acc.genes[j];
            g.sum += v;
            g.sumsq += v * v;
            g.ysum += y;
            g.ysumsq += y * y;
            if v > g.max {
                g.max = v;
            }
            g.nnz += 1;
        }
    }
    Ok(acc)
}

/// Pass 2: per flagged gene, `sum over stored entries of min(clip, (x - mu) / sd)^2`.
fn pass2<I: Idx, T: Real>(x: &CsrView<'_, I, T>, mean: &[f64], sd: &[f64], flag: &[bool], clip: f64, blocks: Blocks) -> Result<Vec<f64>> {
    fn go<I: Idx, T: Real>(x: &CsrView<'_, I, T>, mean: &[f64], sd: &[f64], flag: &[bool], clip: f64, blocks: Blocks, lo: usize, hi: usize) -> Vec<f64> {
        let (n, p) = (x.n_rows, x.n_cols);
        if hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            let (mut a, b) = rayon::join(
                || go(x, mean, sd, flag, clip, blocks, lo, mid),
                || go(x, mean, sd, flag, clip, blocks, mid, hi),
            );
            for (u, v) in a.iter_mut().zip(b) {
                *u += v;
            }
            return a;
        }
        let (r0, r1) = blocks.range(lo, n);
        let mut acc = vec![0.0f64; p];
        for i in r0..r1 {
            let (idx, val) = x.row(i);
            for (j, v) in idx.iter().zip(val) {
                let j = j.to_usize();
                if flag[j] {
                    let z = ((v.to_f64() - mean[j]) / sd[j]).min(clip);
                    acc[j] += z * z;
                }
            }
        }
        acc
    }
    Ok(go(x, mean, sd, flag, clip, blocks, 0, blocks.count))
}

// ------------------------------------------------------------------ loess

/// Local quadratic loess of `y` on `x` evaluated at every `x[i]` (R `loess(y ~ x, span,
/// degree = 2, family = "gaussian", surface = "direct")`).
///
/// For each point, the `q = floor(n * span)` nearest points (itself included) get tricube
/// weights `(1 - (d / h)^3)^3` with `h` the distance to the q-th nearest; the weighted
/// least-squares quadratic in `(x - x_i)` is solved from its normal equations (3 x 3,
/// f64; the centring keeps it well conditioned). When the neighbourhood is degenerate (all
/// points at the same `x`, so `h = 0`), the fit falls back to the weighted mean of the
/// points at that `x`, which is what the local polynomial collapses to. With `span > 1`
/// every point is a neighbour and `h` is scaled by `sqrt(span)`, as R does.
pub fn loess_direct(x: &[f64], y: &[f64], span: f64) -> Result<Vec<f64>> {
    let n = x.len();
    let Some((order, xs, ys)) = sorted_xy(x, y)? else { return Ok(Vec::new()) };
    let q = neighbourhood_size(n, span);
    // R scales the bandwidth by sqrt(span) when span > 1 (checked against stats::loess).
    let bandwidth_scale = span.max(1.0).sqrt();

    let fitted_sorted: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|k| {
            let x0 = xs[k];
            // Window of the q nearest in sorted order: grow from k outwards.
            let (mut lo, mut hi) = (k, k); // inclusive
            while hi - lo + 1 < q {
                let dl = if lo > 0 { x0 - xs[lo - 1] } else { f64::INFINITY };
                let dr = if hi + 1 < n { xs[hi + 1] - x0 } else { f64::INFINITY };
                if dl <= dr {
                    lo -= 1;
                } else {
                    hi += 1;
                }
            }
            let h = (x0 - xs[lo]).max(xs[hi] - x0) * bandwidth_scale;
            if h <= 0.0 {
                // Every neighbour sits at x0: the local polynomial is the weighted mean of
                // the points at x0 (all with weight 1).
                let (mut s, mut c) = (0.0, 0.0);
                let (mut a, mut b) = (k, k);
                while a > 0 && xs[a - 1] == x0 {
                    a -= 1;
                }
                while b + 1 < n && xs[b + 1] == x0 {
                    b += 1;
                }
                for i in a..=b {
                    s += ys[i];
                    c += 1.0;
                }
                return s / c;
            }
            // Points at distance exactly h get weight 0, so extend over ties harmlessly.
            let (mut a, mut b) = (lo, hi);
            while a > 0 && x0 - xs[a - 1] < h {
                a -= 1;
            }
            while b + 1 < n && xs[b + 1] - x0 < h {
                b += 1;
            }
            // Weighted normal equations for [1, u, u^2], u = x - x0.
            let (mut s0, mut s1, mut s2, mut s3, mut s4) = (0.0, 0.0, 0.0, 0.0, 0.0);
            let (mut t0, mut t1, mut t2) = (0.0, 0.0, 0.0);
            for i in a..=b {
                let u = xs[i] - x0;
                let r = u.abs() / h;
                if r >= 1.0 {
                    continue;
                }
                let w = {
                    let t = 1.0 - r * r * r;
                    t * t * t
                };
                let (u2, wy) = (u * u, w * ys[i]);
                s0 += w;
                s1 += w * u;
                s2 += w * u2;
                s3 += w * u2 * u;
                s4 += w * u2 * u2;
                t0 += wy;
                t1 += wy * u;
                t2 += wy * u2;
            }
            solve3([[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]], [t0, t1, t2]).map_or_else(
                || {
                    // Singular (fewer than 3 distinct x in the window): weighted mean.
                    if s0 > 0.0 {
                        t0 / s0
                    } else {
                        ys[k]
                    }
                },
                |c| c[0],
            )
        })
        .collect();

    let mut fitted = vec![0.0; n];
    for (k, &i) in order.iter().enumerate() {
        fitted[i] = fitted_sorted[k];
    }
    Ok(fitted)
}

/// Validates `x` / `y` and returns the permutation sorting `x` (ties by index) with the
/// sorted `x` and `y`; `None` for empty input.
fn sorted_xy(x: &[f64], y: &[f64]) -> Result<Option<(Vec<usize>, Vec<f64>, Vec<f64>)>> {
    let n = x.len();
    if y.len() != n {
        return Err(Error::InvalidInput(format!("loess: x has {n} values, y has {}", y.len())));
    }
    if n == 0 {
        return Ok(None);
    }
    if x.iter().chain(y).any(|v| !v.is_finite()) {
        return Err(Error::InvalidInput("loess: non-finite value".into()));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(a.cmp(&b)));
    let xs = order.iter().map(|&i| x[i]).collect();
    let ys = order.iter().map(|&i| y[i]).collect();
    Ok(Some((order, xs, ys)))
}

/// R's `nf = min(n, floor(n * span + 1e-5))` (`loess_workspace` in loessc.c), at least 1.
fn neighbourhood_size(n: usize, span: f64) -> usize {
    ((n as f64 * span + 1e-5).floor() as usize).clamp(1, n)
}

/// R's default `loess(y ~ x, span, degree = 2, family = "gaussian")`: `surface =
/// "interpolate"`, `cell = 0.2`, the fit evaluated at every `x[i]`. A port of the 1-D path
/// of `lowesb` / `lowese` in R's `src/library/stats/src/loessf.f`:
///
/// - `ehg126`: the bounding box of `x`, widened by 0.5% of its width on each side, gives
///   the first two vertices;
/// - `ehg124`: cells are split breadth-first at the median point (1-based position
///   `floor((l + u) / 2)`), moved to the nearest position (searched in the order 0, +1, -1,
///   +2, ...) whose value differs from the next, until a cell holds at most
///   `floor(n * span * cell)` points or the split value lands on a cell bound;
/// - `ehg127`: at each vertex `v`, the weighted least-squares quadratic in `x - v` over the
///   `nf` nearest points, tricube weights on `|x - v| / h`, `h` the `nf`-th distance
///   (scaled by `sqrt(span)` for `span > 1`), keeping the value and the slope;
/// - `ehg128`: each point is the cubic Hermite interpolant of the values and slopes at the
///   two vertices of the leaf cell holding it (`x <= split` descends left).
///
/// R solves the local fits by QR and an SVD pseudo-inverse; here they are the 3 x 3 normal
/// equations in `(x - v) / h`. Singular fits (fewer than 3 distinct `x` near a vertex)
/// fall back to the weighted mean with slope 0, as [`loess_direct`] does.
pub fn loess_interpolate(x: &[f64], y: &[f64], span: f64) -> Result<Vec<f64>> {
    let n = x.len();
    let Some((_, xs, ys)) = sorted_xy(x, y)? else { return Ok(Vec::new()) };
    let nf = neighbourhood_size(n, span);
    // lowesb: fc = floor(n * (span * cell)); R passes the product span * cell.
    let fc = (n as f64 * (span * 0.2)).floor() as usize;
    // loess_workspace: nvmax = ncmax = max(200, n).
    let nvmax = n.max(200);

    let (lo, hi) = (xs[0], xs[n - 1]);
    let margin = 0.005 * (hi - lo).max(1e-10 * lo.abs().max(hi.abs()) + 1e-30);
    let mut vert = vec![lo - margin, hi + margin];
    struct Cell {
        a: usize, // points xs[a..b]
        b: usize,
        vlo: usize,
        vhi: usize,
        split: Option<(f64, usize, usize)>,
    }
    let mut cells = vec![Cell { a: 0, b: n, vlo: 0, vhi: 1, split: None }];
    let mut p = 0;
    while p < cells.len() {
        let Cell { a, b, vlo, vhi, .. } = cells[p];
        let mut leaf = b - a <= fc || vert[vhi] - vert[vlo] <= 0.0 || nvmax < cells.len() + 2 || nvmax < vert.len() + 1;
        if !leaf {
            // All ties go with the right son: move m to the nearest boundary between values.
            let mut m = (a + 1 + b) / 2 - 1;
            let mut off: isize = 0;
            loop {
                let mm = m as isize + off;
                if mm >= b as isize - 1 || mm < a as isize {
                    break;
                }
                let mm = mm as usize;
                if xs[mm] != xs[mm + 1] {
                    m = mm;
                    break;
                }
                off = -off;
                if off >= 0 {
                    off += 1;
                }
            }
            let xi = xs[m];
            leaf = vert[vlo] == xi || vert[vhi] == xi;
            if !leaf {
                let v = vert.len();
                vert.push(xi);
                let (left, right) = (cells.len(), cells.len() + 1);
                cells.push(Cell { a, b: m + 1, vlo, vhi: v, split: None });
                cells.push(Cell { a: m + 1, b, vlo: v, vhi, split: None });
                cells[p].split = Some((xi, left, right));
            }
        }
        p += 1;
    }

    let fits: Vec<(f64, f64)> = vert.iter().map(|&v| vertex_fit(&xs, &ys, v, nf, span)).collect();

    Ok(x.iter()
        .map(|&z| {
            let mut j = 0;
            while let Some((xi, left, right)) = cells[j].split {
                j = if z <= xi { left } else { right };
            }
            let (lo, hi) = (cells[j].vlo, cells[j].vhi);
            let (v0, v1) = (vert[lo], vert[hi]);
            let ((g0, s0), (g1, s1)) = (fits[lo], fits[hi]);
            let h = (z - v0) / (v1 - v0);
            let phi0 = (1.0 - h) * (1.0 - h) * (1.0 + 2.0 * h);
            let phi1 = h * h * (3.0 - 2.0 * h);
            let psi0 = h * (1.0 - h) * (1.0 - h);
            let psi1 = h * h * (h - 1.0);
            phi0 * g0 + phi1 * g1 + (psi0 * s0 + psi1 * s1) * (v1 - v0)
        })
        .collect())
}

/// Value and slope at `q` of the local quadratic over the `nf` points of sorted `xs`
/// nearest to `q` (R's `ehg127` at one vertex).
fn vertex_fit(xs: &[f64], ys: &[f64], q: f64, nf: usize, span: f64) -> (f64, f64) {
    let n = xs.len();
    let k = xs.partition_point(|&v| v < q);
    let (mut lo, mut hi) = (k, k); // window xs[lo..hi]
    while hi - lo < nf {
        let dl = if lo > 0 { q - xs[lo - 1] } else { f64::INFINITY };
        let dr = if hi < n { xs[hi] - q } else { f64::INFINITY };
        if dl <= dr {
            lo -= 1;
        } else {
            hi += 1;
        }
    }
    // rho: squared distance to the nf-th nearest, times span when span > 1.
    let rho = (q - xs[lo]).powi(2).max((xs[hi - 1] - q).powi(2)) * span.max(1.0);
    if !(rho > 0.0) {
        return (ys[lo..hi].iter().sum::<f64>() / (hi - lo) as f64, 0.0);
    }
    let scale = rho.sqrt();
    let (mut s0, mut s1, mut s2, mut s3, mut s4) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut t0, mut t1, mut t2) = (0.0, 0.0, 0.0);
    for i in lo..hi {
        let u = xs[i] - q;
        let r = (u * u / rho).sqrt();
        if r >= 1.0 {
            continue;
        }
        let w = {
            let t = 1.0 - r * r * r;
            t * t * t
        };
        let t = u / scale;
        let (tt, wy) = (t * t, w * ys[i]);
        s0 += w;
        s1 += w * t;
        s2 += w * tt;
        s3 += w * tt * t;
        s4 += w * tt * tt;
        t0 += wy;
        t1 += wy * t;
        t2 += wy * tt;
    }
    match solve3([[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]], [t0, t1, t2]) {
        Some(c) => (c[0], c[1] / scale),
        None if s0 > 0.0 => (t0 / s0, 0.0),
        None => (ys[lo], 0.0),
    }
}

/// Solution of the 3 x 3 system `a c = b` by Gaussian elimination with partial pivoting;
/// `None` when (numerically) singular.
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    let scale = a.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
    if !(scale > 0.0) {
        return None;
    }
    for c in 0..3 {
        let piv = (c..3).max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap())?;
        if a[piv][c].abs() <= 1e-12 * scale {
            return None;
        }
        a.swap(c, piv);
        b.swap(c, piv);
        for r in c + 1..3 {
            let f = a[r][c] / a[c][c];
            for k in c..3 {
                a[r][k] -= f * a[c][k];
            }
            b[r] -= f * b[c];
        }
    }
    let c2 = b[2] / a[2][2];
    let c1 = (b[1] - a[1][2] * c2) / a[1][1];
    let c0 = (b[0] - a[0][1] * c1 - a[0][2] * c2) / a[0][0];
    let c = [c0, c1, c2];
    c.iter().all(|v| v.is_finite()).then_some(c)
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense (cells x genes) -> CSR arrays, zeros dropped.
    fn csr(dense: &[Vec<f64>]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let mut indptr = vec![0i32];
        let (mut indices, mut data) = (Vec::new(), Vec::new());
        for row in dense {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    indices.push(j as i32);
                    data.push(v);
                }
            }
            indptr.push(data.len() as i32);
        }
        (indptr, indices, data)
    }

    /// Reference vst statistics computed the slow dense way, as Seurat's C++ does.
    fn reference(dense: &[Vec<f64>], span: f64, clip: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = dense.len();
        let p = dense[0].len();
        let mean: Vec<f64> = (0..p).map(|j| dense.iter().map(|r| r[j]).sum::<f64>() / n as f64).collect();
        let var: Vec<f64> = (0..p).map(|j| dense.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f64>() / (n as f64 - 1.0)).collect();
        let nc: Vec<usize> = (0..p).filter(|&j| var[j] > 0.0).collect();
        let lx: Vec<f64> = nc.iter().map(|&j| mean[j].log10()).collect();
        let ly: Vec<f64> = nc.iter().map(|&j| var[j].log10()).collect();
        let fit = loess_interpolate(&lx, &ly, span).unwrap();
        let mut ve = vec![0.0; p];
        for (k, &j) in nc.iter().enumerate() {
            ve[j] = 10f64.powf(fit[k]);
        }
        let vs: Vec<f64> = (0..p)
            .map(|j| {
                let sd = ve[j].sqrt();
                if sd == 0.0 {
                    return 0.0;
                }
                dense.iter().map(|r| ((r[j] - mean[j]) / sd).min(clip).powi(2)).sum::<f64>() / (n as f64 - 1.0)
            })
            .collect();
        (mean, var, ve, vs)
    }

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn simulate(n: usize, p: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut s = seed;
        let rates: Vec<f64> = (0..p).map(|_| 0.02 + 3.0 * lcg(&mut s).powi(3)).collect();
        (0..n)
            .map(|i| {
                let lib = 0.5 + lcg(&mut s);
                (0..p)
                    .map(|j| {
                        // Poisson-ish counts with a group effect on the first genes.
                        let mut r = rates[j] * lib;
                        if j < 20 && i % 3 == j % 3 {
                            r *= 6.0;
                        }
                        let mut k = 0.0;
                        let mut t = (-r).exp();
                        let mut u = lcg(&mut s);
                        while u > t {
                            u -= t;
                            k += 1.0;
                            t *= r / k;
                        }
                        k
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn normalisation_matches_definition() {
        let dense = vec![vec![1.0, 0.0, 3.0], vec![0.0, 0.0, 0.0], vec![2.0, 2.0, 0.0], vec![0.0, 5.0, 1.0]];
        let (ip, ix, dx) = csr(&dense);
        let x = CsrView::new(4, 3, &ip, &ix, &dx).unwrap();
        let r: PreprocessResult<f64> = preprocess(x, &PreprocessParams { n_hvg: 2, ..Default::default() }).unwrap();
        assert_eq!(r.library_size, vec![4.0, 0.0, 4.0, 6.0]);
        let expect: Vec<f64> = vec![(1.0 / 4.0 * 1e4f64).ln_1p(), (3.0 / 4.0 * 1e4f64).ln_1p(), (2.0 / 4.0 * 1e4f64).ln_1p(), (2.0 / 4.0 * 1e4f64).ln_1p(), (5.0 / 6.0 * 1e4f64).ln_1p(), (1.0 / 6.0 * 1e4f64).ln_1p()];
        for (a, b) in r.data.iter().zip(&expect) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(r.hvg.len(), 2);
        assert_eq!(r.clip, 2.0);
        // Per-gene means of the raw counts.
        assert!((r.mean[0] - 0.75).abs() < 1e-12 && (r.mean[1] - 1.75).abs() < 1e-12 && (r.mean[2] - 1.0).abs() < 1e-12);
        // Mean / sd of the log-normalised values of the first HVG, zeros included.
        let j = r.hvg[0] as usize;
        let col: Vec<f64> = (0..4).map(|i| if dense[i][j] > 0.0 { (dense[i][j] / r.library_size[i] * 1e4).ln_1p() } else { 0.0 }).collect();
        let m = col.iter().sum::<f64>() / 4.0;
        let sd = (col.iter().map(|v| (v - m).powi(2)).sum::<f64>() / 3.0).sqrt();
        assert!((r.hvg_mean[0] - m).abs() < 1e-12 && (r.hvg_sd[0] - sd).abs() < 1e-12);
    }

    #[test]
    fn f32_output_and_vst_statistics_match_the_dense_reference() {
        let mut dense = simulate(300, 120, 11);
        // Outliers so that pass 2 (clipping) runs for a few genes.
        dense[5][100] = 80.0;
        dense[17][101] = 40.0;
        dense[17][7] = 200.0;
        let (ip, ix, dx) = csr(&dense);
        let x = CsrView::new(300, 120, &ip, &ix, &dx).unwrap();
        let params = PreprocessParams { n_hvg: 30, ..Default::default() };
        let r: PreprocessResult<f32> = preprocess(x, &params).unwrap();
        let (mean, var, ve, vs) = reference(&dense, 0.3, (300f64).sqrt());
        for j in 0..120 {
            assert!((r.mean[j] - mean[j]).abs() < 1e-10, "mean {j}");
            assert!((r.variance[j] - var[j]).abs() < 1e-9, "var {j}");
            assert!((r.variance_expected[j] - ve[j]).abs() < 1e-9 * ve[j].max(1.0), "ve {j}");
            assert!((r.variance_standardized[j] - vs[j]).abs() < 1e-9 * vs[j].max(1.0), "vs {j}: {} vs {}", r.variance_standardized[j], vs[j]);
        }
        // Ranking: decreasing standardised variance, ties by index.
        let mut expect: Vec<u32> = (0..120).collect();
        expect.sort_by(|&a, &b| vs[b as usize].partial_cmp(&vs[a as usize]).unwrap().then(a.cmp(&b)));
        expect.truncate(30);
        let mut a = r.hvg.clone();
        let mut b = expect.clone();
        a.sort();
        b.sort();
        assert_eq!(a, b);
        assert!(r.hvg.windows(2).all(|w| r.variance_standardized[w[0] as usize] >= r.variance_standardized[w[1] as usize]));
        // Order agrees wherever the values are not within rounding of each other.
        for (k, (&x, &y)) in r.hvg.iter().zip(&expect).enumerate() {
            if x != y {
                assert!((vs[x as usize] - vs[y as usize]).abs() < 1e-8 * vs[x as usize], "rank {k}: {x} vs {y}");
            }
        }
        // Some gene must have been clipped for this to exercise pass 2.
        assert!((0..120).any(|j| ve[j] > 0.0 && dense.iter().any(|row| (row[j] - mean[j]) / ve[j].sqrt() > (300f64).sqrt())));
        // f32 output agrees with the f64 definition.
        for (k, (i, s)) in ip.windows(2).enumerate().flat_map(|(i, w)| (w[0]..w[1]).map(move |s| (i, s as usize))).enumerate() {
            let lib: f64 = dense[i].iter().sum();
            let y = (dx[s] / lib * 1e4).ln_1p();
            assert!((r.data[k] as f64 - y).abs() < 1e-5, "data {k}");
        }
    }

    #[test]
    fn deterministic_across_thread_counts_and_block_sizes() {
        let dense = simulate(700, 60, 5);
        let (ip, ix, dx) = csr(&dense);
        let x = CsrView::new(700, 60, &ip, &ix, &dx).unwrap();
        let params = PreprocessParams { n_hvg: 10, ..Default::default() };
        let a: PreprocessResult<f64> = preprocess(x, &params).unwrap();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let b: PreprocessResult<f64> = pool.install(|| preprocess(x, &params)).unwrap();
        assert_eq!(a.variance_standardized, b.variance_standardized);
        assert_eq!(a.hvg_sd, b.hvg_sd);
        assert_eq!(a.data, b.data);
        assert_eq!(a.hvg, b.hvg);
    }

    #[test]
    fn constant_genes_are_never_variable_and_errors_are_reported() {
        // Gene 1 is all zero, gene 2 is the same in every cell.
        let dense = vec![vec![1.0, 0.0, 2.0, 4.0], vec![3.0, 0.0, 2.0, 1.0], vec![0.0, 0.0, 2.0, 6.0], vec![2.0, 0.0, 2.0, 0.0]];
        let (ip, ix, dx) = csr(&dense);
        let x = CsrView::new(4, 4, &ip, &ix, &dx).unwrap();
        let r: PreprocessResult<f64> = preprocess(x, &PreprocessParams { n_hvg: 4, ..Default::default() }).unwrap();
        assert_eq!(r.variance[1], 0.0);
        assert_eq!(r.variance[2], 0.0);
        assert_eq!(r.variance_standardized[1], 0.0);
        assert_eq!(r.variance_standardized[2], 0.0);
        assert_eq!(&r.hvg[2..], &[1, 2]);
        assert!(r.variance_standardized[r.hvg[0] as usize] >= r.variance_standardized[r.hvg[1] as usize]);

        let bad = vec![-1.0, 2.0];
        let ip2 = vec![0i32, 1, 2];
        let ix2 = vec![0i32, 0];
        let x = CsrView::new(2, 1, &ip2, &ix2, &bad).unwrap();
        assert!(matches!(preprocess::<_, _, f64>(x, &PreprocessParams::default()), Err(Error::InvalidInput(_))));
        let dup_ip = vec![0i32, 2];
        let dup_ix = vec![0i32, 0];
        let dup_dx = vec![1.0, 1.0];
        let x = CsrView::new(1, 2, &dup_ip, &dup_ix, &dup_dx).unwrap();
        assert!(matches!(preprocess::<_, _, f64>(x, &PreprocessParams::default()), Err(Error::InvalidInput(_))));
        let x = CsrView::new(4, 4, &ip, &ix, &dx).unwrap();
        assert!(matches!(preprocess::<_, _, f64>(x, &PreprocessParams { n_hvg: 0, ..Default::default() }), Err(Error::InvalidParameter(_))));
        assert!(matches!(preprocess::<_, _, f64>(x, &PreprocessParams { scale_factor: 0.0, ..Default::default() }), Err(Error::InvalidParameter(_))));
    }

    #[test]
    fn loess_reproduces_r_on_a_fixed_example() {
        // x: R's set.seed(1); sort(runif(12)). The exact match against R's
        // loess(surface = "direct") is checked by the R package tests (test-preprocess.R);
        // here: properties of the exact local quadratic.
        let x: [f64; 12] = [0.0617863, 0.2016819, 0.2059746, 0.2655087, 0.3721239, 0.5728534, 0.6291140, 0.6870228, 0.8983897, 0.9082078, 0.9446753, 0.9919061];
        let y: Vec<f64> = x.iter().map(|v| (4.0 * v).sin()).collect();
        let f = loess_direct(&x, &y, 0.5).unwrap();
        // Properties that hold for the exact local quadratic of a smooth function: close to y
        // everywhere, and exact for a quadratic.
        for (a, b) in f.iter().zip(&y) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
        let yq: Vec<f64> = x.iter().map(|v| 1.0 + 2.0 * v - 3.0 * v * v).collect();
        let fq = loess_direct(&x, &yq, 0.5).unwrap();
        for (a, b) in fq.iter().zip(&yq) {
            assert!((a - b).abs() < 1e-9);
        }
        // Ties: every point at the same x gives the plain mean.
        let xt = [1.0; 5];
        let yt = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(loess_direct(&xt, &yt, 0.5).unwrap().iter().all(|v| (v - 3.0).abs() < 1e-12));
    }
}
