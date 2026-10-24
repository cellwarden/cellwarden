//! Harmony (Korsunsky et al. 2019, Nature Methods 16:1289), replacing
//! `harmony::RunHarmony()` (R) and `harmonypy.run_harmony()`.
//!
//! Follows the C++ core of the R package as installed in the cellwarden image (harmony 2.0.5,
//! src/harmony.cpp; the 1.2 line differs in the points listed under *Versions*), with the
//! defaults of `RunHarmony.default` and `harmony_options()`.
//!
//! # Algorithm
//!
//! With Z the cells x d input embedding, `phi` the one-hot batch design and K clusters:
//!
//! 1. **Setup.** Cosine-normalise every cell (`Z_cos`). Per batch b: `N_b`, `Pr_b = N_b / N`
//!    and `theta_b = theta (1 - exp(-(N_b / (K tau))^2))` (`theta` itself for `tau = 0`).
//! 2. **Init.** Centroids Y (d x K) from seeded k-means on `Z_cos` (see `kmeans.rs`), or the
//!    caller's `init_centroids`, L2-normalised. `dist = 2 (1 - Y^T Z_cos)`,
//!    `R = softmax_k(-dist / sigma)` per cell, `E = rowSums(R) Pr^T` (K x B), `O = R phi^T`.
//! 3. **Clustering** (each round, `max_iter_cluster` passes). From the second round on,
//!    `Z_cos` is the normalised corrected embedding and R, E, O are recomputed from it as in
//!    step 2; Y is not moved here (harmony 2.x moves it in the correction step). One pass:
//!    shuffle the cells (ChaCha8, seeded) and cut them into `ceil(1 / block_size)` blocks of
//!    `floor(N block_size)` cells (the last takes the rest); per block, remove the block's
//!    cells from E and O, set
//!    `R_kn ∝ exp(-dist_kn / sigma) ((2 E_kb + 1) / (O_kb + E_kb + 1))^theta_b` (b the batch
//!    of n, normalised over k), and add the cells back.
//!    After every pass the objective
//!    `2000 / N (sum R dist + sigma sum R ln R + sigma sum_n theta_b R_kn ln((O + E + 1) / (2E + 1))_kb)`
//!    is recorded; from the fifth pass of a round on, the round stops when the sums of the
//!    last three objectives and of the three before them differ by less than
//!    `epsilon_cluster` (relative).
//! 4. **Correction** (mixture of experts, ridge). For every cluster k, the batches with
//!    `O_kb / N_b > batch_prop_cutoff` are kept (none of the cluster's cells is corrected
//!    when fewer than two are). With `r_b = sum_{n in b} R_kn`, `S_b = sum_{n in b} R_kn Z_n`
//!    and the ridge penalties `lambda_b` (`lambda`, or `alpha E_kb` when `lambda` is `None`;
//!    the intercept is not penalised), the weighted ridge regression of Z on
//!    `[1, phi]` has the closed form (the design covariance is an arrowhead matrix):
//!    `W_0 = (sum_b S_b - sum_b r_b / (r_b + lambda_b) S_b) / (sum_b r_b lambda_b / (r_b + lambda_b))`,
//!    `W_b = (S_b - r_b W_0) / (r_b + lambda_b)`.
//!    `W_0` becomes centroid k (Y is normalised after all clusters), and
//!    `Z_corr = Z - sum_k R_kn W_{k, b(n)}`, always from the original Z.
//! 5. **Convergence.** After each round, stop when the clustering objective fell by less
//!    than `epsilon_harmony` (relative) since the previous round, or after `max_iter`
//!    rounds. The result is the last `Z_corr` (not normalised), as `RunHarmony` returns.
//!
//! # Implementation
//!
//! Everything is f64 (harmony computes in f32). The cells are sorted by batch once, so
//! every per-batch product works on a contiguous range, and laid out d x N (one column per
//! cell) next to R (K x N). Distances are never stored: each block computes them for its
//! cells with one small GEMM (faer) per chunk of 64 cells, and the objective is accumulated
//! in the same pass. Per entry of R that is one `exp` (a polynomial accurate to 1e-15,
//! `fastmath.rs`; the loop is also compiled for AVX2 and picked at run time, with
//! bitwise-identical results). The correction is two GEMMs per chunk of a batch (`S_b`,
//! then `Z -= W_b R`). Memory beyond the input: Z and `Z_corr` (2 N d f64), R (N K f64)
//! and per-block sums.
//!
//! # Determinism
//!
//! The block order comes from ChaCha8 seeded with `seed`; within a block the cells are
//! updated simultaneously (as in harmony), so they are processed in batch order. Cells are
//! split into fixed chunks, every faer call runs with `Par::Seq` inside its task and
//! partial sums are added in chunk order, so the result is bitwise identical for any
//! number of rayon threads (on the same machine and build). harmony itself draws from R's
//! RNG, so its output can only be matched statistically, or closely by passing its own
//! k-means centroids as `init_centroids`.
//!
//! # Versions
//!
//! harmony 1.0-1.2 differ from 2.x in three places: the diversity factor
//! `((E + 1) / (O + 1))^theta` (cross-entropy `ln((O + E) / E)`), centroids re-estimated
//! as `normalise(Z_cos R^T)` before every clustering pass instead of in the correction,
//! and no `batch_prop_cutoff`; `RunHarmony` defaulted to `lambda = 1` there and uses the
//! automatic `lambda = NULL` (alpha E) in 2.x. This module implements 2.x.

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::fastmath::{exp_neg, ln_pos};
use super::kmeans::{clear_upper, kmeans_centers, scores};
use crate::device::Device;
use crate::error::{Error, Result};
use crate::types::{Real, RowMajor};

/// Cells per task in the clustering passes (a block is N / 20 cells at the default
/// block size, so small chunks keep every thread busy).
const CHUNK: usize = 64;
/// Cells per task in the correction step.
const CORRECT_CHUNK: usize = 4096;
/// Width of the moving window of the clustering convergence test (harmony's `window_size`).
const WINDOW: usize = 3;
/// Fewer cells than this: harmony refuses to run.
pub const MIN_CELLS: usize = 6;

/// Memory layout of a cells x dims matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Each cell's d values are contiguous (numpy C order, [`RowMajor`]).
    RowMajor,
    /// Each dimension's N values are contiguous (an R matrix, numpy F order).
    ColMajor,
}

/// Parameters. `Default` is `RunHarmony.default()` with `harmony_options()` of harmony 2.0.5.
#[derive(Clone, Debug)]
pub struct HarmonyParams {
    /// Diversity penalty, one value per batch variable (one variable is supported, so one
    /// value). Default 2; 0 disables the diversity term.
    pub theta: Vec<f64>,
    /// Ridge penalty of the batch coefficients. `None` (default, harmony 2.x `lambda = NULL`)
    /// estimates it per cluster and batch as `alpha * E_kb`; harmony 1.x defaulted to 1.
    pub lambda: Option<f64>,
    /// Width of the soft k-means kernel. Default 0.1.
    pub sigma: f64,
    /// Number of clusters K; `None` is `min(round(N / 30), 100)` (R rounding, half to
    /// even), at least 1.
    pub nclust: Option<usize>,
    /// Maximum rounds of clustering + correction (`max_iter`). Default 10.
    pub max_iter: usize,
    /// Maximum clustering passes per round (`max.iter.cluster`). Default 4.
    pub max_iter_cluster: usize,
    /// Relative tolerance of the clustering objective (`epsilon.cluster`). Default 1e-3;
    /// `-inf` never stops early.
    pub epsilon_cluster: f64,
    /// Relative tolerance of the per-round objective (`epsilon.harmony`). Default 1e-2;
    /// `-inf` runs all `max_iter` rounds (harmony's `early_stop = FALSE`).
    pub epsilon_harmony: f64,
    /// Fraction of the cells updated per block (`block.size`), in (0, 1]. Default 0.05.
    /// Forced to 0.2 below 40 cells, as harmony does.
    pub block_size: f64,
    /// Expected cells per cluster for the small-batch protection of theta (`tau`). Default 0
    /// (off).
    pub tau: f64,
    /// Scale of the automatic lambda (`alpha`). Default 0.2.
    pub alpha: f64,
    /// Batches with less than this fraction of their cells in a cluster are not corrected
    /// in it (`batch.prop.cutoff`). Default 1e-5.
    pub batch_prop_cutoff: f64,
    /// Seeded k-means starts; the one with the lowest within-cluster sum of squares is
    /// kept. Default 1, as harmony 2.x.
    pub kmeans_nstart: usize,
    /// Lloyd steps per start. Default 10, as harmony 2.x.
    pub kmeans_iter: usize,
    /// Seed for the k-means initialisation and the block order.
    pub seed: u64,
    /// K x d starting centroids (any norm; they are L2-normalised) instead of k-means, e.g.
    /// `harmonyObj$Y` (transposed) from R for a like-for-like comparison. K must equal
    /// `nclust` when both are given.
    pub init_centroids: Option<RowMajor>,
}

impl Default for HarmonyParams {
    fn default() -> Self {
        Self {
            theta: vec![2.0],
            lambda: None,
            sigma: 0.1,
            nclust: None,
            max_iter: 10,
            max_iter_cluster: 4,
            epsilon_cluster: 1e-3,
            epsilon_harmony: 1e-2,
            block_size: 0.05,
            tau: 0.0,
            alpha: 0.2,
            batch_prop_cutoff: 1e-5,
            kmeans_nstart: 1,
            kmeans_iter: 10,
            seed: 0,
            init_centroids: None,
        }
    }
}

/// Result of [`harmony`] (`Z = RowMajor`, f32) or [`harmony_dense`] (`Z = Vec<f64>` in the
/// input layout).
#[derive(Clone, Debug)]
pub struct HarmonyResult<Z = RowMajor> {
    /// Corrected embedding, cells x d, in the input cell order.
    pub z_corr: Z,
    /// Objective after the initialisation and after each round's clustering
    /// (`objective_harmony`; length `iters + 1`).
    pub objective: Vec<f64>,
    /// Objective after the initialisation and after every clustering pass
    /// (`objective_kmeans`).
    pub objective_kmeans: Vec<f64>,
    /// Clustering passes run in each round (`kmeans_rounds`).
    pub kmeans_rounds: Vec<usize>,
    /// Rounds run.
    pub iters: usize,
    /// Whether the round objective met `epsilon_harmony` (false: stopped at `max_iter`).
    pub converged: bool,
    /// Number of clusters K used.
    pub n_clusters: usize,
}

impl<Z> HarmonyResult<Z> {
    /// The same result around another corrected embedding (e.g. converted to f32).
    pub fn with_z<Y>(self, z_corr: Y) -> HarmonyResult<Y> {
        HarmonyResult {
            z_corr,
            objective: self.objective,
            objective_kmeans: self.objective_kmeans,
            kmeans_rounds: self.kmeans_rounds,
            iters: self.iters,
            converged: self.converged,
            n_clusters: self.n_clusters,
        }
    }

    /// The corrected embedding and the rest of the result.
    pub fn split_z(self) -> (Z, HarmonyResult<()>) {
        let HarmonyResult { z_corr, objective, objective_kmeans, kmeans_rounds, iters, converged, n_clusters } = self;
        (z_corr, HarmonyResult { z_corr: (), objective, objective_kmeans, kmeans_rounds, iters, converged, n_clusters })
    }
}

/// Harmony on a cells x d f32 embedding (e.g. [`crate::pca::PcaResult::embeddings`]).
///
/// - `batches[i]` is the batch of cell i, below `n_batches`; every batch needs a cell.
/// - Computes in f64 and returns f32; [`harmony_dense`] keeps f64 and takes either layout.
/// - `Device::Cuda` is not implemented yet ([`Error::Gpu`]); callers that resolved
///   `"auto"` should use the CPU.
/// - Errors: `InvalidInput` for shape mismatches, fewer than [`MIN_CELLS`] cells,
///   non-finite values, out-of-range or empty batches; `InvalidParameter` for parameters
///   outside their range.
pub fn harmony(z: &RowMajor, batches: &[u32], n_batches: usize, p: &HarmonyParams, dev: Device) -> Result<HarmonyResult> {
    let r = harmony_dense(&z.data, z.n_rows, z.n_cols, Layout::RowMajor, batches, n_batches, p, dev)?;
    let data = r.z_corr.iter().map(|&v| v as f32).collect();
    let z_corr = RowMajor { n_rows: z.n_rows, n_cols: z.n_cols, data };
    Ok(r.with_z(z_corr))
}

/// [`harmony`] on a borrowed f32 / f64 cells x d matrix in either [`Layout`]; the f64
/// result has the same layout.
#[allow(clippy::too_many_arguments)]
pub fn harmony_dense<T: Real>(
    z: &[T],
    n_cells: usize,
    n_dims: usize,
    layout: Layout,
    batches: &[u32],
    n_batches: usize,
    p: &HarmonyParams,
    dev: Device,
) -> Result<HarmonyResult<Vec<f64>>> {
    if let Device::Cuda(_) = dev {
        return Err(Error::gpu("harmony has no CUDA backend yet; use device = \"cpu\""));
    }
    let k = validate(z, n_cells, n_dims, batches, n_batches, p)?;
    // On a pool worker, so the many small parallel loops below are not each injected into
    // the pool from outside (rayon::scope runs its body on a worker thread).
    rayon::scope(|_| {
        let mut h = State::new(z, n_cells, n_dims, layout, batches, n_batches, k, p)?;
        h.run();
        let out = h.output(layout);
        Ok(h.result.with_z(out))
    })
}

/// Checks inputs and parameters; returns K.
fn validate<T: Real>(z: &[T], n: usize, d: usize, batches: &[u32], n_batches: usize, p: &HarmonyParams) -> Result<usize> {
    let bad = |m: String| Err(Error::InvalidParameter(m));
    if n.checked_mul(d) != Some(z.len()) {
        return Err(Error::InvalidInput(format!("{} values for a {n} x {d} embedding", z.len())));
    }
    if d == 0 {
        return Err(Error::InvalidInput("the embedding has no dimensions".into()));
    }
    if n < MIN_CELLS {
        return Err(Error::InvalidInput(format!("harmony needs at least {MIN_CELLS} cells, got {n}")));
    }
    if batches.len() != n {
        return Err(Error::InvalidInput(format!("{} batch labels for {n} cells", batches.len())));
    }
    if n_batches == 0 {
        return Err(Error::InvalidInput("n_batches must be at least 1".into()));
    }
    if let Some(i) = batches.iter().position(|&b| b as usize >= n_batches) {
        return Err(Error::InvalidInput(format!("cell {i} has batch {} but n_batches = {n_batches}", batches[i])));
    }
    let mut seen = vec![false; n_batches];
    batches.iter().for_each(|&b| seen[b as usize] = true);
    if let Some(b) = seen.iter().position(|&s| !s) {
        return Err(Error::InvalidInput(format!("batch {b} has no cells (drop unused levels first)")));
    }
    if let Some(i) = z.par_iter().position_first(|v| !v.to_f64().is_finite()) {
        return Err(Error::InvalidInput(format!("the embedding has a non-finite value at position {i}")));
    }
    if p.theta.len() != 1 {
        return bad(format!("theta needs one value per batch variable (1 supported), got {}", p.theta.len()));
    }
    if !p.theta.iter().all(|t| t.is_finite() && *t >= 0.0) {
        return bad(format!("theta must be finite and >= 0, got {:?}", p.theta));
    }
    if !(p.sigma.is_finite() && p.sigma > 0.0) {
        return bad(format!("sigma must be finite and > 0, got {}", p.sigma));
    }
    match p.lambda {
        Some(l) if !(l.is_finite() && l > 0.0) => return bad(format!("lambda must be finite and > 0, got {l}")),
        None if !(p.alpha.is_finite() && p.alpha > 0.0) => return bad(format!("alpha must be finite and > 0, got {}", p.alpha)),
        _ => {}
    }
    if p.max_iter == 0 || p.max_iter_cluster == 0 || p.kmeans_nstart == 0 {
        return bad("max_iter, max_iter_cluster and kmeans_nstart must be >= 1".into());
    }
    if !(p.block_size > 0.0 && p.block_size <= 1.0) {
        return bad(format!("block_size must be in (0, 1], got {}", p.block_size));
    }
    if !(p.tau.is_finite() && p.tau >= 0.0) {
        return bad(format!("tau must be finite and >= 0, got {}", p.tau));
    }
    if !(p.batch_prop_cutoff.is_finite() && p.batch_prop_cutoff >= 0.0) {
        return bad(format!("batch_prop_cutoff must be finite and >= 0, got {}", p.batch_prop_cutoff));
    }
    if p.epsilon_cluster.is_nan() || p.epsilon_harmony.is_nan() {
        return bad("epsilon_cluster and epsilon_harmony must not be NaN".into());
    }
    let k = match (&p.init_centroids, p.nclust) {
        (Some(c), Some(k)) if c.n_rows != k => return bad(format!("init_centroids has {} rows but nclust = {k}", c.n_rows)),
        (Some(c), _) => c.n_rows,
        (None, Some(k)) => k,
        (None, None) => ((n as f64 / 30.0).round_ties_even() as usize).clamp(1, 100),
    };
    if k == 0 || k > n {
        return bad(format!("nclust must be between 1 and the number of cells ({n}), got {k}"));
    }
    if let Some(c) = &p.init_centroids {
        if c.n_cols != d || c.data.len() != c.n_rows * d {
            return Err(Error::InvalidInput(format!("init_centroids is {} x {}, expected {k} x {d}", c.n_rows, c.n_cols)));
        }
        if c.data.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidInput("init_centroids has a non-finite value".into()));
        }
    }
    Ok(k)
}

/// One cluster's ridge solution: the intercept W_0 and (batch, W_b) for the corrected batches.
type Solved = (Vec<f64>, Vec<(usize, Vec<f64>)>);

/// Sums of a pass over some cells: per-batch sums of R (`segs`, in batch order) and the
/// k-means error and entropy terms of the objective.
#[derive(Default)]
struct PassSums {
    segs: Vec<(usize, Vec<f64>)>,
    kmeans_error: f64,
    entropy: f64,
}

struct State<'a> {
    n: usize,
    d: usize,
    k: usize,
    nb: usize,
    p: &'a HarmonyParams,
    /// perm[i] = input index of internal cell i (cells sorted by batch, stable).
    perm: Vec<usize>,
    /// Internal cells of batch b: batch_ptr[b]..batch_ptr[b + 1].
    batch_ptr: Vec<usize>,
    batch_of: Vec<u32>,
    n_b: Vec<f64>,
    pr_b: Vec<f64>,
    theta_b: Vec<f64>,
    block_size: f64,
    /// d x N, column per cell.
    z_orig: Vec<f64>,
    z_corr: Vec<f64>,
    /// d x K, column per centroid.
    y: Vec<f64>,
    /// K x N, column per cell.
    r: Vec<f64>,
    /// K x B, column per batch.
    o: Vec<f64>,
    e: Vec<f64>,
    rng: ChaCha8Rng,
    result: HarmonyResult<()>,
}

impl<'a> State<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new<T: Real>(z: &[T], n: usize, d: usize, layout: Layout, batches: &[u32], nb: usize, k: usize, p: &'a HarmonyParams) -> Result<Self> {
        let mut batch_ptr = vec![0usize; nb + 1];
        for &b in batches {
            batch_ptr[b as usize + 1] += 1;
        }
        for b in 0..nb {
            batch_ptr[b + 1] += batch_ptr[b];
        }
        let mut fill = batch_ptr.clone();
        let mut perm = vec![0usize; n];
        for (i, &b) in batches.iter().enumerate() {
            perm[fill[b as usize]] = i;
            fill[b as usize] += 1;
        }
        let batch_of: Vec<u32> = perm.iter().map(|&i| batches[i]).collect();
        let n_b: Vec<f64> = (0..nb).map(|b| (batch_ptr[b + 1] - batch_ptr[b]) as f64).collect();
        let pr_b = n_b.iter().map(|&c| c / n as f64).collect();
        let theta_b = n_b
            .iter()
            .map(|&c| if p.tau == 0.0 { p.theta[0] } else { p.theta[0] * (1.0 - (-(c / (k as f64 * p.tau)).powi(2)).exp()) })
            .collect();

        let mut z_orig = vec![0f64; n * d];
        z_orig.par_chunks_mut(d).zip(perm.par_iter()).for_each(|(col, &src)| match layout {
            Layout::RowMajor => col.iter_mut().zip(&z[src * d..(src + 1) * d]).for_each(|(o, v)| *o = v.to_f64()),
            Layout::ColMajor => col.iter_mut().enumerate().for_each(|(j, o)| *o = z[j * n + src].to_f64()),
        });
        let mut z_corr = z_orig.clone();
        normalise_columns(&mut z_corr, d);

        let y = match &p.init_centroids {
            Some(c) => c.data.iter().map(|&v| v as f64).collect(),
            None => kmeans_centers(&z_corr, d, n, k, p.kmeans_nstart, p.kmeans_iter, p.seed),
        };
        let mut rng = ChaCha8Rng::seed_from_u64(p.seed);
        rng.set_stream(0);
        Ok(Self {
            n,
            d,
            k,
            nb,
            p,
            perm,
            batch_ptr,
            batch_of,
            n_b,
            pr_b,
            theta_b,
            block_size: if n < 40 { 0.2 } else { p.block_size },
            z_orig,
            z_corr,
            y,
            r: vec![0f64; k * n],
            o: vec![0f64; k * nb],
            e: vec![0f64; k * nb],
            rng,
            result: HarmonyResult {
                z_corr: (),
                objective: Vec::new(),
                objective_kmeans: Vec::new(),
                kmeans_rounds: Vec::new(),
                iters: 0,
                converged: false,
                n_clusters: k,
            },
        })
    }

    fn run(&mut self) {
        normalise_columns(&mut self.y, self.d);
        let sums = self.assign();
        let obj = self.objective(&sums);
        self.result.objective_kmeans.push(obj);
        self.result.objective.push(obj);
        for round in 0..self.p.max_iter {
            if round > 0 {
                normalise_columns(&mut self.z_corr, self.d);
                self.assign();
            }
            self.cluster();
            self.correct();
            self.result.iters = round + 1;
            let h = &self.result.objective;
            let (old, new) = (h[h.len() - 2], h[h.len() - 1]);
            if (old - new) / old.abs() < self.p.epsilon_harmony {
                self.result.converged = true;
                break;
            }
        }
    }

    /// Clustering passes of one round (harmony's `cluster_cpp` after its restart).
    fn cluster(&mut self) {
        let mut passes = 0;
        for it in 0..self.p.max_iter_cluster {
            let sums = self.update_r();
            let obj = self.objective(&sums);
            let ok = &mut self.result.objective_kmeans;
            ok.push(obj);
            passes = it + 1;
            if it > WINDOW {
                let m = ok.len();
                let old: f64 = (0..WINDOW).map(|i| ok[m - 2 - i]).sum();
                let new: f64 = (0..WINDOW).map(|i| ok[m - 1 - i]).sum();
                if (old - new).abs() / old.abs() < self.p.epsilon_cluster {
                    break;
                }
            }
        }
        self.result.kmeans_rounds.push(passes);
        let last = *self.result.objective_kmeans.last().unwrap_or(&f64::NAN);
        self.result.objective.push(last);
    }

    /// `R = softmax(-dist / sigma)` for every cell from the current `z_corr` (unit norm) and
    /// Y, then O and E from R. Returns the pass sums for the objective.
    fn assign(&mut self) -> PassSums {
        let (d, k, sigma) = (self.d, self.k, self.p.sigma);
        let (y, batch_of) = (&self.y, &self.batch_of);
        let parts: Vec<PassSums> = self
            .r
            .par_chunks_mut(CHUNK * k)
            .zip(self.z_corr.par_chunks(CHUNK * d))
            .enumerate()
            .map(|(c, (rc, zc))| {
                let cells: Vec<usize> = (c * CHUNK..c * CHUNK + zc.len() / d).collect();
                soft_assign(y, zc, &cells, batch_of, d, k, sigma, None, rc)
            })
            .collect();
        let sums = merge(parts);
        self.o.iter_mut().for_each(|v| *v = 0.0);
        add_segments(&mut self.o, &mut self.e, &sums.segs, &self.pr_b, k, 1.0, true);
        sums
    }

    /// One clustering pass in random blocks (harmony's `update_R`).
    fn update_r(&mut self) -> PassSums {
        let (n, d, k) = (self.n, self.d, self.k);
        let n_blocks = ((1.0 / self.block_size) - 1e-9).ceil().max(1.0) as usize;
        let per_block = (n as f64 * self.block_size).floor() as usize;
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(&mut self.rng);
        // Block i takes positions [i per_block, (i + 1) per_block) of the shuffled order (the
        // last block also the rest). All cells of a block are updated from the same E and O,
        // so their order within the block is free: listing them in internal (batch) order,
        // by a counting sort, makes the per-batch sums contiguous.
        let mut block_of = vec![0u32; n];
        for (pos, &cell) in order.iter().enumerate() {
            block_of[cell] = pos.checked_div(per_block).map_or(n_blocks - 1, |b| b.min(n_blocks - 1)) as u32;
        }
        let mut block_ptr = vec![0usize; n_blocks + 1];
        for &b in &block_of {
            block_ptr[b as usize + 1] += 1;
        }
        for i in 0..n_blocks {
            block_ptr[i + 1] += block_ptr[i];
        }
        let mut fill = block_ptr.clone();
        for (cell, &b) in block_of.iter().enumerate() {
            order[fill[b as usize]] = cell;
            fill[b as usize] += 1;
        }
        let mut total = PassSums::default();
        let mut rsum_b = vec![0f64; k * self.nb];
        for i in 0..n_blocks {
            let cells = &order[block_ptr[i]..block_ptr[i + 1]];
            if cells.is_empty() {
                continue;
            }

            // remove the block's cells from E and O
            let r = &self.r;
            let old: Vec<Vec<(usize, Vec<f64>)>> =
                cells.par_chunks(CHUNK).map(|cc| segments(cc, &self.batch_of, k, |j, _| &r[cc[j] * k..(cc[j] + 1) * k])).collect();
            add_segments(&mut self.o, &mut self.e, old.iter().flatten(), &self.pr_b, k, -1.0, false);

            // log of the diversity factor ((2E + 1) / (O + E + 1))^theta_b, K x B
            let factor: Vec<f64> = (0..k * self.nb)
                .map(|t| {
                    let (e, o) = (self.e[t], self.o[t]);
                    self.theta_b[t / k] * ln_pos((2.0 * e + 1.0) / (o + e + 1.0))
                })
                .collect();

            let (y, z, batch_of, sigma) = (&self.y, &self.z_corr, &self.batch_of, self.p.sigma);
            let r_out = DisjointColumns::new(&mut self.r, k);
            let parts: Vec<PassSums> = cells
                .par_chunks(CHUNK)
                .map(|cc| {
                    let mut zc = vec![0f64; cc.len() * d];
                    for (dst, &j) in zc.chunks_exact_mut(d).zip(cc) {
                        dst.copy_from_slice(&z[j * d..(j + 1) * d]);
                    }
                    let mut rc = vec![0f64; cc.len() * k];
                    let sums = soft_assign(y, &zc, cc, batch_of, d, k, sigma, Some(&factor), &mut rc);
                    for (col, &j) in rc.chunks_exact(k).zip(cc) {
                        // SAFETY: the cells of a block are distinct and every chunk holds
                        // different cells, so no column is written twice or read meanwhile.
                        unsafe { r_out.column(j) }.copy_from_slice(col);
                    }
                    sums
                })
                .collect();
            add_segments(&mut self.o, &mut self.e, parts.iter().flat_map(|p| &p.segs), &self.pr_b, k, 1.0, false);
            for part in parts {
                total.kmeans_error += part.kmeans_error;
                total.entropy += part.entropy;
                for (b, v) in &part.segs {
                    rsum_b[b * k..(b + 1) * k].iter_mut().zip(v).for_each(|(a, x)| *a += x);
                }
            }
        }
        // per-batch sums of the new R over the whole pass, for the cross-entropy term
        total.segs = rsum_b.chunks_exact(k).map(|v| v.to_vec()).enumerate().collect();
        total
    }

    /// harmony's `compute_objective`, from a pass's sums and the current O, E.
    fn objective(&self, s: &PassSums) -> f64 {
        let (k, sigma) = (self.k, self.p.sigma);
        let mut cross = 0.0;
        for (b, rsum) in &s.segs {
            let th = self.theta_b[*b];
            for (c, &rs) in rsum.iter().enumerate() {
                let (o, e) = (self.o[b * k + c], self.e[b * k + c]);
                cross += sigma * th * ln_pos((o + e + 1.0) / (2.0 * e + 1.0)) * rs;
            }
        }
        (s.kmeans_error + sigma * s.entropy + cross) * 2000.0 / self.n as f64
    }

    /// Mixture-of-experts ridge correction (harmony's `moe_correct_ridge_cpp`).
    fn correct(&mut self) {
        let (d, k, nb) = (self.d, self.k, self.nb);
        let ranges: Vec<(usize, usize, usize)> = (0..nb)
            .flat_map(|b| {
                let (lo, hi) = (self.batch_ptr[b], self.batch_ptr[b + 1]);
                (lo..hi).step_by(CORRECT_CHUNK).map(move |s| (b, s, (s + CORRECT_CHUNK).min(hi)))
            })
            .collect();

        // S_b = Z_b R_b^T (d x K) and r_b = rowSums(R_b) (K), per chunk then per batch
        let (z, r) = (&self.z_orig, &self.r);
        let parts: Vec<(Vec<f64>, Vec<f64>)> = ranges
            .par_iter()
            .map(|&(_, lo, hi)| {
                let m = hi - lo;
                let zc = MatRef::from_column_major_slice(&z[lo * d..hi * d], d, m);
                let rc = MatRef::from_column_major_slice(&r[lo * k..hi * k], k, m);
                let mut s = vec![0f64; d * k];
                matmul(MatMut::from_column_major_slice_mut(&mut s, d, k), Accum::Replace, zc, rc.transpose(), 1.0, Par::Seq);
                clear_upper();
                let mut rs = vec![0f64; k];
                for col in r[lo * k..hi * k].chunks_exact(k) {
                    rs.iter_mut().zip(col).for_each(|(a, v)| *a += v);
                }
                (s, rs)
            })
            .collect();
        let mut s_b = vec![0f64; nb * d * k];
        let mut r_b = vec![0f64; nb * k];
        for (&(b, _, _), (s, rs)) in ranges.iter().zip(&parts) {
            s_b[b * d * k..(b + 1) * d * k].iter_mut().zip(s).for_each(|(a, v)| *a += v);
            r_b[b * k..(b + 1) * k].iter_mut().zip(rs).for_each(|(a, v)| *a += v);
        }

        // per cluster: the arrowhead ridge solve -> intercept W_0 and batch terms W_b
        let solved: Vec<Option<Solved>> = (0..k)
            .into_par_iter()
            .map(|c| {
                let keep: Vec<usize> = (0..nb).filter(|&b| self.o[b * k + c] / self.n_b[b] > self.p.batch_prop_cutoff).collect();
                if keep.len() < 2 {
                    return None;
                }
                let s = |b: usize| &s_b[b * d * k + c * d..b * d * k + (c + 1) * d];
                let mut w0 = vec![0f64; d];
                let mut u = 0.0;
                let mut lam = Vec::with_capacity(keep.len());
                for &b in &keep {
                    let rb = r_b[b * k + c];
                    let l = self.p.lambda.unwrap_or(self.p.alpha * self.e[b * k + c]);
                    lam.push(l);
                    let w = l / (rb + l); // 1 - r_b / (r_b + lambda_b)
                    u += rb * w;
                    w0.iter_mut().zip(s(b)).for_each(|(a, v)| *a += w * v);
                }
                if u.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                    return None;
                }
                w0.iter_mut().for_each(|v| *v /= u);
                let wb = keep
                    .iter()
                    .zip(&lam)
                    .map(|(&b, &l)| {
                        let rb = r_b[b * k + c];
                        let col = s(b).iter().zip(&w0).map(|(sv, w)| (sv - rb * w) / (rb + l)).collect();
                        (b, col)
                    })
                    .collect();
                Some((w0, wb))
            })
            .collect();

        // W per batch as d x K (zero where a cluster does not correct the batch); new Y
        let mut w = vec![0f64; nb * d * k];
        for (c, sol) in solved.into_iter().enumerate() {
            if let Some((w0, wb)) = sol {
                self.y[c * d..(c + 1) * d].copy_from_slice(&w0);
                for (b, col) in wb {
                    w[b * d * k + c * d..b * d * k + (c + 1) * d].copy_from_slice(&col);
                }
            }
        }
        normalise_columns(&mut self.y, d);

        // Z_corr = Z - W_b R_b, chunk by chunk
        let mut slices = Vec::with_capacity(ranges.len());
        let mut rest = &mut self.z_corr[..];
        for &(b, lo, hi) in &ranges {
            let (head, tail) = rest.split_at_mut((hi - lo) * d);
            slices.push((b, lo, hi, head));
            rest = tail;
        }
        let (z, r) = (&self.z_orig, &self.r);
        slices.into_par_iter().for_each(|(b, lo, hi, out)| {
            out.copy_from_slice(&z[lo * d..hi * d]);
            let m = hi - lo;
            let wb = MatRef::from_column_major_slice(&w[b * d * k..(b + 1) * d * k], d, k);
            let rc = MatRef::from_column_major_slice(&r[lo * k..hi * k], k, m);
            matmul(MatMut::from_column_major_slice_mut(out, d, m), Accum::Add, wb, rc, -1.0, Par::Seq);
            clear_upper();
        });
    }

    /// `z_corr` in the input order and `layout`.
    fn output(&self, layout: Layout) -> Vec<f64> {
        let (n, d) = (self.n, self.d);
        let mut out = vec![0f64; n * d];
        match layout {
            Layout::RowMajor => {
                let mut inv = vec![0usize; n];
                self.perm.iter().enumerate().for_each(|(i, &src)| inv[src] = i);
                out.par_chunks_mut(d).zip(inv.par_iter()).for_each(|(row, &i)| row.copy_from_slice(&self.z_corr[i * d..(i + 1) * d]));
            }
            Layout::ColMajor => out.par_chunks_mut(n).enumerate().for_each(|(j, col)| {
                for (i, &src) in self.perm.iter().enumerate() {
                    col[src] = self.z_corr[i * d + j];
                }
            }),
        }
        out
    }
}

/// New R for the cells `cells` (internal indices, batch order) whose unit-norm columns are
/// `zc` (d x m): `R_kn ∝ exp(-dist_kn / sigma + log_factor_kb)`, written to `rc` (K x m),
/// with the pass sums.
///
/// Per cell, with the scores `s_k = y_k . z`, `dist_k = 2 (1 - s_k)`,
/// `l_k = -dist_k / sigma + log_factor_k - max` and `T = sum_k exp(l_k)`: one `exp` per
/// entry and one `ln` per cell, since `sum_k R_k dist_k = 2 (1 - sum_k R_k s_k)` and
/// `sum_k R_k ln R_k = sum_k R_k l_k - ln T`. The sums use four fixed lanes (same order on
/// every thread) so they do not serialise on one accumulator.
#[allow(clippy::too_many_arguments)]
fn soft_assign(y: &[f64], zc: &[f64], cells: &[usize], batch_of: &[u32], d: usize, k: usize, sigma: f64, log_factor: Option<&[f64]>, rc: &mut [f64]) -> PassSums {
    scores(y, zc, d, k, rc);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is available (checked above).
        return unsafe { soft_assign_avx2(cells, batch_of, k, sigma, log_factor, rc) };
    }
    soft_assign_rows(cells, batch_of, k, sigma, log_factor, rc)
}

/// [`soft_assign_rows`] compiled for AVX2 (4-wide `exp`). The same operations in the same
/// order, so the results are bitwise those of the baseline build (no FMA contraction).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn soft_assign_avx2(cells: &[usize], batch_of: &[u32], k: usize, sigma: f64, log_factor: Option<&[f64]>, rc: &mut [f64]) -> PassSums {
    soft_assign_rows(cells, batch_of, k, sigma, log_factor, rc)
}

/// The per-cell part of [`soft_assign`], on the scores already in `rc`.
#[inline(always)]
fn soft_assign_rows(cells: &[usize], batch_of: &[u32], k: usize, sigma: f64, log_factor: Option<&[f64]>, rc: &mut [f64]) -> PassSums {
    let inv_sigma = 1.0 / sigma;
    let zero = vec![0f64; k];
    let mut kmeans_error = 0.0;
    let mut entropy = 0.0;
    let mut logit = vec![0f64; k];
    let mut v = vec![0f64; k];
    for (j, col) in rc.chunks_exact_mut(k).enumerate() {
        let b = batch_of[cells[j]] as usize;
        let lf = log_factor.map_or(&zero[..], |f| &f[b * k..(b + 1) * k]);
        for ((lv, &s), &f) in logit.iter_mut().zip(col.iter()).zip(lf) {
            *lv = 2.0 * (s - 1.0) * inv_sigma + f;
        }
        let max = lanes(&logit, f64::NEG_INFINITY, |m, x| m.max(x), |a, b| a.max(b));
        for (vv, lv) in v.iter_mut().zip(logit.iter_mut()) {
            *lv -= max;
            *vv = exp_neg(*lv);
        }
        let total = lanes(&v, 0.0, |a, x| a + x, |a, b| a + b);
        let inv = 1.0 / total;
        let vs = dot_lanes(&v, col);
        let vl = dot_lanes(&v, &logit);
        for (c, &vv) in col.iter_mut().zip(&v) {
            *c = vv * inv;
        }
        kmeans_error += 2.0 * (1.0 - vs * inv);
        entropy += vl * inv - ln_pos(total);
    }
    let segs = segments(cells, batch_of, k, |j, _| &rc[j * k..(j + 1) * k]);
    PassSums { segs, kmeans_error, entropy }
}

/// Fold of `x` over four interleaved lanes, combined as ((l0 . l1) . (l2 . l3)).
#[inline(always)]
fn lanes(x: &[f64], init: f64, step: impl Fn(f64, f64) -> f64, comb: impl Fn(f64, f64) -> f64) -> f64 {
    let mut acc = [init; 4];
    let (quads, rest) = x.as_chunks::<4>();
    for q in quads {
        for l in 0..4 {
            acc[l] = step(acc[l], q[l]);
        }
    }
    for (l, &r) in rest.iter().enumerate() {
        acc[l] = step(acc[l], r);
    }
    comb(comb(acc[0], acc[1]), comb(acc[2], acc[3]))
}

/// `sum_i a_i b_i` over four interleaved lanes.
#[inline(always)]
fn dot_lanes(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0f64; 4];
    let ((qa, ra), (qb, rb)) = (a.as_chunks::<4>(), b.as_chunks::<4>());
    for (x, y) in qa.iter().zip(qb) {
        for l in 0..4 {
            acc[l] += x[l] * y[l];
        }
    }
    for (l, (x, y)) in ra.iter().zip(rb).enumerate() {
        acc[l] += x * y;
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// Per-batch sums of the K-vectors `col(j)` of the cells `cells` (batch order), as
/// (batch, sum) runs.
fn segments<'s>(cells: &[usize], batch_of: &[u32], k: usize, col: impl Fn(usize, usize) -> &'s [f64]) -> Vec<(usize, Vec<f64>)> {
    let mut out: Vec<(usize, Vec<f64>)> = Vec::new();
    for (j, &cell) in cells.iter().enumerate() {
        let b = batch_of[cell] as usize;
        if out.last().is_none_or(|(lb, _)| *lb != b) {
            out.push((b, vec![0f64; k]));
        }
        let acc = &mut out.last_mut().map(|s| &mut s.1).expect("segment pushed above")[..];
        acc.iter_mut().zip(col(j, cell)).for_each(|(a, v)| *a += v);
    }
    out
}

/// Chunk sums in chunk order; runs of the same batch at chunk boundaries are merged.
fn merge(parts: Vec<PassSums>) -> PassSums {
    let mut total = PassSums::default();
    for part in parts {
        total.kmeans_error += part.kmeans_error;
        total.entropy += part.entropy;
        for (b, v) in part.segs {
            match total.segs.last_mut() {
                Some((lb, lv)) if *lb == b => lv.iter_mut().zip(&v).for_each(|(a, x)| *a += x),
                _ => total.segs.push((b, v)),
            }
        }
    }
    total
}

/// `O[:, b] += sign * sum`, `E[:, b'] += sign * rowsum Pr_b'` for every run; with `reset_e`
/// E is first set to zero (fresh assignment).
fn add_segments<'s>(o: &mut [f64], e: &mut [f64], segs: impl IntoIterator<Item = &'s (usize, Vec<f64>)>, pr_b: &[f64], k: usize, sign: f64, reset_e: bool) {
    if reset_e {
        e.iter_mut().for_each(|v| *v = 0.0);
    }
    let mut rsum = vec![0f64; k];
    for (b, v) in segs {
        o[b * k..(b + 1) * k].iter_mut().zip(v).for_each(|(a, x)| *a += sign * x);
        rsum.iter_mut().zip(v).for_each(|(a, x)| *a += x);
    }
    for (b, &pr) in pr_b.iter().enumerate() {
        e[b * k..(b + 1) * k].iter_mut().zip(&rsum).for_each(|(a, x)| *a += sign * x * pr);
    }
}

/// Mutable access to distinct length-`k` columns of a buffer from several tasks at once
/// (the block update writes the R columns of randomly placed cells).
struct DisjointColumns {
    ptr: *mut f64,
    k: usize,
    n: usize,
}

// SAFETY: only `column` hands out access, and its contract makes the accesses disjoint.
unsafe impl Send for DisjointColumns {}
unsafe impl Sync for DisjointColumns {}

impl DisjointColumns {
    fn new(buf: &mut [f64], k: usize) -> Self {
        Self { ptr: buf.as_mut_ptr(), k, n: buf.len() / k }
    }

    /// Column `j`.
    ///
    /// # Safety
    /// No two live references may be to the same column, and the buffer must outlive them
    /// and not be accessed otherwise meanwhile.
    #[allow(clippy::mut_from_ref)]
    unsafe fn column(&self, j: usize) -> &mut [f64] {
        assert!(j < self.n, "column {j} out of range");
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(j * self.k), self.k) }
    }
}

/// L2-normalises every length-`d` column in place (zero columns stay zero).
fn normalise_columns(x: &mut [f64], d: usize) {
    x.par_chunks_mut(d * 256).for_each(|block| {
        for col in block.chunks_exact_mut(d) {
            let norm = col.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                col.iter_mut().for_each(|v| *v /= norm);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// `types` cell types x `nb` batches of Gaussian blobs in `d` dims; every batch adds its
    /// own shift. Returns (cells x d, batch, type).
    fn blobs(n_per: usize, types: usize, nb: usize, d: usize, seed: u64) -> (RowMajor, Vec<u32>, Vec<usize>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut gauss = move || {
            let (u1, u2): (f64, f64) = (1.0 - rng.random::<f64>(), rng.random::<f64>());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        let centres: Vec<Vec<f64>> = (0..types).map(|_| (0..d).map(|_| 10.0 * gauss()).collect()).collect();
        let shifts: Vec<Vec<f64>> = (0..nb).map(|_| (0..d).map(|_| 4.0 * gauss()).collect()).collect();
        let mut data = Vec::new();
        let (mut batch, mut ty) = (Vec::new(), Vec::new());
        for i in 0..n_per * types * nb {
            // interleaved, so the batches are not contiguous in the input
            let (t, b) = (i % types, (i / types) % nb);
            for j in 0..d {
                data.push((centres[t][j] + shifts[b][j] + gauss()) as f32);
            }
            batch.push(b as u32);
            ty.push(t);
        }
        let n = batch.len();
        (RowMajor { n_rows: n, n_cols: d, data }, batch, ty)
    }

    fn centroid(z: &RowMajor, sel: impl Fn(usize) -> bool) -> Vec<f64> {
        let mut c = vec![0f64; z.n_cols];
        let mut m = 0usize;
        for i in (0..z.n_rows).filter(|&i| sel(i)) {
            c.iter_mut().zip(z.row(i)).for_each(|(a, &v)| *a += v as f64);
            m += 1;
        }
        c.iter_mut().for_each(|v| *v /= m as f64);
        c
    }

    fn dist(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
    }

    /// Mean distance between batch centroids within a type, and the smallest distance
    /// between type centroids, of an embedding.
    fn mixing(z: &RowMajor, batch: &[u32], ty: &[usize], types: usize, nb: usize) -> (f64, f64) {
        let mut within = Vec::new();
        for t in 0..types {
            let cs: Vec<Vec<f64>> = (0..nb).map(|b| centroid(z, |i| ty[i] == t && batch[i] as usize == b)).collect();
            for a in 0..nb {
                for b in a + 1..nb {
                    within.push(dist(&cs[a], &cs[b]));
                }
            }
        }
        let tc: Vec<Vec<f64>> = (0..types).map(|t| centroid(z, |i| ty[i] == t)).collect();
        let mut between = f64::INFINITY;
        for a in 0..types {
            for b in a + 1..types {
                between = between.min(dist(&tc[a], &tc[b]));
            }
        }
        (within.iter().sum::<f64>() / within.len() as f64, between)
    }

    #[test]
    fn removes_batch_shift_and_keeps_types() {
        let (types, nb) = (3, 3);
        let (z, batch, ty) = blobs(200, types, nb, 10, 1);
        let (w0, b0) = mixing(&z, &batch, &ty, types, nb);
        let res = harmony(&z, &batch, nb, &HarmonyParams::default(), Device::Cpu).unwrap();
        assert_eq!((res.z_corr.n_rows, res.z_corr.n_cols), (z.n_rows, z.n_cols));
        let (w1, b1) = mixing(&res.z_corr, &batch, &ty, types, nb);
        assert!(w1 < 0.2 * w0, "batch centroid distance within types {w0:.3} -> {w1:.3}");
        assert!(b1 > 5.0 * w1 && b1 > 0.5 * b0, "types merged: between {b0:.3} -> {b1:.3}, within {w1:.3}");
        assert!(res.iters >= 1 && res.iters <= 10);
        assert_eq!(res.objective.len(), res.iters + 1);
        assert_eq!(res.kmeans_rounds.len(), res.iters);
        assert_eq!(res.objective_kmeans.len(), 1 + res.kmeans_rounds.iter().sum::<usize>());
        assert_eq!(res.n_clusters, 60); // round(1800 / 30)
    }

    #[test]
    fn objective_does_not_increase() {
        let (z, batch, _) = blobs(150, 3, 3, 8, 2);
        let p = HarmonyParams { epsilon_harmony: f64::NEG_INFINITY, lambda: Some(1.0), ..Default::default() };
        let res = harmony(&z, &batch, 3, &p, Device::Cpu).unwrap();
        assert_eq!(res.iters, 10);
        assert!(!res.converged);
        let obj = &res.objective;
        for w in obj.windows(2) {
            assert!(w[1] <= w[0] + 1e-3 * w[0].abs(), "round objective rose: {obj:?}");
        }
        assert!(obj[obj.len() - 1] < obj[0], "{obj:?}");
    }

    #[test]
    fn identical_for_any_thread_count_and_layout() {
        let (z, batch, _) = blobs(300, 3, 4, 12, 3);
        let p = HarmonyParams { seed: 5, ..Default::default() };
        let many = harmony(&z, &batch, 4, &p, Device::Cpu).unwrap();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let one = pool.install(|| harmony(&z, &batch, 4, &p, Device::Cpu)).unwrap();
        assert!(many.z_corr == one.z_corr, "harmony differs between thread counts");
        assert_eq!(many.objective, one.objective);
        let other = harmony(&z, &batch, 4, &HarmonyParams { seed: 6, ..p.clone() }, Device::Cpu).unwrap();
        assert!(other.z_corr != many.z_corr, "the seed must matter");

        // column-major f64 input gives the same numbers, column-major
        let (n, d) = (z.n_rows, z.n_cols);
        let cm: Vec<f64> = (0..n * d).map(|t| z.data[(t % n) * d + t / n] as f64).collect();
        let r = harmony_dense(&cm, n, d, Layout::ColMajor, &batch, 4, &p, Device::Cpu).unwrap();
        for i in 0..n {
            for j in 0..d {
                assert_eq!(r.z_corr[j * n + i] as f32, many.z_corr.data[i * d + j]);
            }
        }
    }

    #[test]
    fn injected_centroids_replace_kmeans() {
        let (z, batch, _) = blobs(100, 3, 2, 6, 4);
        let a = harmony(&z, &batch, 2, &HarmonyParams { nclust: Some(5), ..Default::default() }, Device::Cpu).unwrap();
        let init = RowMajor::from_vec(5, 6, z.data[..30].to_vec()).unwrap();
        let p = HarmonyParams { init_centroids: Some(init), ..Default::default() };
        let b = harmony(&z, &batch, 2, &p, Device::Cpu).unwrap();
        assert_eq!(b.n_clusters, 5);
        assert!(a.z_corr != b.z_corr);
        let c = harmony(&z, &batch, 2, &HarmonyParams { seed: 9, ..p }, Device::Cpu).unwrap();
        assert_eq!(c.objective[0], b.objective[0], "with given centroids the start does not depend on the seed");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_soft_assignment_is_bitwise_the_baseline() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let (k, m) = (37, 50);
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let scores: Vec<f64> = (0..k * m).map(|_| 2.0 * rng.random::<f64>() - 1.0).collect();
        let log_factor: Vec<f64> = (0..k * 2).map(|_| -rng.random::<f64>()).collect();
        let cells: Vec<usize> = (0..m).collect();
        let batch_of: Vec<u32> = (0..m as u32).map(|i| i % 2).collect();
        let (mut a, mut b) = (scores.clone(), scores);
        let sa = soft_assign_rows(&cells, &batch_of, k, 0.1, Some(&log_factor), &mut a);
        // SAFETY: AVX2 checked above.
        let sb = unsafe { soft_assign_avx2(&cells, &batch_of, k, 0.1, Some(&log_factor), &mut b) };
        assert!(a == b);
        assert_eq!((sa.kmeans_error, sa.entropy), (sb.kmeans_error, sb.entropy));
    }

    #[test]
    fn single_batch_is_left_unchanged() {
        let (z, _, _) = blobs(50, 3, 1, 5, 5);
        let batch = vec![0u32; z.n_rows];
        let res = harmony(&z, &batch, 1, &HarmonyParams::default(), Device::Cpu).unwrap();
        assert!(res.z_corr == z);
    }

    #[test]
    fn rejects_bad_input() {
        let (z, batch, _) = blobs(20, 3, 2, 4, 6);
        let p = HarmonyParams::default();
        let input = |r: Result<HarmonyResult>| matches!(r, Err(Error::InvalidInput(_)));
        let param = |r: Result<HarmonyResult>| matches!(r, Err(Error::InvalidParameter(_)));
        assert!(input(harmony(&z, &batch[1..], 2, &p, Device::Cpu)));
        assert!(input(harmony(&z, &batch, 1, &p, Device::Cpu)), "batch label out of range");
        assert!(input(harmony(&z, &batch, 3, &p, Device::Cpu)), "empty batch");
        let small = RowMajor::from_vec(5, 4, z.data[..20].to_vec()).unwrap();
        assert!(input(harmony(&small, &batch[..5], 2, &p, Device::Cpu)), "fewer than 6 cells");
        let mut nan = z.clone();
        nan.data[7] = f32::NAN;
        assert!(input(harmony(&nan, &batch, 2, &p, Device::Cpu)));
        for bad in [
            HarmonyParams { theta: vec![2.0, 2.0], ..p.clone() },
            HarmonyParams { theta: vec![-1.0], ..p.clone() },
            HarmonyParams { sigma: 0.0, ..p.clone() },
            HarmonyParams { lambda: Some(0.0), ..p.clone() },
            HarmonyParams { alpha: 0.0, ..p.clone() },
            HarmonyParams { nclust: Some(0), ..p.clone() },
            HarmonyParams { nclust: Some(10_000), ..p.clone() },
            HarmonyParams { max_iter: 0, ..p.clone() },
            HarmonyParams { block_size: 0.0, ..p.clone() },
            HarmonyParams { block_size: 1.5, ..p.clone() },
            HarmonyParams { tau: f64::NAN, ..p.clone() },
            HarmonyParams { init_centroids: Some(RowMajor::zeros(3, 4)), nclust: Some(4), ..p.clone() },
        ] {
            assert!(param(harmony(&z, &batch, 2, &bad, Device::Cpu)), "{bad:?}");
        }
        let wrong_dims = HarmonyParams { init_centroids: Some(RowMajor::zeros(3, 5)), ..p.clone() };
        assert!(input(harmony(&z, &batch, 2, &wrong_dims, Device::Cpu)));
        assert!(matches!(harmony(&z, &batch, 2, &p, Device::Cuda(0)), Err(Error::Gpu(_))));
    }
}
