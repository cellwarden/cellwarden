//! Implicitly scaled PCA: replaces Seurat's `ScaleData` + `RunPCA` without ever building
//! the dense scaled matrix (docs/RUST_ENGINE.md sections 1 and 8).
//!
//! With X the log-normalised cells x genes matrix (HVGs only), per-gene mean `mu` and
//! standard deviation `sigma`, the scaled matrix is `Z = (X - 1 mu^T) diag(1/sigma)`.
//! The SVD only needs `Z v` and `Z^T u`, which are sparse products plus a rank-one
//! correction (and a small sparse correction for Seurat's clip), so memory stays at the
//! size of the sparse X.
//!
//! # Algorithm
//!
//! 1. One parallel pass over the CSR for per-gene mean and standard deviation (n - 1
//!    denominator, as `ScaleData`); constant genes get sd 1.
//! 2. Block Krylov SVD (Musco & Musco 2015) on the implicit operator: Gaussian test matrix
//!    (genes x b, b = n_comps + oversample) from ChaCha8 seeded with `seed`; `power_iters`
//!    subspace iterations, re-orthonormalising after every product and keeping every
//!    cells x b block, so the basis K spans the Krylov space (width b x (power_iters + 1));
//!    each block is orthonormalised against the earlier ones as it is made, so the
//!    `Z^T K` of the final thin SVD is the products the iteration already computed.
//! 3. Embeddings `K V_b diag(d)`, loadings `U_b`, all in f64, returned as f32.
//!
//! # Memory
//!
//! Besides the borrowed X: the cells x width f64 basis K (2.2 KB per cell at the defaults,
//! b = 40 and width 280, against 16 KB per cell for `ScaleData`'s dense 2,000-HVG matrix),
//! a few genes x width matrices, per-task chunk buffers, and the `Z^T U` accumulators (see
//! `operator.rs`). No transposed copy of X.
//!
//! # Determinism
//!
//! Every reduction uses a partition fixed by the input shape and a fixed summation tree,
//! and every faer call runs with `Par::Seq` inside its task, so a given seed and input give
//! bitwise-identical output for any number of rayon threads (on the same machine and build).

use faer::reborrow::{Reborrow, ReborrowMut};

use crate::device::Device;
use crate::error::{Error, Result};
use crate::types::{CsrView, Idx, Real, RowMajor};

#[cfg(feature = "cuda")]
mod cuda;
mod dense;
mod operator;

use operator::{Blocks, Clipped, ScaledOp};

/// The implicit operator `Z` as the SVD sees it: `Z v` and `Z^T u` on
/// column-major f64 faer matrices. Implemented by the CPU [`ScaledOp`] and, with feature
/// `cuda`, by [`cuda::CudaScaledOp`].
pub(crate) trait Operator {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    /// `out = Z v`, `v` genes x l, `out` cells x l.
    fn apply(&self, v: faer::MatRef<'_, f64>, out: faer::MatMut<'_, f64>) -> Result<()>;
    /// `out = (Z - C) v` for the sparse clip correction `C` of [`Clipped`]. The default
    /// subtracts `C v` after `apply`; the CPU operator does it inside its row tasks.
    fn apply_clipped(&self, v: faer::MatRef<'_, f64>, mut out: faer::MatMut<'_, f64>, clip: &operator::ClipEntries) -> Result<()> {
        self.apply(v, out.rb_mut())?;
        clip.subtract_from(v, out);
        Ok(())
    }
    /// `Z^T u`, `u` cells x l; returns genes x l.
    fn apply_t(&self, u: faer::MatRef<'_, f64>) -> Result<faer::Mat<f64>>;
}

/// Parameters, with Seurat-compatible defaults.
#[derive(Clone, Debug)]
pub struct PcaParams {
    /// Number of components (Seurat `npcs`; pipeline default 30).
    pub n_comps: usize,
    /// Extra random directions per Krylov block (block width `n_comps + oversample`).
    /// Default 10 (see `Default`).
    pub oversample: usize,
    /// Power iterations, i.e. Krylov blocks beyond the first. Default 6.
    pub power_iters: usize,
    /// RNG seed (Seurat `seed.use`; pipeline default 0). Same seed, same result.
    pub seed: u64,
    /// Divide genes by their standard deviation (Seurat `do.scale = TRUE`).
    pub scale: bool,
    /// Seurat's `scale.max` clip (default 10), applied exactly as a sparse correction to the
    /// implicit operator; `None` or `inf` leaves the scaled values unclipped.
    pub clip: Option<f64>,
}

/// Defaults chosen on PBMC 3k against Seurat 5.6 (Seurat's HVGs, clipped `RunPCA`, kNN
/// overlap on the top 10 PCs, k = 20): block Krylov with `oversample = 10, power_iters = 6`
/// gives 0.977 (exact SVD: 0.9975) in 1.1 s at 100k cells on 12 threads, where the plain
/// randomised SVD needed `oversample = 60, power_iters = 30` (6.4 s) for the same accuracy
/// and gave 0.905 at `30 / 10` (1.4 s). PCs 9-11 there have nearly equal singular values,
/// which is the case Krylov methods converge on fastest.
impl Default for PcaParams {
    fn default() -> Self {
        Self { n_comps: 30, oversample: 10, power_iters: 6, seed: 0, scale: true, clip: Some(10.0) }
    }
}

/// Output in Seurat's convention (`RunPCA` via irlba on the scaled matrix).
#[derive(Clone, Debug)]
pub struct PcaResult {
    /// Cell embeddings, cells x n_comps: U * diag(d) (Seurat `Embeddings`, Scanpy `X_pca`).
    pub embeddings: RowMajor,
    /// Gene loadings, genes x n_comps: V (Seurat `Loadings`, Scanpy `varm["PCs"]`).
    pub loadings: RowMajor,
    /// Standard deviations of the components: d / sqrt(n_cells - 1) (Seurat `Stdev`).
    pub stdev: Vec<f32>,
    /// Per-gene means and standard deviations used for the implicit scaling.
    pub gene_mean: Vec<f32>,
    pub gene_sd: Vec<f32>,
}

/// PCA of the implicitly centred (and, if `params.scale`, scaled) matrix `x`.
///
/// - `x`: cells x genes, log-normalised, typically restricted to the HVGs.
/// - Genes with zero variance get sd 1 (they contribute nothing), as in Seurat.
/// - Components are ordered by decreasing singular value; the sign of each component is
///   fixed deterministically (largest-magnitude loading positive) so results are stable.
/// - Parallel over rows/columns with rayon; deterministic for a fixed seed regardless of
///   the number of threads.
///
/// Details:
/// - `gene_sd` is the sd actually used: the sample sd (n - 1 denominator) for scaled genes,
///   1 for constant genes, and 1 for every gene when `params.scale` is false.
/// - The Krylov width `(n_comps + oversample) x (power_iters + 1)` is capped at `min(n_cells, n_genes)`.
/// - `params.clip` must be positive (`inf` allowed); finite values clip scaled values above it.
/// - Errors: `InvalidParameter` for `n_comps == 0`, `n_comps >= min(n_cells, n_genes)` or a
///   non-positive / NaN `clip`; `InvalidInput` for malformed CSR structure (including
///   duplicate column indices within a row) or non-finite values.
pub fn implicit_scaled_pca<I: Idx, T: Real>(x: CsrView<'_, I, T>, params: &PcaParams) -> Result<PcaResult> {
    implicit_scaled_pca_on(x, params, Device::Cpu)
}

/// [`implicit_scaled_pca`] on a chosen [`Device`] (docs/scfast-integration.md Phase 6).
///
/// `Device::Cpu` is exactly [`implicit_scaled_pca`]. `Device::Cuda(i)` computes the gene
/// statistics on the CPU, uploads the CSR (f32 values) and runs `gram_svd` there: the
/// Gram matrix Z^T Z and, up to 4096 genes, its exact eigendecomposition (cuSOLVER) on the
/// device, then one product for the cell scores; the clip is applied exactly (see
/// `cuda_pca`). It needs the `cuda` feature, a visible device and enough device memory for
/// the CSR, the Gram matrix and a few cells x b buffers, otherwise it returns [`Error::Gpu`]
/// without touching the CPU path (callers that asked for `"auto"` can retry on the CPU).
/// The GPU result is not bitwise equal to the CPU one (f32 products, atomics, exact rather
/// than Krylov eigenvectors); the parity target is a top-10 PC subspace similarity above 0.99.
pub fn implicit_scaled_pca_on<I: Idx, T: Real>(x: CsrView<'_, I, T>, params: &PcaParams, device: Device) -> Result<PcaResult> {
    validate(&x, params)?;
    let (n, p) = (x.n_rows, x.n_cols);
    let blocks = Blocks::new(n, p, x.data.len());
    let stats = operator::gene_stats(&x, blocks)?;
    match device {
        Device::Cpu => {
            let (gene_sd, inv_sd) = scaling(&stats, params);
            let b = block_width(n, p, params);
            let op = ScaledOp::new(x, &stats.mean, &inv_sd, blocks);
            let svd = match params.clip.filter(|c| c.is_finite()) {
                Some(c) => block_krylov_svd(&Clipped::new(op, &x, &stats.mean, &inv_sd, c), b, params.power_iters, params.seed)?,
                None => block_krylov_svd(&op, b, params.power_iters, params.seed)?,
            };
            Ok(assemble(svd, &stats, gene_sd, n, p, params.n_comps))
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(ordinal) => cuda_pca(cuda::CsrDev::upload(&x, ordinal)?, stats, params),
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => Err(Error::gpu("scfast was built without CUDA support (feature `cuda`); use Device::Cpu")),
    }
}

/// [`implicit_scaled_pca_on`] for a CSR matrix that is already on CUDA device `ordinal`
/// (e.g. a CuPy matrix): the data are copied device-to-device and the gene statistics are
/// computed there, so nothing of the matrix crosses to the host.
///
/// # Safety
/// The pointers in `x` must be valid device memory as described by [`DeviceCsr`], with all
/// work writing them finished. The structure is not validated on the device.
#[cfg(feature = "cuda")]
pub unsafe fn implicit_scaled_pca_device(x: &DeviceCsr, params: &PcaParams, ordinal: usize) -> Result<PcaResult> {
    validate_params(x.n, x.p, params)?;
    let csr = unsafe { cuda::CsrDev::copy_from(x, ordinal)? };
    let stats = csr.stats()?.finish(x.n)?;
    cuda_pca(csr, stats, params)
}

/// The device PCA on the device copy `csr` of X. Seurat's clip is exact on the device:
/// a scaled value `(x - m) / sd` exceeds `clip` only at a stored entry, and there it equals
/// the clip exactly when `x` is replaced by `m + clip sd`, so the stored values are capped
/// in place and the operator keeps the unclipped means. The Gram correction then needs the
/// column sums of the capped X (a second statistics pass).
#[cfg(feature = "cuda")]
fn cuda_pca(mut csr: cuda::CsrDev, stats: operator::GeneStats, params: &PcaParams) -> Result<PcaResult> {
    let (n, p) = (csr.n, csr.p);
    let (gene_sd, inv_sd) = scaling(&stats, params);
    let colsum = match params.clip.filter(|c| c.is_finite()) {
        Some(c) => {
            let caps: Vec<f32> =
                (0..p).map(|j| if inv_sd[j] == 0.0 { f32::INFINITY } else { (stats.mean[j] + c * gene_sd[j]) as f32 }).collect();
            csr.cap(&caps)?;
            csr.stats()?.sum
        }
        None => stats.mean.iter().map(|&m| m * n as f64).collect(),
    };
    let b = block_width(n, p, params);
    let op = cuda::CudaScaledOp::new(csr, &stats.mean, &inv_sd, colsum, b)?;
    let svd = if p <= GRAM_MAX_GENES {
        gram_svd(&op, b, params.power_iters, params.seed)?
    } else {
        block_krylov_svd(&op, b, params.power_iters, params.seed)?
    };
    Ok(assemble(svd, &stats, gene_sd, n, p, params.n_comps))
}

/// Krylov block width `n_comps + oversample`, capped at `min(n, p)`.
fn block_width(n: usize, p: usize, params: &PcaParams) -> usize {
    params.n_comps.saturating_add(params.oversample).min(n.min(p))
}

#[cfg(feature = "cuda")]
pub use cuda::DeviceCsr;

/// Per-gene sd used for scaling (1 when off or constant) and the operator's 1/sd (0 for
/// constant genes).
fn scaling(stats: &operator::GeneStats, params: &PcaParams) -> (Vec<f64>, Vec<f64>) {
    let p = stats.mean.len();
    let gene_sd: Vec<f64> = (0..p)
        .map(|j| if !params.scale || stats.constant[j] || stats.sd[j] == 0.0 { 1.0 } else { stats.sd[j] })
        .collect();
    let inv_sd = (0..p).map(|j| if stats.constant[j] { 0.0 } else { 1.0 / gene_sd[j] }).collect();
    (gene_sd, inv_sd)
}

/// (Q column-major, U_B, singular values, V_B) of [`block_krylov_svd`]: the operator is Q B
/// with B^T = U_B S V_B^T.
type Svd = (Vec<f64>, faer::Mat<f64>, Vec<f64>, faer::Mat<f64>);

/// Signs, loadings, embeddings and standard deviations from the SVD factors.
fn assemble((q, ub, d, vb): Svd, stats: &operator::GeneStats, gene_sd: Vec<f64>, n: usize, p: usize, k: usize) -> PcaResult {
    let signs: Vec<f64> = (0..k)
        .map(|c| {
            let col = ub.col(c);
            let mut best = 0usize;
            for j in 1..p {
                if col[j].abs() > col[best].abs() {
                    best = j;
                }
            }
            if col[best] < 0.0 { -1.0 } else { 1.0 }
        })
        .collect();

    // Constant genes have an exactly zero row in Z^T Q; the dense SVD leaves ~1e-24 round-off
    // there, so their loadings are set to the exact zero.
    let mut loadings = RowMajor::zeros(p, k);
    for j in (0..p).filter(|&j| !stats.constant[j]) {
        for c in 0..k {
            loadings.data[j * k + c] = (ub[(j, c)] * signs[c]) as f32;
        }
    }
    let q = faer::MatRef::from_column_major_slice(&q, n, vb.nrows());
    let embeddings = embed(q, vb.as_ref(), &d, &signs, k);
    let denom = ((n - 1) as f64).sqrt();
    PcaResult {
        embeddings,
        loadings,
        stdev: d[..k].iter().map(|s| (s / denom) as f32).collect(),
        gene_mean: stats.mean.iter().map(|&m| m as f32).collect(),
        gene_sd: gene_sd.iter().map(|&s| s as f32).collect(),
    }
}

fn validate<I: Idx, T: Real>(x: &CsrView<'_, I, T>, params: &PcaParams) -> Result<()> {
    let (n, p) = (x.n_rows, x.n_cols);
    // Cheap structural checks; per-entry checks happen in the statistics pass.
    if x.indptr.len() != n + 1 {
        return Err(Error::InvalidInput(format!("indptr has {} entries, expected n_rows + 1 = {}", x.indptr.len(), n + 1)));
    }
    if x.indices.len() != x.data.len() {
        return Err(Error::InvalidInput(format!(
            "indices ({}) and data ({}) differ in length",
            x.indices.len(),
            x.data.len()
        )));
    }
    if x.indptr[0].to_usize() != 0 || x.indptr[n].to_usize() != x.data.len() {
        return Err(Error::InvalidInput("indptr must start at 0 and end at nnz".into()));
    }
    validate_params(n, p, params)
}

fn validate_params(n: usize, p: usize, params: &PcaParams) -> Result<()> {
    if params.n_comps == 0 {
        return Err(Error::InvalidParameter("n_comps must be at least 1".into()));
    }
    if params.n_comps >= n.min(p) {
        return Err(Error::InvalidParameter(format!(
            "n_comps ({}) must be smaller than both the number of cells ({n}) and of genes ({p})",
            params.n_comps
        )));
    }
    if let Some(c) = params.clip {
        if c.is_nan() || c <= 0.0 {
            return Err(Error::InvalidParameter(format!("clip must be positive (or None), got {c}")));
        }
    }
    Ok(())
}

/// Block Krylov SVD (Musco & Musco 2015): subspace iteration that keeps every
/// power-iteration block, so the basis spans `[Z Ω, (Z Z^T) Z Ω, ...]` and the
/// top singular vectors converge much faster when singular values are close.
///
/// Each new block is orthonormalised against the previous ones as it is made (block
/// Lanczos with full reorthogonalisation), so K is orthonormal throughout and the
/// `Z^T K_i` the next block needs is also the block of `B^T = Z^T K` for the final thin
/// SVD: one sparse product per block in each direction, no orthonormalisation of the
/// whole cells x width basis.
/// Returns (K column-major: n x width orthonormal, U_b: p x width, d, V_b) with
/// `Z ≈ (K V_b) diag(d) U_b^T`.
fn block_krylov_svd<O: Operator>(op: &O, b: usize, iters: usize, seed: u64) -> Result<Svd> {
    let (n, p) = (op.nrows(), op.ncols());
    let blocks = (iters + 1).min(n.min(p) / b);
    let width = blocks * b;
    // K, column-major. `vec!` of zeros is a lazily zeroed allocation, so its pages are first
    // touched by the parallel products below rather than by a serial fill.
    let mut kbuf = vec![0.0f64; n * width];
    let mut k = faer::MatMut::from_column_major_slice_mut(&mut kbuf, n, width);
    let mut bt = faer::Mat::<f64>::zeros(p, width);
    let omega = dense::gaussian(p, b, seed);
    op.apply(omega.as_ref(), k.rb_mut().subcols_mut(0, b))?;
    drop(omega);
    dense::orthonormalize(k.rb_mut().subcols_mut(0, b));
    // A basis capped by min(n, p) cannot hold every block: the surplus iterations refine
    // the start block instead.
    for _ in blocks..iters + 1 {
        let mut g = op.apply_t(k.rb().subcols(0, b))?;
        dense::orthonormalize(g.as_mut());
        op.apply(g.as_ref(), k.rb_mut().subcols_mut(0, b))?;
        dense::orthonormalize(k.rb_mut().subcols_mut(0, b));
    }
    for i in 0..blocks {
        let mut g = op.apply_t(k.rb().subcols(i * b, b))?;
        bt.as_mut().subcols_mut(i * b, b).copy_from(g.as_ref());
        if i + 1 < blocks {
            dense::orthonormalize(g.as_mut());
            let (done, rest) = k.rb_mut().split_at_col_mut((i + 1) * b);
            let mut y = rest.subcols_mut(0, b);
            op.apply(g.as_ref(), y.rb_mut())?;
            // K^T y = K^T Z g = (Z^T K)^T g: the first projection needs no pass over K.
            let kty = dense::mul_tn(bt.as_ref().subcols(0, (i + 1) * b), g.as_ref());
            dense::orthonormalize_against(done.rb(), y, kty);
        }
    }
    let (ub, d, vb) = dense::thin_svd(bt.as_ref())?;
    Ok((kbuf, ub, d, vb))
}

/// Largest gene count for which the CUDA path forms the Gram matrix (8 p^2 bytes on the
/// device; 0.5 GiB at this size).
#[cfg(feature = "cuda")]
const GRAM_MAX_GENES: usize = 8192;

/// Largest gene count for which [`gram_svd`] takes the exact eigendecomposition of the Gram
/// matrix on the device (cuSOLVER; cubic in the gene count).
#[cfg(feature = "cuda")]
const EIG_MAX_GENES: usize = 4096;

/// [`block_krylov_svd`] for the device: G = Z^T Z (genes x genes) in one pass over the
/// data, then either its exact eigendecomposition on the device (up to [`EIG_MAX_GENES`]) or,
/// on the host, the gene-side Krylov basis `[Ω, G Ω, ..., G^q Ω]` (Z times it is the cell-side basis of
/// `block_krylov_svd`, so the same `power_iters` gives the same subspace) with Rayleigh-Ritz
/// on `K^T G K`; then one `apply` for the cell scores. Avoids 2 (power_iters + 1) cells x b
/// round trips and their host QR factorisations. Returns (U, V, S, I) so that `embed` gives Z V.
#[cfg(feature = "cuda")]
fn gram_svd(op: &cuda::CudaScaledOp, b: usize, iters: usize, seed: u64) -> Result<Svd> {
    let (n, p) = (op.nrows(), op.ncols());
    let (vr, lambda) = if p <= EIG_MAX_GENES {
        op.gram_top_eigen(b)?
    } else {
        let g = op.gram()?;
        let blocks = (iters + 1).min(p / b);
        let mut k = faer::Mat::<f64>::zeros(p, blocks * b);
        let mut v = dense::gaussian(p, b, seed);
        for i in 0..blocks {
            dense::orthonormalize(v.as_mut());
            k.as_mut().submatrix_mut(0, i * b, p, b).copy_from(v.as_ref());
            v = &g * &v;
        }
        dense::orthonormalize(k.as_mut());
        let h = k.transpose() * &g * &k;
        let (w, lambda, _) = dense::thin_svd(h.as_ref())?;
        (&k * w.as_ref().subcols(0, b), lambda[..b].to_vec())
    };
    let s: Vec<f64> = lambda.iter().map(|&x| x.max(0.0).sqrt()).collect();
    let mut u = faer::Mat::<f64>::zeros(n, b);
    op.apply(vr.as_ref(), u.as_mut())?;
    for (c, &sc) in s.iter().enumerate() {
        let inv = if sc > 0.0 { 1.0 / sc } else { 0.0 };
        u.col_mut(c).iter_mut().for_each(|y| *y *= inv);
    }
    let u: Vec<f64> = (0..b).flat_map(|c| u.col(c).iter().copied().collect::<Vec<_>>()).collect();
    Ok((u, vr, s, faer::Mat::<f64>::identity(b, b)))
}

/// Embeddings `Q V_b[:, :k] diag(d_k signs)` as row-major f32, parallel over fixed row
/// chunks (each a sequential faer product, so thread-count independent).
fn embed(q: faer::MatRef<'_, f64>, vb: faer::MatRef<'_, f64>, d: &[f64], signs: &[f64], k: usize) -> RowMajor {
    use rayon::prelude::*;
    const ROWS: usize = 1024;
    let n = q.nrows();
    let w = faer::Mat::<f64>::from_fn(vb.nrows(), k, |t, c| vb[(t, c)] * d[c] * signs[c]);
    let mut out = RowMajor::zeros(n, k);
    out.data.par_chunks_mut(ROWS * k).zip(q.par_row_chunks(ROWS)).for_each(|(dst, qc)| {
        let mut e = faer::Mat::<f64>::zeros(qc.nrows(), k);
        faer::linalg::matmul::matmul(e.as_mut(), faer::Accum::Replace, qc, w.as_ref(), 1.0, faer::Par::Seq);
        for r in 0..qc.nrows() {
            for c in 0..k {
                dst[r * k + c] = e[(r, c)] as f32;
            }
        }
    });
    out
}
