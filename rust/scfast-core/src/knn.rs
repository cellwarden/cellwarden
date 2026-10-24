//! k nearest neighbours on PC embeddings: replaces the kNN part of Seurat's
//! `FindNeighbors` (Annoy, approximate) with exact search for moderate sizes and an
//! approximate k-d tree (up to 16 dimensions) or HNSW for large ones.
//!
//! # Output contract (all methods)
//!
//! Row `i` of the result holds `k` entries: the cell itself first (distance 0), then the
//! `k - 1` other cells in increasing Euclidean distance, ties broken by lower index. The
//! reported distances are Euclidean (not squared), recomputed in f64 from the f32 inputs
//! and rounded to f32 once, so exact and HNSW results agree wherever they pick the same
//! neighbours. Duplicated points of `i` come after `i` itself even when their index is
//! lower.
//!
//! # Exact (`KnnMethod::Exact`)
//!
//! Up to `KDTREE_MAX_DIMS` dimensions: a k-d tree searched to completion (`knn/kdtree.rs`),
//! which returns the same answer as the brute force below, bit for bit. Measured on the
//! 10 leading PCs of 500k synthetic cells (12-core M-series, shared): 3.9 s wall (41 CPU-s)
//! against 11.3 s for the brute force, whose cost grows as n^2.
//!
//! Above that, blocked brute force. Data are centred (translation does not change distances but makes
//! the norms, and so the rounding error below, small), then for a block of queries and a
//! chunk of reference points the cross term comes from one dense f32 GEMM (faer):
//! `|q - r|^2 = |q|^2 + |r|^2 - 2 q.r`. Each query keeps a bounded max-heap of
//! `k - 1 + slack` candidates, ranked by the GEMM estimate. The candidates' distances are
//! then recomputed exactly (f64, direct differences), and a rigorous rounding-error bound
//! checks that no point outside the heap could beat the `(k - 1)`-th exact distance; the
//! rare query that fails the check (for example more than `slack` points tied at the same
//! distance) is redone by a direct f64 scan. The result is therefore exactly the naive
//! O(n^2) answer, bit for bit, independent of thread count and CPU features.
//!
//! Memory: one `QUERY_BLOCK x REF_CHUNK` f32 buffer (2 MiB) and the candidate heaps per
//! worker, plus a centred n x d copy of the input. Never n x n.
//!
//! # HNSW (`KnnMethod::Hnsw`)
//!
//! hnsw_rs 0.3 with squared L2 (a monotone transform of L2, so every comparison HNSW makes
//! is unchanged and the square root is skipped). Points are inserted in parallel, in an
//! order shuffled by `params.seed`; each cell is then searched with
//! `ef = max(ef_search, k)`. Results are post-processed to the same contract: self forced
//! to the front, duplicates dropped, exact distances recomputed and sorted with the index
//! tie-break. A query that comes back with fewer than `k - 1` other points falls back to a
//! direct scan.
//!
//! Determinism: **HNSW results are not bit-reproducible across runs.** hnsw_rs draws
//! each point's layer from one mutex-guarded RNG with a fixed internal seed (397) that the
//! public API does not expose, and parallel insertion makes both the order of those draws
//! and the graph wiring depend on thread timing. `params.seed` only fixes the insertion
//! order. Repeated runs give nearly identical neighbour sets (recall vs exact is what is
//! guaranteed in practice, not identity). Use `Exact` when reproducibility matters.
//!
//! # Approximate k-d tree (`KnnMethod::KdTree`)
//!
//! The exact k-d tree search with the (1 + eps) rule of Arya et al.: a node is skipped once
//! its box is farther than the current (k - 1)-th distance divided by `1 + eps`.
//! Deterministic and independent of the thread count; distances and ordering follow the
//! contract above. Measured with eps = 0.7 (10 PCs, k = 20, 12-core M-series): share of each
//! cell's neighbours that the exact answer also has 0.957 / 0.955 / 0.946 on 100k and 500k
//! synthetic cells and the 1.29M-cell 10x brain set, in 0.12 / 0.96 / 4.6 s against
//! 0.31 / 3.3 / 12.6 s exact. A fixed cap on visited leaves was tried first: its overlap fell
//! with n (0.96 at 500k synthetic, 0.82 at 1.29M brain for the same cap). HNSW
//! (m = 16, ef_construction = 200) took 8.5 s at 100k cells.
//!
//! # Auto (`KnnMethod::Auto`)
//!
//! `Exact` when `n < auto_threshold`. Otherwise `KdTree { eps: 0.7 }` up to
//! `KDTREE_MAX_DIMS` dimensions, and HNSW with m = 16, ef_construction = 200,
//! ef_search = max(2k, 64) above.

use crate::device::Device;
use crate::error::{Error, Result};
use crate::types::{Knn, RowMajor};

#[cfg(feature = "cuda")]
mod cuda;
mod exact;
mod hnsw;
mod kdtree;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KnnMethod {
    /// Exact brute force (blocked, parallel). O(n^2 d): fine up to roughly 50k-100k cells.
    Exact,
    /// Approximate HNSW (hnsw_rs). `m` = links per node, `ef_construction`, `ef_search`.
    Hnsw { m: usize, ef_construction: usize, ef_search: usize },
    /// k-d tree search with the (1 + eps) pruning rule; approximate for `eps > 0`, exact
    /// at 0. Meant for up to `KDTREE_MAX_DIMS` dimensions (any works, but slows down above).
    KdTree { eps: f64 },
    /// Exact below `auto_threshold` cells; above, `KdTree` up to `KDTREE_MAX_DIMS`
    /// dimensions and HNSW beyond.
    Auto,
}

#[derive(Clone, Debug)]
pub struct KnnParams {
    /// Neighbours per cell INCLUDING the cell itself (Seurat `k.param`, default 20).
    pub k: usize,
    pub method: KnnMethod,
    /// Cell count where `Auto` switches from exact to HNSW. Set from the bench scaling curve.
    pub auto_threshold: usize,
    /// Seed for HNSW construction (exact search ignores it).
    pub seed: u64,
}

impl Default for KnnParams {
    fn default() -> Self {
        Self { k: 20, method: KnnMethod::Auto, auto_threshold: 50_000, seed: 0 }
    }
}

/// Up to this many dimensions exact search uses the k-d tree, and `Auto` the approximate
/// k-d tree, instead of the GEMM brute force and HNSW.
pub const KDTREE_MAX_DIMS: usize = 16;
/// `eps` used by `KnnMethod::Auto` (see the module docs for the measurement).
pub const AUTO_KD_EPS: f64 = 0.7;

/// HNSW settings used by `KnnMethod::Auto` above `auto_threshold`.
const AUTO_M: usize = 16;
const AUTO_EF_CONSTRUCTION: usize = 200;
const AUTO_EF_SEARCH_MIN: usize = 64;

/// hnsw_rs aborts the process (not a panic) above 256 links per node.
const HNSW_MAX_M: usize = 256;

/// k nearest neighbours (Euclidean) of every row of `x` (cells x dims, e.g. the first 30
/// PCs). Returns `Knn` with the cell itself first (distance 0) and the remaining k-1
/// neighbours in increasing distance; ties broken by lower index so results are
/// deterministic.
pub fn knn(x: &RowMajor, params: &KnnParams) -> Result<Knn> {
    knn_on(x, params, Device::Cpu)
}

/// [`knn`] on a chosen [`Device`] (docs/scfast-integration.md Phase 6).
///
/// `Device::Cpu` is exactly [`knn`]. `Device::Cuda(i)` runs a fused distance + top-k kernel
/// (one query per thread), then re-ranks the candidates by exact f64
/// distances on the host (same output contract: self first, Euclidean distances, ties by
/// lower index); `params.method` is ignored there (the GPU search is brute force at every
/// size) and `k` must be at most 129. Needs the `cuda` feature, a visible device and
/// device memory for the embeddings and the candidate lists, otherwise [`Error::Gpu`] is
/// returned without touching the CPU path.
pub fn knn_on(x: &RowMajor, params: &KnnParams, device: Device) -> Result<Knn> {
    validate_input(x)?;
    let n = x.n_rows;
    let k = params.k;
    if k < 2 {
        return Err(Error::InvalidParameter(format!("k = {k}: k counts the cell itself and must be at least 2")));
    }
    if k > n {
        return Err(Error::InvalidParameter(format!("k = {k} exceeds the number of cells ({n})")));
    }
    match device {
        Device::Cpu => {}
        #[cfg(feature = "cuda")]
        Device::Cuda(ordinal) => return cuda::cuda_knn(x, k, ordinal),
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => {
            return Err(Error::gpu("scfast was built without CUDA support (feature `cuda`); use Device::Cpu"));
        }
    }
    match params.method {
        KnnMethod::Exact => Ok(exact_knn(x, k)),
        KnnMethod::Hnsw { m, ef_construction, ef_search } => {
            validate_hnsw(m, ef_construction, ef_search)?;
            Ok(hnsw::hnsw_knn(x, k, m, ef_construction, ef_search, params.seed))
        }
        KnnMethod::KdTree { eps } => {
            if !(eps >= 0.0 && eps.is_finite()) {
                return Err(Error::InvalidParameter(format!("k-d tree eps must be finite and >= 0, got {eps}")));
            }
            Ok(kdtree::kdtree_knn(x, k, eps))
        }
        KnnMethod::Auto => {
            if n < params.auto_threshold {
                Ok(exact_knn(x, k))
            } else if x.n_cols <= KDTREE_MAX_DIMS {
                Ok(kdtree::kdtree_knn(x, k, AUTO_KD_EPS))
            } else {
                let ef_search = (2 * k).max(AUTO_EF_SEARCH_MIN);
                Ok(hnsw::hnsw_knn(x, k, AUTO_M, AUTO_EF_CONSTRUCTION, ef_search, params.seed))
            }
        }
    }
}

fn exact_knn(x: &RowMajor, k: usize) -> Knn {
    if x.n_cols <= KDTREE_MAX_DIMS {
        kdtree::kdtree_knn(x, k, 0.0)
    } else {
        exact::exact_knn(x, k)
    }
}

fn validate_input(x: &RowMajor) -> Result<()> {
    if x.data.len() != x.n_rows * x.n_cols {
        return Err(Error::InvalidInput(format!(
            "{} values for a {}x{} matrix",
            x.data.len(),
            x.n_rows,
            x.n_cols
        )));
    }
    if x.n_cols == 0 {
        return Err(Error::InvalidInput("x has no columns (dimensions)".into()));
    }
    if x.n_rows > u32::MAX as usize {
        return Err(Error::InvalidInput(format!("{} cells exceed the u32 index range", x.n_rows)));
    }
    if let Some(pos) = x.data.iter().position(|v| !v.is_finite()) {
        return Err(Error::InvalidInput(format!(
            "non-finite value {} at cell {}, dimension {}",
            x.data[pos],
            pos / x.n_cols,
            pos % x.n_cols
        )));
    }
    Ok(())
}

fn validate_hnsw(m: usize, ef_construction: usize, ef_search: usize) -> Result<()> {
    if !(2..=HNSW_MAX_M).contains(&m) {
        return Err(Error::InvalidParameter(format!("HNSW m = {m}: must be between 2 and {HNSW_MAX_M}")));
    }
    if ef_construction == 0 {
        return Err(Error::InvalidParameter("HNSW ef_construction must be at least 1".into()));
    }
    if ef_search == 0 {
        return Err(Error::InvalidParameter("HNSW ef_search must be at least 1".into()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// Shared helpers for both methods.

/// Exact squared distance in f64 (f32 inputs, products exact in f64, fixed summation
/// order). Every distance this module reports goes through here.
#[inline]
fn exact_d2(a: &[f32], b: &[f32]) -> f64 {
    let mut s = 0.0f64;
    for (&u, &v) in a.iter().zip(b) {
        let t = u as f64 - v as f64;
        s += t * t;
    }
    s
}

/// Total order used for neighbours: distance, then lower index.
#[inline]
fn cmp_neighbour(a: &(f64, u32), b: &(f64, u32)) -> std::cmp::Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
}

/// The `kk` nearest other cells of cell `i` by a direct f64 scan, sorted by
/// `cmp_neighbour`. O(n d); used as a fallback only.
fn brute_one(x: &RowMajor, i: usize, kk: usize, scratch: &mut Vec<(f64, u32)>) {
    scratch.clear();
    let q = x.row(i);
    scratch.extend((0..x.n_rows).filter(|&j| j != i).map(|j| (exact_d2(q, x.row(j)), j as u32)));
    if kk < scratch.len() {
        scratch.select_nth_unstable_by(kk - 1, cmp_neighbour);
        scratch.truncate(kk);
    }
    scratch.sort_unstable_by(cmp_neighbour);
}

/// Writes one output row: self first, then the first `k - 1` entries of `sorted`.
#[inline]
fn write_row(i: usize, sorted: &[(f64, u32)], idx_out: &mut [u32], dist_out: &mut [f32]) {
    idx_out[0] = i as u32;
    dist_out[0] = 0.0;
    for (t, &(d2, j)) in sorted.iter().take(idx_out.len() - 1).enumerate() {
        idx_out[t + 1] = j;
        dist_out[t + 1] = d2.sqrt() as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn data(n: usize, d: usize, offset: f32, scale: f32, seed: u64) -> RowMajor {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let v = (0..n * d).map(|_| offset + scale * (rng.random::<f32>() - 0.5)).collect();
        RowMajor::from_vec(n, d, v).unwrap()
    }

    /// The error-bound certificate must pass for almost every query on ordinary data,
    /// otherwise exact search silently degrades to the O(n^2 d) f64 scan.
    #[test]
    fn exact_certificate_rarely_falls_back() {
        for (label, x) in [
            ("uniform 30d", data(3000, 30, 0.0, 1.0, 1)),
            ("uniform 30d offset 100", data(3000, 30, 100.0, 1.0, 2)),
            ("uniform 30d scale 1e3", data(3000, 30, 0.0, 1e3, 3)),
            ("uniform 5d", data(3000, 5, 0.0, 1.0, 4)),
        ] {
            let (_, fb) = exact::exact_knn_with_stats(&x, 20);
            println!("{label}: {fb} of {} queries fell back", x.n_rows);
            assert!(fb * 100 <= x.n_rows, "{label}: {fb} of {} queries fell back", x.n_rows);
        }
    }

    #[test]
    fn hnsw_rarely_falls_back() {
        let x = data(2000, 30, 0.0, 1.0, 5);
        let (_, fb) = hnsw::hnsw_knn_with_stats(&x, 20, 16, 100, 64, 0);
        assert!(fb * 100 <= x.n_rows, "{fb} of {} cells fell back", x.n_rows);
    }
}
