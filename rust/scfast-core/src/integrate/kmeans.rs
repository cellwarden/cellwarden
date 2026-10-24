//! Seeded k-means for the Harmony initialisation, as harmony 2.x `kmeans_centers`
//! (src/utils.cpp): a distance-weighted seeding followed by plain Lloyd steps on the
//! cosine-normalised cells.
//!
//! # Seeding (`initialize_centroids`)
//!
//! K cells are drawn uniformly (`floor(u (n - 1))`); then, for each centroid i, every cell j
//! gets the key `-ln(u_j) / |2 (1 - y_i . x_j)|` (an exponential race, i.e. a draw with
//! probability proportional to the cosine distance from the i-th uniform draw) and the cell
//! with the smallest key that is not a centroid yet becomes centroid i. The reference
//! points are fixed up front, so all K races are scored in one parallel GEMM pass.
//!
//! # Lloyd steps (`arma::kmeans(keep_existing)`, one iteration, called `iters` times)
//!
//! Assign every cell to the nearest centroid (squared Euclidean, ties to the lower index),
//! then move each centroid to the mean of its cells. A centroid without cells is moved to
//! the cell farthest from its own centroid (Armadillo uses a similar rule; the choice of
//! cell differs). Steps stop early at a fixed point, which changes nothing.
//!
//! # Determinism
//!
//! Random numbers come from ChaCha8 streams addressed by (start, centroid, cell) through
//! `set_word_pos`, cells are processed in fixed chunks and partial sums are added in chunk
//! order, so the centroids do not depend on the number of rayon threads.

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::fastmath::ln_pos;

/// Cells per task.
const CHUNK: usize = 2048;

/// `y^T x` for the centroids `y` (d x k) and the cells `x` (d x m), into `out` (k x m), all
/// column-major.
pub(super) fn scores(y: &[f64], x: &[f64], d: usize, k: usize, out: &mut [f64]) {
    let m = x.len() / d;
    let y = MatRef::from_column_major_slice(y, d, k);
    let x = MatRef::from_column_major_slice(x, d, m);
    matmul(MatMut::from_column_major_slice_mut(out, k, m), Accum::Replace, y.transpose(), x, 1.0, Par::Seq);
    clear_upper();
}

/// faer's AVX / AVX-512 kernels can return with the upper halves of the vector registers
/// dirty; the SSE2 code that follows (the crate is built for baseline x86-64) then pays a
/// state-transition penalty on every instruction (the soft assignment ran 3x slower on
/// AMD EPYC). `vzeroupper` clears that state. Call after each faer product whose result
/// is consumed by scalar loops.
#[inline]
pub(super) fn clear_upper() {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx") {
        // SAFETY: AVX is available (checked above); no vector values are live here.
        unsafe { zeroupper() }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn zeroupper() {
    std::arch::x86_64::_mm256_zeroupper()
}

/// Centroids (d x k, column per centroid) of the cells `x` (d x n, column per cell, unit
/// norm): the best of `nstart` seeded runs by within-cluster sum of squares.
pub(super) fn kmeans_centers(x: &[f64], d: usize, n: usize, k: usize, nstart: usize, iters: usize, seed: u64) -> Vec<f64> {
    let mut best: Option<(f64, Vec<f64>)> = None;
    for start in 0..nstart.max(1) {
        let mut y = seed_centroids(x, d, n, k, seed, start as u64);
        let sse = lloyd(x, d, n, k, &mut y, iters);
        if best.as_ref().is_none_or(|(b, _)| sse < *b) {
            best = Some((sse, y));
        }
    }
    best.map(|(_, y)| y).unwrap_or_default()
}

/// One uniform in (0, 1] per (centroid draw, cell) of start `start`: word position
/// `2 * index` of ChaCha8 stream `1 + start` (each f64 consumes one u64).
fn uniforms(seed: u64, start: u64, first: u128, out: &mut [f64]) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(1 + start);
    rng.set_word_pos(2 * first);
    for v in out.iter_mut() {
        *v = 1.0 - rng.random::<f64>();
    }
}

/// Candidates kept per centroid and chunk in [`seed_centroids`].
const SEED_CANDIDATES: usize = 4;

/// harmony's `initialize_centroids`. Centroid i is drawn relative to the i-th uniform draw
/// `x_{idx_i}`, which no earlier step changes, so the keys of all K draws come from one
/// GEMM pass over the cells; each keeps its [`SEED_CANDIDATES`] smallest keys, and the
/// sequential part only skips cells that earlier centroids took (with a full rescan in
/// the unlikely case that all candidates are taken).
fn seed_centroids(x: &[f64], d: usize, n: usize, k: usize, seed: u64, start: u64) -> Vec<f64> {
    let mut u0 = vec![0f64; k];
    uniforms(seed, start, 0, &mut u0);
    let mut y0 = vec![0f64; d * k];
    for (i, &u) in u0.iter().enumerate() {
        // harmony draws in [0, 1) and floors u (n - 1); (0, 1] here, so 1 - u keeps the range.
        let j = (((1.0 - u) * (n - 1) as f64).floor() as usize).min(n - 1);
        y0[i * d..(i + 1) * d].copy_from_slice(&x[j * d..(j + 1) * d]);
    }
    // per chunk: for each centroid, its smallest (key, cell) pairs, ascending
    let parts: Vec<Vec<Vec<(f64, usize)>>> = x
        .par_chunks(CHUNK * d)
        .enumerate()
        .map(|(c, xc)| {
            let m = xc.len() / d;
            let j0 = c * CHUNK;
            // x^T y0 (m x k, column per centroid), so each race reads a contiguous column
            let mut s = vec![0f64; m * k];
            let (xr, yr) = (MatRef::from_column_major_slice(xc, d, m), MatRef::from_column_major_slice(&y0, d, k));
            matmul(MatMut::from_column_major_slice_mut(&mut s, m, k), Accum::Replace, xr.transpose(), yr, 1.0, Par::Seq);
            clear_upper();
            let mut u = vec![0f64; m];
            (0..k)
                .map(|i| {
                    uniforms(seed, start, (k + i * n + j0) as u128, &mut u);
                    let mut best: Vec<(f64, usize)> = Vec::with_capacity(SEED_CANDIDATES + 1);
                    for (jj, &sj) in s[i * m..(i + 1) * m].iter().enumerate() {
                        let dist = (2.0 * (1.0 - sj)).abs();
                        if dist == 0.0 {
                            continue; // key +inf: never drawn
                        }
                        let key = -ln_pos(u[jj]) / dist;
                        if best.len() < SEED_CANDIDATES || key < best[best.len() - 1].0 {
                            let at = best.partition_point(|&(b, _)| b <= key);
                            best.insert(at, (key, j0 + jj));
                            best.truncate(SEED_CANDIDATES);
                        }
                    }
                    best
                })
                .collect()
        })
        .collect();
    let mut chosen = vec![false; n];
    let mut y = vec![0f64; d * k];
    for i in 0..k {
        let mut cand: Vec<(f64, usize)> = parts.iter().flat_map(|p| p[i].iter().copied()).collect();
        cand.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let j = match cand.iter().take(SEED_CANDIDATES).find(|&&(_, j)| !chosen[j]) {
            Some(&(_, j)) => j,
            None => rescan(x, d, n, k, &y0[i * d..(i + 1) * d], i, &chosen, seed, start),
        };
        chosen[j] = true;
        y[i * d..(i + 1) * d].copy_from_slice(&x[j * d..(j + 1) * d]);
    }
    y
}

/// The cell with the smallest key for centroid `i` among those not `chosen`.
#[allow(clippy::too_many_arguments)]
fn rescan(x: &[f64], d: usize, n: usize, k: usize, yi: &[f64], i: usize, chosen: &[bool], seed: u64, start: u64) -> usize {
    let mut u = vec![0f64; n];
    uniforms(seed, start, (k + i * n) as u128, &mut u);
    let mut best: Option<(f64, usize)> = None;
    for (j, xj) in x.chunks_exact(d).enumerate() {
        if chosen[j] {
            continue;
        }
        let dot: f64 = yi.iter().zip(xj).map(|(a, b)| a * b).sum();
        let dist = (2.0 * (1.0 - dot)).abs();
        if dist == 0.0 {
            continue;
        }
        let key = -ln_pos(u[j]) / dist;
        if best.is_none_or(|(b, _)| key < b) {
            best = Some((key, j));
        }
    }
    // Every remaining cell coincides with the seed point: take the first one not used.
    best.map(|(_, j)| j).or_else(|| chosen.iter().position(|&c| !c)).unwrap_or(0)
}

/// Per-chunk partial result of an assignment pass.
struct Assign {
    sums: Vec<f64>,
    counts: Vec<usize>,
    sse: f64,
}

/// `iters` Lloyd steps on `y` (d x k) in place; returns the within-cluster sum of squares
/// of the last assignment.
fn lloyd(x: &[f64], d: usize, n: usize, k: usize, y: &mut [f64], iters: usize) -> f64 {
    let mut label = vec![u32::MAX; n];
    let mut sse = f64::INFINITY;
    for _ in 0..iters {
        let ynorm: Vec<f64> = y.chunks_exact(d).map(|c| c.iter().map(|v| v * v).sum()).collect();
        let parts: Vec<(Assign, bool)> = x
            .par_chunks(CHUNK * d)
            .zip(label.par_chunks_mut(CHUNK))
            .map(|(xc, lc)| {
                let m = xc.len() / d;
                let mut s = vec![0f64; k * m];
                scores(y, xc, d, k, &mut s);
                let mut part = Assign { sums: vec![0f64; d * k], counts: vec![0; k], sse: 0.0 };
                let mut changed = false;
                for (jj, (sj, xj)) in s.chunks_exact(k).zip(xc.chunks_exact(d)).enumerate() {
                    let (mut best, mut bd) = (0usize, f64::INFINITY);
                    for (c, (&sc, &yn)) in sj.iter().zip(&ynorm).enumerate() {
                        let dist = yn - 2.0 * sc;
                        if dist < bd {
                            bd = dist;
                            best = c;
                        }
                    }
                    let xn: f64 = xj.iter().map(|v| v * v).sum();
                    part.sse += (bd + xn).max(0.0);
                    changed |= lc[jj] != best as u32;
                    lc[jj] = best as u32;
                    part.counts[best] += 1;
                    for (a, &v) in part.sums[best * d..(best + 1) * d].iter_mut().zip(xj) {
                        *a += v;
                    }
                }
                (part, changed)
            })
            .collect();
        let changed = parts.iter().any(|(_, c)| *c);
        let mut sums = vec![0f64; d * k];
        let mut counts = vec![0usize; k];
        sse = 0.0;
        for (p, _) in &parts {
            for (a, b) in sums.iter_mut().zip(&p.sums) {
                *a += b;
            }
            for (a, b) in counts.iter_mut().zip(&p.counts) {
                *a += b;
            }
            sse += p.sse;
        }
        if !changed {
            break; // fixed point: the update below would reproduce y
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f64;
                for (yv, &s) in y[c * d..(c + 1) * d].iter_mut().zip(&sums[c * d..(c + 1) * d]) {
                    *yv = s * inv;
                }
            }
        }
        let dead: Vec<usize> = (0..k).filter(|&c| counts[c] == 0).collect();
        if !dead.is_empty() {
            revive(x, d, &label, y, &dead);
        }
    }
    sse
}

/// Moves each dead centroid to one of the cells farthest from their own centroid (largest
/// first, ties to the lower index).
fn revive(x: &[f64], d: usize, label: &[u32], y: &mut [f64], dead: &[usize]) {
    let far: Vec<f64> = x
        .par_chunks(d)
        .zip(label.par_iter())
        .map(|(xj, &l)| {
            let yc = &y[l as usize * d..(l as usize + 1) * d];
            xj.iter().zip(yc).map(|(a, b)| (a - b) * (a - b)).sum()
        })
        .collect();
    let mut order: Vec<usize> = (0..far.len()).collect();
    order.sort_by(|&a, &b| far[b].total_cmp(&far[a]).then(a.cmp(&b)));
    for (&c, &j) in dead.iter().zip(&order) {
        y[c * d..(c + 1) * d].copy_from_slice(&x[j * d..(j + 1) * d]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit-norm cells around `k` well separated directions.
    fn clustered(n: usize, d: usize, k: usize) -> (Vec<f64>, Vec<usize>) {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut x = vec![0f64; n * d];
        let mut truth = vec![0; n];
        for j in 0..n {
            let c = j % k;
            truth[j] = c;
            for t in 0..d {
                let centre = if t == c { 1.0 } else { 0.0 };
                x[j * d + t] = centre + 0.05 * (rng.random::<f64>() - 0.5);
            }
            let norm: f64 = x[j * d..(j + 1) * d].iter().map(|v| v * v).sum::<f64>().sqrt();
            x[j * d..(j + 1) * d].iter_mut().for_each(|v| *v /= norm);
        }
        (x, truth)
    }

    #[test]
    fn finds_separated_clusters_and_is_thread_independent() {
        let (n, d, k) = (3000, 6, 4);
        let (x, _) = clustered(n, d, k);
        let y = kmeans_centers(&x, d, n, k, 3, 10, 1);
        // every true direction has a centroid close to it
        for c in 0..k {
            let best = (0..k).map(|i| y[i * d + c]).fold(f64::MIN, f64::max);
            assert!(best > 0.95, "direction {c}: best centroid coordinate {best}");
        }
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let y1 = pool.install(|| kmeans_centers(&x, d, n, k, 3, 10, 1));
        assert!(y == y1, "k-means differs between thread counts");
        assert!(y != kmeans_centers(&x, d, n, k, 1, 0, 2), "seed must matter");
    }
}
