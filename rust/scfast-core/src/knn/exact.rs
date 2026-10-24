//! Exact kNN by blocked brute force (method and error bound: see the parent module docs).
//!
//! For a block of queries `Q` and a chunk of reference points `R`, one f32 GEMM produces
//! `s_qr = |r|^2 - 2 q.r` directly, by augmenting the operands:
//! `[q, 1] . [-2 r, |r|^2]`. Adding `|q|^2` (in f64) gives the squared-distance estimate.
//! Each query keeps the `cap` smallest estimates in a bounded max-heap; the candidates are
//! then re-ranked by exact f64 distances and certified with a rounding-error bound.

use super::{brute_one, cmp_neighbour, exact_d2, write_row};
use crate::types::{Knn, RowMajor};
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use rayon::prelude::*;

/// Queries per GEMM block (upper bound; smaller for small n so every core gets work).
const MAX_QUERY_BLOCK: usize = 256;
const MIN_QUERY_BLOCK: usize = 32;
/// Reference points per GEMM chunk. Buffer = block x chunk f32 = 2 MiB at most.
const REF_CHUNK: usize = 2048;
/// Lanes checked at once before falling into the scalar insertion path.
const SCAN_LANES: usize = 16;
/// Above this squared norm the f32 GEMM could overflow; use the direct f64 scan instead.
const F32_SAFE_SQ_NORM: f64 = 1e36;

/// Exact kNN plus the number of queries that needed the direct-scan fallback.
pub(super) fn exact_knn_with_stats(x: &RowMajor, k: usize) -> (Knn, usize) {
    let n = x.n_rows;
    let d = x.n_cols;
    let kk = k - 1; // neighbours other than the cell itself
    let mut indices = vec![0u32; n * k];
    let mut distances = vec![0f32; n * k];

    let prep = Prepared::new(x);
    if prep.rho_max * prep.rho_max * 4.0 > F32_SAFE_SQ_NORM {
        // Values so large that f32 squares overflow: direct f64 scan for every cell.
        indices
            .par_chunks_mut(k)
            .zip(distances.par_chunks_mut(k))
            .enumerate()
            .for_each_init(Vec::new, |cand, (i, (irow, drow))| {
                brute_one(x, i, kk, cand);
                write_row(i, cand, irow, drow);
            });
        return (Knn { n, k, indices, distances }, n);
    }

    // Candidates kept per query: k - 1 plus slack, so the certificate below usually passes.
    let slack = (kk / 2).max(8);
    let cap = (kk + slack).min(n - 1);
    let threads = rayon::current_num_threads().max(1);
    let qb = n.div_ceil(4 * threads).clamp(MIN_QUERY_BLOCK, MAX_QUERY_BLOCK);
    let rc = REF_CHUNK.min(n);
    let da = d + 1;

    let fallbacks: usize = indices
        .par_chunks_mut(qb * k)
        .zip(distances.par_chunks_mut(qb * k))
        .enumerate()
        .map_init(
            || Workspace::new(qb, rc, cap),
            |ws, (b, (idx_blk, dist_blk))| {
                let q0 = b * qb;
                let nq = idx_blk.len() / k;
                ws.lens[..nq].fill(0);

                // 1. GEMM over reference chunks, keeping the `cap` best estimates per query.
                let rhs = MatRef::from_row_major_slice(&prep.qaug[q0 * da..(q0 + nq) * da], nq, da).transpose();
                for j0 in (0..n).step_by(rc) {
                    let nr = rc.min(n - j0);
                    let buf = &mut ws.gemm[..nr * nq];
                    {
                        let dst = MatMut::from_column_major_slice_mut(&mut *buf, nr, nq);
                        let lhs = MatRef::from_row_major_slice(&prep.raug[j0 * da..(j0 + nr) * da], nr, da);
                        matmul(dst, Accum::Replace, lhs, rhs, 1.0f32, Par::Seq);
                    }
                    for t in 0..nq {
                        let col = &buf[t * nr..(t + 1) * nr];
                        let heap = &mut ws.heaps[t * cap..(t + 1) * cap];
                        scan(col, j0, q0 + t, heap, &mut ws.lens[t]);
                    }
                }

                // 2. Exact re-ranking of the candidates, certified by the error bound.
                let mut fallbacks = 0usize;
                for t in 0..nq {
                    let i = q0 + t;
                    let len = ws.lens[t];
                    let heap = &ws.heaps[t * cap..t * cap + len];
                    let xi = x.row(i);
                    let cand = &mut ws.cand;
                    cand.clear();
                    cand.extend(heap.iter().map(|&(_, j)| (exact_d2(xi, x.row(j as usize)), j)));
                    cand.sort_unstable_by(cmp_neighbour);
                    // Every non-candidate j has estimate >= the heap maximum, and
                    // |estimate - exact| <= bound, so it is provably farther than the
                    // (k-1)-th candidate whenever heap_max - bound > that distance.
                    let certified = len == n - 1 || {
                        let est_worst = prep.ny[i] + heap[0].0 as f64;
                        est_worst - prep.bound(i) > cand[kk - 1].0
                    };
                    if !certified {
                        brute_one(x, i, kk, cand);
                        fallbacks += 1;
                    }
                    write_row(i, cand, &mut idx_blk[t * k..(t + 1) * k], &mut dist_blk[t * k..(t + 1) * k]);
                }
                fallbacks
            },
        )
        .sum();

    (Knn { n, k, indices, distances }, fallbacks)
}

pub(super) fn exact_knn(x: &RowMajor, k: usize) -> Knn {
    exact_knn_with_stats(x, k).0
}

/// Centred, augmented copies of the data and the per-point error-bound terms.
struct Prepared {
    /// n x (d+1) row-major: `[y_i, 1]`, with `y = x - mean` rounded to f32.
    qaug: Vec<f32>,
    /// n x (d+1) row-major: `[-2 y_j, |y_j|^2]`.
    raug: Vec<f32>,
    /// `|y_i|^2` in f64.
    ny: Vec<f64>,
    /// Per-point radius `rho_i >= |y_i|`, inflated to cover the centring rounding.
    rho: Vec<f64>,
    rho_max: f64,
    /// Bound on |estimate - exact| for query i: coef * (rho_i + rho_max)^2.
    coef: f64,
}

impl Prepared {
    fn new(x: &RowMajor) -> Self {
        let (n, d) = (x.n_rows, x.n_cols);
        let da = d + 1;
        let mut mean = vec![0f64; d];
        for i in 0..n {
            for (m, &v) in mean.iter_mut().zip(x.row(i)) {
                *m += v as f64;
            }
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        let max_abs = x.data.iter().fold(0f64, |a, &v| a.max((v as f64).abs()));

        // Unit roundoff of f32 (2^-24). Centring error per point is at most
        // u |y_i| (f32 rounding) + sqrt(d) 2^-52 max|x| (f64 subtraction), folded into rho.
        let u = 0.5 * f32::EPSILON as f64;
        let eta = (d as f64).sqrt() * max_abs * 2f64.powi(-28);

        let mut qaug = vec![0f32; n * da];
        let mut raug = vec![0f32; n * da];
        let mut ny = vec![0f64; n];
        let mut rho = vec![0f64; n];
        qaug.par_chunks_mut(da)
            .zip(raug.par_chunks_mut(da))
            .zip(ny.par_iter_mut().zip(rho.par_iter_mut()))
            .enumerate()
            .for_each(|(i, ((q, r), (nyi, rhoi)))| {
                let xi = x.row(i);
                let mut s = 0f64;
                for t in 0..d {
                    let y = (xi[t] as f64 - mean[t]) as f32;
                    q[t] = y;
                    r[t] = -2.0 * y;
                    s += y as f64 * y as f64;
                }
                q[d] = 1.0;
                r[d] = s as f32;
                *nyi = s;
                *rhoi = s.sqrt() * (1.0 + 2.0 * u) + eta;
            });
        let rho_max = rho.iter().copied().fold(0f64, f64::max);
        // GEMM over d+1 terms: (d+1) u (|2 y_i.y_j| + |y_j|^2) <= (d+2) u (rho_i + rho_j)^2,
        // plus |y_j|^2 rounding and the centring term 2u (rho_i + rho_j)^2: in total below
        // (d+5) u (rho_i + rho_j)^2. Doubled for safety (f64 terms, second-order terms).
        let coef = 2.0 * (d as f64 + 6.0) * u;
        Self { qaug, raug, ny, rho, rho_max, coef }
    }

    #[inline]
    fn bound(&self, i: usize) -> f64 {
        let r = self.rho[i] + self.rho_max;
        self.coef * r * r + f64::MIN_POSITIVE
    }
}

/// Per-worker scratch space, reused across query blocks.
struct Workspace {
    gemm: Vec<f32>,
    /// `qb` bounded max-heaps of (estimate, index), `cap` slots each.
    heaps: Vec<(f32, u32)>,
    lens: Vec<usize>,
    cand: Vec<(f64, u32)>,
}

impl Workspace {
    fn new(qb: usize, rc: usize, cap: usize) -> Self {
        Self {
            gemm: vec![0f32; qb * rc],
            heaps: vec![(0f32, 0u32); qb * cap],
            lens: vec![0; qb],
            cand: Vec::with_capacity(cap.max(64)),
        }
    }
}

/// Offers one GEMM column (estimates of one query against a reference chunk starting at
/// `j0`) to the query's bounded max-heap. Indices arrive in increasing order, so a strict
/// `<` keeps the lower index among equal estimates.
#[inline]
fn scan(col: &[f32], j0: usize, self_idx: usize, heap: &mut [(f32, u32)], len: &mut usize) {
    let cap = heap.len();
    let mut tau = if *len == cap { heap[0].0 } else { f32::INFINITY };
    for (c, chunk) in col.chunks(SCAN_LANES).enumerate() {
        // Branch-free test of the whole chunk (vectorises); most chunks are rejected here.
        if !chunk.iter().fold(false, |hit, &s| hit | (s < tau)) {
            continue;
        }
        let base = j0 + c * SCAN_LANES;
        for (o, &s) in chunk.iter().enumerate() {
            if s >= tau {
                continue;
            }
            let j = base + o;
            if j == self_idx {
                continue;
            }
            if *len < cap {
                heap_push(&mut heap[..*len + 1], (s, j as u32));
                *len += 1;
                if *len == cap {
                    tau = heap[0].0;
                }
            } else {
                heap_replace_top(heap, (s, j as u32));
                tau = heap[0].0;
            }
        }
    }
}

/// `heap[..len-1]` is a max-heap on `.0`; places `item` at the end and sifts it up.
#[inline]
fn heap_push(heap: &mut [(f32, u32)], item: (f32, u32)) {
    let mut pos = heap.len() - 1;
    while pos > 0 {
        let parent = (pos - 1) / 2;
        if heap[parent].0 >= item.0 {
            break;
        }
        heap[pos] = heap[parent];
        pos = parent;
    }
    heap[pos] = item;
}

/// Replaces the maximum of a full max-heap with `item` and sifts it down.
#[inline]
fn heap_replace_top(heap: &mut [(f32, u32)], item: (f32, u32)) {
    let len = heap.len();
    let mut pos = 0;
    loop {
        let l = 2 * pos + 1;
        if l >= len {
            break;
        }
        let r = l + 1;
        let child = if r < len && heap[r].0 > heap[l].0 { r } else { l };
        if heap[child].0 <= item.0 {
            break;
        }
        heap[pos] = heap[child];
        pos = child;
    }
    heap[pos] = item;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_keeps_smallest() {
        let cap = 5;
        let mut heap = vec![(0f32, 0u32); cap];
        let mut len = 0;
        let col: Vec<f32> = vec![9.0, 3.0, 7.0, 1.0, 8.0, 2.0, 6.0, 0.5, 4.0, 5.0];
        scan(&col, 0, usize::MAX, &mut heap, &mut len);
        let mut got: Vec<u32> = heap[..len].iter().map(|e| e.1).collect();
        got.sort();
        assert_eq!(got, vec![1, 3, 5, 7, 8]);
        assert_eq!(heap[0].0, 4.0);
    }

    #[test]
    fn heap_skips_self_and_prefers_lower_index_on_ties() {
        let mut heap = vec![(0f32, 0u32); 2];
        let mut len = 0;
        scan(&[1.0, 1.0, 1.0, 1.0], 10, 10, &mut heap, &mut len);
        let mut got: Vec<u32> = heap[..len].iter().map(|e| e.1).collect();
        got.sort();
        assert_eq!(got, vec![11, 12]);
    }
}
