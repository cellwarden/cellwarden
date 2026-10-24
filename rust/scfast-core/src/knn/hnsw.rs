//! Approximate kNN with hnsw_rs 0.3 (method and determinism notes: see the parent module).

use super::{brute_one, cmp_neighbour, exact_d2, write_row};
use crate::types::{Knn, RowMajor};
use hnsw_rs::anndists::dist::distances::Distance;
use hnsw_rs::hnsw::Hnsw;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// hnsw_rs caps the number of layers at 16.
const MAX_LAYERS: usize = 16;
/// Points inserted one at a time before the parallel insertion, so the graph has a
/// connected core and an entry point before threads start racing on it.
const SERIAL_PREFIX: usize = 128;

/// Squared Euclidean distance: a monotone transform of L2, so HNSW's comparisons (search
/// and the neighbour-selection heuristic) are unchanged and no square root is taken.
#[derive(Clone, Copy, Default)]
struct SqL2;

impl Distance<f32> for SqL2 {
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        // Eight independent accumulators so the loop vectorises.
        let mut acc = [0f32; 8];
        let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
        let (ra, rb) = (ca.remainder(), cb.remainder());
        for (u, v) in ca.zip(cb) {
            for l in 0..8 {
                let t = u[l] - v[l];
                acc[l] += t * t;
            }
        }
        let mut s = acc.iter().sum::<f32>();
        for (u, v) in ra.iter().zip(rb) {
            let t = u - v;
            s += t * t;
        }
        s
    }
}

/// Approximate kNN plus the number of cells whose HNSW answer was too short and needed the
/// direct-scan fallback.
pub(super) fn hnsw_knn_with_stats(
    x: &RowMajor,
    k: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    seed: u64,
) -> (Knn, usize) {
    let n = x.n_rows;
    let kk = k - 1;

    let mut index: Hnsw<'static, f32, SqL2> = Hnsw::new(m, n, MAX_LAYERS, ef_construction, SqL2);
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    let items: Vec<(&[f32], usize)> = order.iter().map(|&i| (x.row(i), i)).collect();
    let split = SERIAL_PREFIX.min(n);
    for &item in &items[..split] {
        index.insert_slice(item);
    }
    index.parallel_insert_slice(&items[split..].to_vec());
    index.set_searching_mode(true);

    // Ask for one extra so k - 1 others remain when self is among the results.
    let knbn = (k + 1).min(n);
    let ef = ef_search.max(knbn);
    let mut indices = vec![0u32; n * k];
    let mut distances = vec![0f32; n * k];
    let index = &index;
    let fallbacks: usize = indices
        .par_chunks_mut(k)
        .zip(distances.par_chunks_mut(k))
        .enumerate()
        .map_init(
            || Vec::with_capacity(knbn),
            |cand: &mut Vec<(f64, u32)>, (i, (irow, drow))| {
                let xi = x.row(i);
                cand.clear();
                for nb in index.search(xi, knbn, ef) {
                    let j = nb.d_id;
                    if j != i && j < n {
                        cand.push((exact_d2(xi, x.row(j)), j as u32));
                    }
                }
                // Exact distances are deterministic, so duplicates are adjacent after sorting.
                cand.sort_unstable_by(cmp_neighbour);
                cand.dedup_by(|a, b| a.1 == b.1);
                let short = cand.len() < kk;
                if short {
                    brute_one(x, i, kk, cand);
                }
                write_row(i, cand, irow, drow);
                short as usize
            },
        )
        .sum();
    (Knn { n, k, indices, distances }, fallbacks)
}

pub(super) fn hnsw_knn(x: &RowMajor, k: usize, m: usize, ef_construction: usize, ef_search: usize, seed: u64) -> Knn {
    hnsw_knn_with_stats(x, k, m, ef_construction, ef_search, seed).0
}
