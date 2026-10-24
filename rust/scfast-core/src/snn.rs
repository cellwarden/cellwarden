//! Shared-nearest-neighbour graph: replaces the SNN part of Seurat's `FindNeighbors`
//! (`ComputeSNN`), which is the graph `FindClusters` clusters on.
//!
//! Seurat's definition: with each cell's neighbour set N(i) of size k INCLUDING itself,
//! the edge weight is the Jaccard index |N(i) & N(j)| / |N(i) | N(j)| =
//! s / (2k - s) with s = |N(i) & N(j)|, and weights below `prune` (default 1/15) are set to
//! zero. The result is symmetric and includes the diagonal (weight 1).
//!
//! # How it is computed
//!
//! Seurat forms the binary n x n kNN matrix A and takes A A^T. Here the same counts come
//! from a reverse kNN index (for each cell l, the cells that list l as a neighbour): the cells
//! sharing at least one neighbour with i are exactly the cells in `rev(l)` for `l` in N(i),
//! and the number of times j is reached that way is s = |N(i) & N(j)|. Each row is
//! therefore found from about k^2 lookups with a per-thread dense counter, rows are
//! independent (parallel with rayon), and nothing n x n is ever built.
//!
//! The output is written in two passes over the rows (count, then fill), so peak memory is
//! the final CSR plus the reverse index (n * k `u32`) and one O(n) counter per worker, never
//! a second copy of the graph. Every weight is computed with Seurat's own floating-point
//! expression, `s / (k + (k - s))` in f64, and the prune test is done in f64 before storing as
//! f32, so the nonzero pattern is identical to Seurat's for the same kNN indices.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::types::{Knn, SparseGraph};

#[derive(Clone, Debug)]
pub struct SnnParams {
    /// Seurat `prune.SNN` (default 1/15).
    pub prune: f64,
    /// Keep the diagonal (self weight 1), as Seurat's `RNA_snn` does.
    pub keep_diagonal: bool,
}

impl Default for SnnParams {
    fn default() -> Self {
        Self { prune: 1.0 / 15.0, keep_diagonal: true }
    }
}

/// Build the SNN graph from a kNN result (self included, as returned by `knn::knn`).
/// Output: full symmetric CSR (both directions stored), rows with sorted column indices.
///
/// Each row of `knn.indices` is taken as the neighbour set exactly as given, which is what
/// Seurat does with its Annoy indices (so passing Seurat's own `Indices()` reproduces
/// `RNA_snn`). `knn.distances` is not read and may be empty.
///
/// Errors: `indices.len() != n * k`, `k == 0` with cells present, an index `>= n`, or the same
/// neighbour listed twice in one row (`InvalidInput`); `prune` outside `[0, 1]`
/// (`InvalidParameter`).
pub fn snn_from_knn(knn: &Knn, params: &SnnParams) -> Result<SparseGraph> {
    if !(0.0..=1.0).contains(&params.prune) {
        return Err(Error::InvalidParameter(format!("prune must be in [0, 1], got {}", params.prune)));
    }
    validate_knn(knn)?;
    let n = knn.n;
    if n == 0 {
        return Ok(SparseGraph { n: 0, indptr: vec![0], indices: Vec::new(), weights: Vec::new() });
    }

    let rev = ReverseIndex::build(knn);
    let table = WeightTable::new(knn.k, params.prune);
    let keep_diagonal = params.keep_diagonal;
    // Rows are processed in breadth-first order of the kNN graph, so consecutive rows are
    // neighbours and reuse the same counter and reverse-index cache lines (1.29M cells:
    // pass 1 0.73 -> 0.50 s). Each row is still written to its own slot.
    let order = bfs_order(knn);
    let block = block_len(n);

    // Pass 1: number of surviving entries per row, written to indptr[i + 1].
    let mut counts = vec![0usize; n];
    counts.par_chunks_mut(block).zip(order.par_chunks(block)).for_each(|(out, rows)| {
        let mut acc = Accumulator::new(n);
        for (slot, &i) in out.iter_mut().zip(rows) {
            let i = i as usize;
            acc.accumulate(knn, &rev, i);
            *slot = acc.touched.iter().filter(|&&j| table.keeps(acc.count[j as usize], i, j, keep_diagonal)).count();
            acc.reset();
        }
    });
    let mut indptr = vec![0usize; n + 1];
    for (&i, &c) in order.iter().zip(&counts) {
        indptr[i as usize + 1] = c;
    }
    drop(counts);
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }

    // Pass 2: fill each row's columns (sorted) and weights into its slice of the output.
    let nnz = indptr[n];
    let mut indices = vec![0u32; nnz];
    let mut weights = vec![0.0f32; nnz];
    {
        let out = RowSlices { indices: indices.as_mut_ptr(), weights: weights.as_mut_ptr() };
        let (indptr, out) = (&indptr, &out);
        order.par_chunks(block).for_each(|rows| {
            let mut acc = Accumulator::new(n);
            for &i in rows {
                let i = i as usize;
                // SAFETY: row i owns indptr[i]..indptr[i + 1], inside both buffers, and each
                // row appears once in `order`, so no two writers share an element.
                let (cols, ws) = unsafe { out.row(indptr[i], indptr[i + 1]) };
                acc.accumulate(knn, &rev, i);
                let mut pos = 0;
                for &j in &acc.touched {
                    if table.keeps(acc.count[j as usize], i, j, keep_diagonal) {
                        cols[pos] = j;
                        pos += 1;
                    }
                }
                debug_assert_eq!(pos, cols.len(), "pass 1 and pass 2 disagree on row {i}");
                cols.sort_unstable();
                for (w, &j) in ws.iter_mut().zip(cols.iter()) {
                    *w = table.weight[acc.count[j as usize] as usize];
                }
                acc.reset();
            }
        });
    }

    Ok(SparseGraph { n, indptr, indices, weights })
}

/// Shape, bounds and duplicate checks, parallel over rows.
fn validate_knn(knn: &Knn) -> Result<()> {
    let (n, k) = (knn.n, knn.k);
    if n > u32::MAX as usize {
        return Err(Error::InvalidInput(format!("{n} cells exceed the u32 index range")));
    }
    let expected = n.checked_mul(k).ok_or_else(|| Error::InvalidInput(format!("n * k overflows ({n} x {k})")))?;
    if knn.indices.len() != expected {
        return Err(Error::InvalidInput(format!(
            "knn.indices has {} entries, expected n * k = {n} x {k} = {expected}",
            knn.indices.len()
        )));
    }
    if n > 0 && k == 0 {
        return Err(Error::InvalidInput("k must be at least 1 (each row is the cell's neighbour set)".into()));
    }
    (0..n).into_par_iter().try_for_each_init(
        || Vec::with_capacity(k),
        |buf: &mut Vec<u32>, i| {
            let row = &knn.indices[i * k..(i + 1) * k];
            if let Some(&bad) = row.iter().find(|&&j| j as usize >= n) {
                return Err(Error::InvalidInput(format!("knn row {i}: neighbour index {bad} out of range for {n} cells")));
            }
            buf.clear();
            buf.extend_from_slice(row);
            buf.sort_unstable();
            if let Some(w) = buf.windows(2).find(|w| w[0] == w[1]) {
                return Err(Error::InvalidInput(format!("knn row {i}: neighbour {} is listed more than once", w[0])));
            }
            Ok(())
        },
    )
}

/// For each cell l, the cells whose neighbour set contains l (CSR, rows in increasing order).
struct ReverseIndex {
    ptr: Vec<usize>,
    idx: Vec<u32>,
}

impl ReverseIndex {
    fn build(knn: &Knn) -> Self {
        let (n, k) = (knn.n, knn.k);
        let mut ptr = vec![0usize; n + 1];
        for &l in &knn.indices {
            ptr[l as usize + 1] += 1;
        }
        for l in 0..n {
            ptr[l + 1] += ptr[l];
        }
        let mut cursor = ptr.clone();
        let mut idx = vec![0u32; knn.indices.len()];
        for i in 0..n {
            for &l in &knn.indices[i * k..(i + 1) * k] {
                idx[cursor[l as usize]] = i as u32;
                cursor[l as usize] += 1;
            }
        }
        Self { ptr, idx }
    }

    #[inline]
    fn row(&self, l: usize) -> &[u32] {
        &self.idx[self.ptr[l]..self.ptr[l + 1]]
    }
}

/// Weight and prune decision for every possible overlap s = 0..=k, computed once with
/// Seurat's exact expression so the comparison against `prune` matches it bit for bit.
struct WeightTable {
    keep: Vec<bool>,
    weight: Vec<f32>,
}

impl WeightTable {
    fn new(k: usize, prune: f64) -> Self {
        let kf = k as f64;
        let (mut keep, mut weight) = (Vec::with_capacity(k + 1), Vec::with_capacity(k + 1));
        for s in 0..=k {
            let sf = s as f64;
            // ComputeSNN: value / (k + (k - value)); `value < prune` -> 0 -> removed, so
            // equality survives (w and prune are finite here, so `>=` is exactly `!(<)`)
            let w = sf / (kf + (kf - sf));
            keep.push(s > 0 && w >= prune);
            weight.push(w as f32);
        }
        Self { keep, weight }
    }

    #[inline]
    fn keeps(&self, s: u32, i: usize, j: u32, keep_diagonal: bool) -> bool {
        self.keep[s as usize] && (keep_diagonal || j as usize != i)
    }
}

/// Per-worker dense overlap counter plus the list of cells it touched, so resetting costs
/// only the row's own candidates.
struct Accumulator {
    count: Vec<u32>,
    touched: Vec<u32>,
}

impl Accumulator {
    fn new(n: usize) -> Self {
        // vec![0; n] is a zeroed allocation: pages are only materialised where touched
        Self { count: vec![0u32; n], touched: Vec::new() }
    }

    /// count[j] = |N(i) & N(j)| for every j sharing a neighbour with i.
    #[inline]
    fn accumulate(&mut self, knn: &Knn, rev: &ReverseIndex, i: usize) {
        let k = knn.k;
        for &l in &knn.indices[i * k..(i + 1) * k] {
            for &j in rev.row(l as usize) {
                let c = &mut self.count[j as usize];
                if *c == 0 {
                    self.touched.push(j);
                }
                *c += 1;
            }
        }
    }

    #[inline]
    fn reset(&mut self) {
        for &j in &self.touched {
            self.count[j as usize] = 0;
        }
        self.touched.clear();
    }
}

/// Rows per parallel block: about 8 blocks per worker thread so work-stealing can balance
/// them, and few enough that the O(n) counter allocated per block stays negligible.
fn block_len(n: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    n.div_ceil(threads * 8).max(256)
}

/// Rows in breadth-first order over the kNN graph, restarting at the lowest unvisited row.
fn bfs_order(knn: &Knn) -> Vec<u32> {
    let (n, k) = (knn.n, knn.k);
    let mut seen = vec![false; n];
    let mut order: Vec<u32> = Vec::with_capacity(n);
    for root in 0..n {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut head = order.len();
        order.push(root as u32);
        while head < order.len() {
            let v = order[head] as usize;
            head += 1;
            for &j in &knn.indices[v * k..(v + 1) * k] {
                if !std::mem::replace(&mut seen[j as usize], true) {
                    order.push(j);
                }
            }
        }
    }
    order
}

/// The output buffers, shared by the pass-2 workers, each of which writes only its rows.
struct RowSlices {
    indices: *mut u32,
    weights: *mut f32,
}

// SAFETY: workers only touch disjoint row ranges (see `RowSlices::row`).
unsafe impl Sync for RowSlices {}

impl RowSlices {
    /// # Safety
    /// `lo..hi` must lie inside both buffers and not overlap any range another thread
    /// holds at the same time.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, lo: usize, hi: usize) -> (&mut [u32], &mut [f32]) {
        unsafe {
            (
                std::slice::from_raw_parts_mut(self.indices.add(lo), hi - lo),
                std::slice::from_raw_parts_mut(self.weights.add(lo), hi - lo),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn knn(n: usize, k: usize, indices: Vec<u32>) -> Knn {
        Knn { n, k, indices, distances: Vec::new() }
    }

    #[test]
    fn weight_table_matches_seurat_prune_rule() {
        // k = 20, prune = 1/15: s / (40 - s) >= 1/15  <=>  s >= 2.5
        let t = WeightTable::new(20, 1.0 / 15.0);
        assert!(!t.keep[0] && !t.keep[1] && !t.keep[2]);
        assert!(t.keep[3]);
        assert_eq!(t.weight[20], 1.0);
        // boundary: k = 8, s = 1 gives exactly 1/15 in f64, which Seurat keeps (`<` prune)
        let t = WeightTable::new(8, 1.0 / 15.0);
        assert!(t.keep[1]);
    }

    #[test]
    fn empty_and_invalid_inputs() {
        let g = snn_from_knn(&knn(0, 5, vec![]), &SnnParams::default()).unwrap();
        assert_eq!((g.n, g.indptr.as_slice(), g.nnz()), (0, &[0usize][..], 0));
        assert!(snn_from_knn(&knn(2, 2, vec![0, 1, 1]), &SnnParams::default()).is_err());
        assert!(snn_from_knn(&knn(2, 2, vec![0, 1, 1, 2]), &SnnParams::default()).is_err());
        assert!(snn_from_knn(&knn(2, 2, vec![0, 0, 1, 0]), &SnnParams::default()).is_err());
        assert!(snn_from_knn(&knn(2, 0, vec![]), &SnnParams::default()).is_err());
        for prune in [-0.1, 1.5, f64::NAN] {
            let p = SnnParams { prune, keep_diagonal: true };
            assert!(snn_from_knn(&knn(2, 2, vec![0, 1, 1, 0]), &p).is_err(), "prune {prune}");
        }
    }

    #[test]
    fn bfs_order_is_a_permutation() {
        // two components: 0-1-2 and 3-4
        let k = knn(5, 2, vec![0, 1, 1, 2, 2, 0, 3, 4, 4, 3]);
        let order = bfs_order(&k);
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
        let k = knn(4, 2, vec![0, 3, 1, 2, 2, 1, 3, 1]);
        let mut o = bfs_order(&k);
        assert_eq!(o, vec![0, 3, 1, 2]);
        o.sort();
        assert_eq!(o, vec![0, 1, 2, 3]);
    }
}
