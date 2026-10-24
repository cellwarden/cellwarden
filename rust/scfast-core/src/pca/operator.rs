//! The implicitly scaled operator `Z = (X - 1 mu^T) diag(1/sigma)` and the one-pass gene
//! statistics it needs.
//!
//! `X` is the borrowed CSR (cells x genes). Neither `Z` nor a transposed copy of `X` is ever
//! built:
//!
//! - `Z V   = X (S V) - 1 (mu^T S V)` with `S = diag(1/sigma)`: one row-parallel pass over
//!   the CSR. Each output row depends on one input row only, so it is deterministic.
//! - `Z^T U = S (X^T U - mu (1^T U))`: also a row-parallel pass over the CSR, scattering
//!   `x_ij * U[i, :]` into a small genes x l accumulator. Row blocks are fixed by the input
//!   shape (never by the thread count) and their partial accumulators are summed by a
//!   fixed binary tree, so the floating-point summation order, and therefore the result, is
//!   identical for any number of threads.
//!
//! Why not a CSC copy for `X^T U`: a gene-parallel gather over a CSC copy needs no
//! reduction, but it costs a second copy of the HVG matrix (8 bytes per non-zero as
//! u32 + f32, 12 as i32 + f64) and reads `U[i, :]` once per non-zero from all over an
//! n x l matrix that does not fit in cache at large n. The scatter reads `U` once,
//! sequentially, and keeps the genes x l accumulator (640 KB at 2,000 HVGs and l = 40) in
//! cache. The price is one accumulator per live tree node, bounded by roughly
//! threads x log2(blocks) of them and by at most `MAX_BLOCKS` leaves.

use faer::reborrow::ReborrowMut;
use faer::{Mat, MatMut, MatRef};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::types::{CsrView, Idx, Real};

/// Upper bound on the number of row blocks for the reductions (stats and `Z^T U`).
const MAX_BLOCKS: usize = 256;
/// Lower bound on rows per reduction block.
const MIN_BLOCK_ROWS: usize = 64;
/// A leaf should do at least this many times more scatter work than it spends zeroing and
/// merging its accumulator.
const ACC_WORK_RATIO: usize = 64;
/// Rows per task for the row-parallel `Z V` product (no reduction, so any size is exact).
const ZV_ROWS: usize = 256;
/// Rows of `U` transposed into a local row-major buffer at a time inside a `Z^T U` leaf.
const ZTU_SUB_ROWS: usize = 256;

/// `acc += a * w` (equal lengths), in fixed 8-wide chunks so LLVM emits straight-line SIMD
/// without per-element bounds checks. `FMA` selects fused multiply-add (only used where the
/// CPU has it: see `use_fma`).
#[inline(always)]
fn axpy<const FMA: bool>(acc: &mut [f64], a: f64, w: &[f64]) {
    debug_assert_eq!(acc.len(), w.len());
    let mut ac = acc.chunks_exact_mut(8);
    let mut wc = w.chunks_exact(8);
    for (x, y) in (&mut ac).zip(&mut wc) {
        let x: &mut [f64; 8] = x.try_into().unwrap();
        let y: &[f64; 8] = y.try_into().unwrap();
        for t in 0..8 {
            x[t] = if FMA { a.mul_add(y[t], x[t]) } else { x[t] + a * y[t] };
        }
    }
    for (x, y) in ac.into_remainder().iter_mut().zip(wc.remainder()) {
        *x = if FMA { a.mul_add(*y, *x) } else { *x + a * y };
    }
}

/// Columns of `Z V` and `Z^T U` updated in registers at a time (60 = 3 tiles at the
/// default l). Keeping one tile of the accumulator (`Z V`) or of the row of `U` (`Z^T U`)
/// in registers across a row's non-zeros halved both products on 500k cells x 2,000 HVGs
/// (0.145 -> 0.061 s and 0.135 -> 0.084 s per product, 12-core M-series).
const TILE: usize = 20;

#[inline(always)]
fn fma_tile<const FMA: bool>(acc: &mut [f64; TILE], a: f64, w: &[f64; TILE]) {
    for t in 0..TILE {
        acc[t] = if FMA { a.mul_add(w[t], acc[t]) } else { acc[t] + a * w[t] };
    }
}

/// `buf[r, :] = sum_j x[r0 + r, j] vs[j, :] - shift` for `m` rows (buf row-major m x l).
/// Each output sums its terms in non-zero order, tile by tile.
#[inline(always)]
fn zv_kernel<I: Idx, T: Real, const FMA: bool>(
    x: &CsrView<'_, I, T>,
    vs: &[f64],
    shift: &[f64],
    r0: usize,
    buf: &mut [f64],
) {
    let l = shift.len();
    let tiled = l - l % TILE;
    for (r, out) in buf.chunks_exact_mut(l).enumerate() {
        let (idx, val) = x.row(r0 + r);
        for c0 in (0..tiled).step_by(TILE) {
            let mut acc = [0.0f64; TILE];
            for (j, xv) in idx.iter().zip(val) {
                let o = j.to_usize() * l + c0;
                fma_tile::<FMA>(&mut acc, xv.to_f64(), vs[o..o + TILE].try_into().unwrap());
            }
            out[c0..c0 + TILE].copy_from_slice(&acc);
        }
        let rest = &mut out[tiled..];
        rest.fill(0.0);
        if !rest.is_empty() {
            for (j, xv) in idx.iter().zip(val) {
                let j = j.to_usize();
                axpy::<FMA>(rest, xv.to_f64(), &vs[j * l + tiled..(j + 1) * l]);
            }
        }
        for (a, s) in out.iter_mut().zip(shift) {
            *a -= s;
        }
    }
}

/// `acc[j, :] += x[s0 + r, j] ub[r, :]` and `acc[p, :] += ub[r, :]` for the rows of `ub`.
/// Each `acc` element receives its terms in row order, tile by tile.
#[inline(always)]
fn ztu_kernel<I: Idx, T: Real, const FMA: bool>(x: &CsrView<'_, I, T>, ub: &[f64], l: usize, s0: usize, acc: &mut [f64]) {
    let p = x.n_cols;
    let tiled = l - l % TILE;
    for (r, ur) in ub.chunks_exact(l).enumerate() {
        for (a, w) in acc[p * l..].iter_mut().zip(ur) {
            *a += w;
        }
        let (idx, val) = x.row(s0 + r);
        for c0 in (0..tiled).step_by(TILE) {
            let w: &[f64; TILE] = ur[c0..c0 + TILE].try_into().unwrap();
            for (j, xv) in idx.iter().zip(val) {
                let o = j.to_usize() * l + c0;
                fma_tile::<FMA>((&mut acc[o..o + TILE]).try_into().unwrap(), xv.to_f64(), w);
            }
        }
        if tiled < l {
            for (j, xv) in idx.iter().zip(val) {
                let j = j.to_usize();
                axpy::<FMA>(&mut acc[j * l + tiled..(j + 1) * l], xv.to_f64(), &ur[tiled..]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn zv_kernel_avx2<I: Idx, T: Real>(x: &CsrView<'_, I, T>, vs: &[f64], shift: &[f64], r0: usize, buf: &mut [f64]) {
    zv_kernel::<I, T, true>(x, vs, shift, r0, buf)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ztu_kernel_avx2<I: Idx, T: Real>(x: &CsrView<'_, I, T>, ub: &[f64], l: usize, s0: usize, acc: &mut [f64]) {
    ztu_kernel::<I, T, true>(x, ub, l, s0, acc)
}

/// Whether the AVX2 + FMA kernels can run (x86_64, detected once at run time). On aarch64
/// FMA is part of the baseline, so the fused kernels are always used there. Fused and
/// unfused kernels round differently, so results are reproducible per machine, not across
/// CPU generations.
#[inline]
fn use_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn zv_rows<I: Idx, T: Real>(x: &CsrView<'_, I, T>, vs: &[f64], shift: &[f64], r0: usize, buf: &mut [f64], fma: bool) {
    #[cfg(target_arch = "x86_64")]
    if fma {
        // SAFETY: only called when use_fma() detected AVX2 and FMA.
        return unsafe { zv_kernel_avx2(x, vs, shift, r0, buf) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        let _ = fma;
        return zv_kernel::<I, T, true>(x, vs, shift, r0, buf);
    }
    #[allow(unreachable_code)]
    {
        let _ = fma;
        zv_kernel::<I, T, false>(x, vs, shift, r0, buf)
    }
}

fn ztu_rows<I: Idx, T: Real>(x: &CsrView<'_, I, T>, ub: &[f64], l: usize, s0: usize, acc: &mut [f64], fma: bool) {
    #[cfg(target_arch = "x86_64")]
    if fma {
        // SAFETY: only called when use_fma() detected AVX2 and FMA.
        return unsafe { ztu_kernel_avx2(x, ub, l, s0, acc) };
    }
    #[cfg(target_arch = "aarch64")]
    {
        let _ = fma;
        return ztu_kernel::<I, T, true>(x, ub, l, s0, acc);
    }
    #[allow(unreachable_code)]
    {
        let _ = fma;
        ztu_kernel::<I, T, false>(x, ub, l, s0, acc)
    }
}

/// Fixed partition of the rows for the reductions. Depends on the input shape only.
#[derive(Clone, Copy, Debug)]
pub(super) struct Blocks {
    pub rows: usize,
    pub count: usize,
}

impl Blocks {
    pub fn new(n: usize, p: usize, nnz: usize) -> Self {
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

/// Deterministic parallel reduction: leaves `lo..hi` combined by a binary tree whose shape
/// depends only on the leaf count, never on the scheduling.
fn tree_reduce<A, L, M>(lo: usize, hi: usize, leaf: &L, merge: &M) -> Result<A>
where
    A: Send,
    L: Fn(usize) -> Result<A> + Sync,
    M: Fn(&mut A, A) + Sync,
{
    if hi - lo == 1 {
        return leaf(lo);
    }
    let mid = lo + (hi - lo) / 2;
    let (a, b) = rayon::join(|| tree_reduce(lo, mid, leaf, merge), || tree_reduce(mid, hi, leaf, merge));
    let mut a = a?;
    merge(&mut a, b?);
    Ok(a)
}

/// Per-gene statistics for the implicit scaling.
#[derive(Clone, Debug)]
pub(super) struct GeneStats {
    pub mean: Vec<f64>,
    /// Sample standard deviation (n - 1 denominator); 0 for constant genes.
    pub sd: Vec<f64>,
    /// Exactly constant genes (all zero, or the same stored value in every cell).
    pub constant: Vec<bool>,
}

/// Per-gene accumulators of [`gene_stats`] (also filled on the device, `pca/cuda.rs`).
pub(super) struct StatsAcc {
    pub sum: Vec<f64>,
    pub sumsq: Vec<f64>,
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    pub nnz: Vec<u64>,
}

/// One parallel pass over the CSR: per-gene sum, sum of squares, min/max of stored values and
/// stored-entry counts, reduced by the fixed tree. Also validates the sparse structure, so a
/// `CsrView` built without `CsrView::new` cannot cause out-of-bounds access later.
pub(super) fn gene_stats<I: Idx, T: Real>(x: &CsrView<'_, I, T>, blocks: Blocks) -> Result<GeneStats> {
    let (n, p, nnz) = (x.n_rows, x.n_cols, x.data.len());
    let leaf = |b: usize| -> Result<StatsAcc> {
        let (r0, r1) = blocks.range(b, n);
        let mut acc = StatsAcc {
            sum: vec![0.0; p],
            sumsq: vec![0.0; p],
            min: vec![f64::INFINITY; p],
            max: vec![f64::NEG_INFINITY; p],
            nnz: vec![0; p],
        };
        // Row stamp per gene to detect duplicate column indices within a row.
        let mut seen = vec![u32::MAX; p];
        for i in r0..r1 {
            let (s, e) = (x.indptr[i].to_usize(), x.indptr[i + 1].to_usize());
            if s > e || e > nnz {
                return Err(Error::InvalidInput(format!("indptr is invalid at row {i}")));
            }
            let stamp = (i - r0) as u32;
            for (j, v) in x.indices[s..e].iter().zip(&x.data[s..e]) {
                let j = j.to_usize();
                if j >= p {
                    return Err(Error::InvalidInput(format!("column index {j} out of range for {p} columns")));
                }
                if seen[j] == stamp {
                    return Err(Error::InvalidInput(format!(
                        "duplicate column index {j} in row {i}; sum duplicates first"
                    )));
                }
                seen[j] = stamp;
                let v = v.to_f64();
                acc.sum[j] += v;
                acc.sumsq[j] += v * v;
                acc.min[j] = acc.min[j].min(v);
                acc.max[j] = acc.max[j].max(v);
                acc.nnz[j] += 1;
            }
        }
        Ok(acc)
    };
    let merge = |a: &mut StatsAcc, b: StatsAcc| {
        for j in 0..p {
            a.sum[j] += b.sum[j];
            a.sumsq[j] += b.sumsq[j];
            a.min[j] = a.min[j].min(b.min[j]);
            a.max[j] = a.max[j].max(b.max[j]);
            a.nnz[j] += b.nnz[j];
        }
    };
    tree_reduce(0, blocks.count, &leaf, &merge)?.finish(n)
}

impl StatsAcc {
    /// Mean, sample sd and the constant flag of each gene over `n` cells.
    pub(super) fn finish(self, n: usize) -> Result<GeneStats> {
        let acc = self;
        let p = acc.sum.len();
        let nf = n as f64;
        let mut mean = vec![0.0; p];
        let mut sd = vec![0.0; p];
        let mut constant = vec![false; p];
        for j in 0..p {
            if !acc.sum[j].is_finite() || !acc.sumsq[j].is_finite() {
                return Err(Error::InvalidInput(format!("non-finite values in column {j}")));
            }
            let all_stored = acc.nnz[j] as usize == n;
            let is_const = acc.nnz[j] == 0 || (acc.min[j] == acc.max[j] && (all_stored || acc.max[j] == 0.0));
            if is_const {
                constant[j] = true;
                // The exact constant, so the centred column is exactly zero.
                mean[j] = if all_stored { acc.min[j] } else { 0.0 };
                continue;
            }
            let m = acc.sum[j] / nf;
            let m2 = (acc.sumsq[j] - acc.sum[j] * m).max(0.0);
            mean[j] = m;
            sd[j] = (m2 / (nf - 1.0)).sqrt();
        }
        Ok(GeneStats { mean, sd, constant })
    }
}

/// `Z = (X - 1 mu^T) diag(inv_sd)`, applied implicitly.
pub(super) struct ScaledOp<'a, I: Idx, T: Real> {
    x: CsrView<'a, I, T>,
    mean: &'a [f64],
    /// 1/sigma for scaled genes, 1 when scaling is off, and 0 for constant genes (their
    /// centred column is zero, so this is the same operator, with exact zeros).
    inv_sd: &'a [f64],
    blocks: Blocks,
}

impl<'a, I: Idx, T: Real> ScaledOp<'a, I, T> {
    pub fn new(x: CsrView<'a, I, T>, mean: &'a [f64], inv_sd: &'a [f64], blocks: Blocks) -> Self {
        Self { x, mean, inv_sd, blocks }
    }

    pub fn nrows(&self) -> usize {
        self.x.n_rows
    }
    pub fn ncols(&self) -> usize {
        self.x.n_cols
    }

    /// `out = Z v - C v`, with `v` genes x l, `out` cells x l (column-major) and `C` the
    /// sparse `clip` entries (row-sorted). Each task subtracts its rows' entries from its
    /// row-major buffer, after `Z v`, in entry order.
    pub fn apply(&self, v: MatRef<'_, f64>, out: MatMut<'_, f64>, clip: Option<&ClipEntries>) {
        let (n, p, l) = (self.nrows(), self.ncols(), v.ncols());
        assert!(v.nrows() == p && out.nrows() == n && out.ncols() == l);
        // vs = diag(inv_sd) v, row-major so one gene's l values are contiguous.
        let mut vs = vec![0.0f64; p * l];
        for c in 0..l {
            let col = v.col(c);
            for j in 0..p {
                vs[j * l + c] = col[j] * self.inv_sd[j];
            }
        }
        // shift = mu^T diag(inv_sd) v, summed in gene order.
        let mut shift = vec![0.0f64; l];
        for j in 0..p {
            let m = self.mean[j];
            if m != 0.0 {
                for (s, w) in shift.iter_mut().zip(&vs[j * l..(j + 1) * l]) {
                    *s += m * w;
                }
            }
        }
        // v row-major, for the clip entries.
        let vr: Vec<f64> = if clip.is_some() { (0..p * l).map(|e| v[(e / l, e % l)]).collect() } else { Vec::new() };
        let x = &self.x;
        let fma = use_fma();
        out.par_row_chunks_mut(ZV_ROWS).enumerate().for_each(|(b, mut oc)| {
            let r0 = b * ZV_ROWS;
            let m = oc.nrows();
            let mut buf = vec![0.0f64; m * l];
            zv_rows(x, &vs, &shift, r0, &mut buf, fma);
            if let Some(cl) = clip {
                let (e0, e1) = (cl.rows.partition_point(|&i| (i as usize) < r0), cl.rows.partition_point(|&i| (i as usize) < r0 + m));
                for e in e0..e1 {
                    let (i, j, w) = (cl.rows[e] as usize - r0, cl.cols[e] as usize, cl.vals[e]);
                    for (o, vj) in buf[i * l..(i + 1) * l].iter_mut().zip(&vr[j * l..(j + 1) * l]) {
                        *o -= w * vj;
                    }
                }
            }
            for c in 0..l {
                let col = oc.rb_mut().col_mut(c).try_as_col_major_mut().expect("column-major").as_slice_mut();
                for (r, y) in col.iter_mut().enumerate() {
                    *y = buf[r * l + c];
                }
            }
        });
    }

    /// `Z^T u`, with `u` cells x l (column-major); returns genes x l.
    pub fn apply_t(&self, u: MatRef<'_, f64>) -> Mat<f64> {
        let (n, p, l) = (self.nrows(), self.ncols(), u.ncols());
        assert!(u.nrows() == n);
        let x = &self.x;
        let blocks = self.blocks;
        let fma = use_fma();
        // Accumulator rows 0..p hold X^T u, row p holds the column sums 1^T u.
        let leaf = |b: usize| -> Result<Vec<f64>> {
            let (r0, r1) = blocks.range(b, n);
            let mut acc = vec![0.0f64; (p + 1) * l];
            let mut ub = vec![0.0f64; ZTU_SUB_ROWS.min(r1 - r0) * l];
            let mut s0 = r0;
            while s0 < r1 {
                let s1 = (s0 + ZTU_SUB_ROWS).min(r1);
                let m = s1 - s0;
                for c in 0..l {
                    let col = u.col(c).subrows(s0, m);
                    for r in 0..m {
                        ub[r * l + c] = col[r];
                    }
                }
                ztu_rows(x, &ub[..m * l], l, s0, &mut acc, fma);
                s0 = s1;
            }
            Ok(acc)
        };
        let merge = |a: &mut Vec<f64>, b: Vec<f64>| {
            for (x, y) in a.iter_mut().zip(&b) {
                *x += y;
            }
        };
        let acc = tree_reduce(0, blocks.count, &leaf, &merge).expect("leaves are infallible");
        let colsum = &acc[p * l..];
        Mat::from_fn(p, l, |j, c| (acc[j * l + c] - self.mean[j] * colsum[c]) * self.inv_sd[j])
    }
}

impl<I: Idx, T: Real> super::Operator for ScaledOp<'_, I, T> {
    fn nrows(&self) -> usize {
        ScaledOp::nrows(self)
    }
    fn ncols(&self) -> usize {
        ScaledOp::ncols(self)
    }
    fn apply(&self, v: MatRef<'_, f64>, out: MatMut<'_, f64>) -> Result<()> {
        ScaledOp::apply(self, v, out, None);
        Ok(())
    }
    fn apply_clipped(&self, v: MatRef<'_, f64>, out: MatMut<'_, f64>, clip: &ClipEntries) -> Result<()> {
        ScaledOp::apply(self, v, out, Some(clip));
        Ok(())
    }
    fn apply_t(&self, u: MatRef<'_, f64>) -> Result<Mat<f64>> {
        Ok(ScaledOp::apply_t(self, u))
    }
}

/// `Z - C`, where `C` holds `z_ij - clip` at every entry with `z_ij > clip`: Seurat's
/// `ScaleData(scale.max = clip)` applied exactly. Only stored entries of `X` can exceed a
/// positive clip (a zero maps to `-mean / sd <= 0`), so `C` is sparse and small.
pub(super) struct Clipped<O> {
    inner: O,
    entries: ClipEntries,
}

/// The entries of `C` in row order (column order within a row).
pub(crate) struct ClipEntries {
    rows: Vec<u32>,
    cols: Vec<u32>,
    vals: Vec<f64>,
}

impl ClipEntries {
    /// `out -= C v`: one task per output column, each element receiving its terms in entry
    /// order, so the result does not depend on the thread count.
    pub(crate) fn subtract_from(&self, v: MatRef<'_, f64>, mut out: MatMut<'_, f64>) {
        let (rows, cols, vals) = (&self.rows, &self.cols, &self.vals);
        out.rb_mut().par_col_chunks_mut(1).enumerate().for_each(|(c, mut oc)| {
            let o = oc.rb_mut().col_mut(0).try_as_col_major_mut().expect("column-major").as_slice_mut();
            let vc = col(v, c);
            for ((&i, &j), &w) in rows.iter().zip(cols).zip(vals) {
                o[i as usize] -= w * vc[j as usize];
            }
        });
    }

    /// `out -= C^T u`, in the same way.
    fn subtract_t_from(&self, u: MatRef<'_, f64>, mut out: MatMut<'_, f64>) {
        let (rows, cols, vals) = (&self.rows, &self.cols, &self.vals);
        out.rb_mut().par_col_chunks_mut(1).enumerate().for_each(|(c, mut oc)| {
            let o = oc.rb_mut().col_mut(0).try_as_col_major_mut().expect("column-major").as_slice_mut();
            let uc = col(u, c);
            for ((&i, &j), &w) in rows.iter().zip(cols).zip(vals) {
                o[j as usize] -= w * uc[i as usize];
            }
        });
    }
}

impl<O: super::Operator> Clipped<O> {
    pub fn new<I: Idx, T: Real>(inner: O, x: &CsrView<'_, I, T>, mean: &[f64], inv_sd: &[f64], clip: f64) -> Self {
        assert!(x.n_rows <= u32::MAX as usize && x.n_cols <= u32::MAX as usize, "clip entries are stored as u32");
        let per_row: Vec<Vec<(u32, u32, f64)>> = (0..x.n_rows)
            .into_par_iter()
            .with_min_len(ZV_ROWS)
            .map(|i| {
                let (idx, val) = x.row(i);
                idx.iter()
                    .zip(val)
                    .filter_map(|(j, v)| {
                        let j = j.to_usize();
                        let z = (v.to_f64() - mean[j]) * inv_sd[j];
                        (z > clip).then_some((i as u32, j as u32, z - clip))
                    })
                    .collect()
            })
            .collect();
        let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
        for (i, j, c) in per_row.into_iter().flatten() {
            rows.push(i);
            cols.push(j);
            vals.push(c);
        }
        Self { inner, entries: ClipEntries { rows, cols, vals } }
    }
}

/// Column `c` of a matrix whose columns are contiguous, as a slice.
fn col<'a>(m: MatRef<'a, f64>, c: usize) -> &'a [f64] {
    m.col(c).try_as_col_major().expect("column-major").as_slice()
}

impl<O: super::Operator> super::Operator for Clipped<O> {
    fn nrows(&self) -> usize {
        self.inner.nrows()
    }
    fn ncols(&self) -> usize {
        self.inner.ncols()
    }
    fn apply(&self, v: MatRef<'_, f64>, out: MatMut<'_, f64>) -> Result<()> {
        self.inner.apply_clipped(v, out, &self.entries)
    }
    fn apply_t(&self, u: MatRef<'_, f64>) -> Result<Mat<f64>> {
        let mut out = self.inner.apply_t(u)?;
        self.entries.subtract_t_from(u, out.as_mut());
        Ok(out)
    }
}
