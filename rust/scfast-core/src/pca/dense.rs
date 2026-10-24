//! Dense steps of the block Krylov SVD, on faer 0.24 with explicit `Par::Seq` inside each
//! task so no result depends on faer's global parallelism or the rayon pool size.
//!
//! - `orthonormalize`: thin Q of a tall matrix by CholeskyQR2 (Gram matrix and products
//!   over fixed row chunks), or by TSQR when the Gram matrix is too ill-conditioned:
//!   rows are split into a fixed number of chunks (from the shape only), each chunk is
//!   QR-factored in parallel, the stacked R factors are QR-factored once more, and each
//!   chunk's Householder reflectors are applied to its slice of the second Q.
//! - `orthonormalize_against`: a new Krylov block orthonormalised against the earlier
//!   ones (block Gram-Schmidt run twice).
//! - `thin_svd`: faer's SVD of the small genes x l matrix.
//! - `gaussian`: the test matrix, from ChaCha8 seeded with the PCA seed.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::householder::{
    apply_block_householder_sequence_on_the_left_in_place_scratch,
    apply_block_householder_sequence_on_the_left_in_place_with_conj,
};
use faer::linalg::qr::no_pivoting::factor::{qr_in_place, qr_in_place_scratch, recommended_block_size};
use faer::linalg::cholesky::llt::factor::{cholesky_in_place, cholesky_in_place_scratch};
use faer::linalg::matmul::matmul;
use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors};
use faer::linalg::triangular_inverse::invert_lower_triangular;
use faer::reborrow::{Reborrow, ReborrowMut};
use faer::{Accum, Conj, Mat, MatMut, MatRef, Par};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::error::{Error, Result};

/// Target rows per TSQR chunk. Matrices with at most this many rows take one plain QR.
pub(super) const TSQR_ROWS: usize = 8192;

/// One factored chunk: Householder block coefficients and the R factor (s x l).
struct Factored {
    coeff: Mat<f64>,
    r: Mat<f64>,
}

/// Factor `a` (m x l, m >= 1) in place: Householder basis below the diagonal, R in the
/// upper triangle. Returns the block coefficients and a copy of R (min(m, l) x l).
fn factor_in_place(mut a: MatMut<'_, f64>) -> Factored {
    let (m, l) = a.shape();
    let s = m.min(l);
    let bs = recommended_block_size::<f64>(m, l);
    let mut coeff = Mat::<f64>::zeros(bs, s);
    let mut mem = MemBuffer::new(qr_in_place_scratch::<f64>(m, l, bs, Par::Seq, Default::default()));
    qr_in_place(a.rb_mut(), coeff.as_mut(), Par::Seq, MemStack::new(&mut mem), Default::default());
    let r = Mat::from_fn(s, l, |i, j| if i <= j { a[(i, j)] } else { 0.0 });
    Factored { coeff, r }
}

/// `target <- H target`, with H the Householder sequence stored in `basis` / `coeff`.
fn apply_q(basis: MatRef<'_, f64>, coeff: MatRef<'_, f64>, target: MatMut<'_, f64>) {
    let s = coeff.ncols();
    let basis = basis.subcols(0, s);
    let mut mem = MemBuffer::new(apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(
        basis.nrows(),
        coeff.nrows(),
        target.ncols(),
    ));
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        basis,
        coeff,
        Conj::No,
        target,
        Par::Seq,
        MemStack::new(&mut mem),
    );
}

/// Replace `a` (m x l, m >= l) by an orthonormal basis Q (m x l) with range(Q) ⊇ range(a).
/// Sequential: used for one chunk, the stacked R, and small genes x l matrices.
fn orthonormalize_seq(mut a: MatMut<'_, f64>) {
    let (m, l) = a.shape();
    debug_assert!(m >= l);
    let f = factor_in_place(a.rb_mut());
    let mut q = Mat::<f64>::from_fn(m, l, |i, j| if i == j { 1.0 } else { 0.0 });
    apply_q(a.rb(), f.coeff.as_ref(), q.as_mut());
    a.copy_from(&q);
}

/// In-place thin Q of a tall matrix (m x l, m >= l): CholeskyQR2 (two `cholesky_qr`
/// rounds, GEMMs over fixed row chunks), falling back to TSQR when the Gram matrix is too
/// ill-conditioned for Cholesky. Bitwise identical for any number of threads.
pub(super) fn orthonormalize(mut a: MatMut<'_, f64>) {
    if !(cholesky_qr(a.rb_mut()) && cholesky_qr(a.rb_mut())) {
        orthonormalize_chunked(a, TSQR_ROWS);
    }
}

pub(super) fn orthonormalize_chunked(mut a: MatMut<'_, f64>, chunk_rows: usize) {
    let (m, l) = a.shape();
    assert!(m >= l, "orthonormalize needs a tall matrix");
    let chunk_rows = chunk_rows.max(2 * l).max(1);
    let count = m.div_ceil(chunk_rows);
    if count <= 1 {
        orthonormalize_seq(a);
        return;
    }
    // Phase 1: factor every chunk in place (equal-sized chunks, each with at least l rows).
    let factored: Vec<Factored> = a.rb_mut().par_row_partition_mut(count).map(factor_in_place).collect();
    // Phase 2: QR of the stacked R factors.
    let total: usize = factored.iter().map(|f| f.r.nrows()).sum();
    let mut stacked = Mat::<f64>::zeros(total, l);
    let mut offsets = Vec::with_capacity(count);
    let mut off = 0;
    for f in &factored {
        stacked.as_mut().subrows_mut(off, f.r.nrows()).copy_from(&f.r);
        offsets.push(off);
        off += f.r.nrows();
    }
    orthonormalize_seq(stacked.as_mut());
    // Phase 3: Q_chunk = H_chunk [Q2_chunk; 0].
    a.par_row_partition_mut(count).zip(factored.par_iter().zip(offsets.par_iter())).for_each(|(mut ac, (f, &off))| {
        let (mc, s) = (ac.nrows(), f.r.nrows());
        let mut t = Mat::<f64>::zeros(mc, l);
        t.as_mut().subrows_mut(0, s).copy_from(stacked.as_ref().subrows(off, s));
        apply_q(ac.rb(), f.coeff.as_ref(), t.as_mut());
        ac.copy_from(&t);
    });
}

/// `a^T b` for tall `a`, `b` with the same rows: fixed row chunks in parallel, partial
/// products added in chunk order.
fn cross(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let parts: Vec<Mat<f64>> = a
        .par_row_chunks(TSQR_ROWS)
        .zip(b.par_row_chunks(TSQR_ROWS))
        .map(|(ac, bc)| {
            let mut c = Mat::<f64>::zeros(a.ncols(), b.ncols());
            matmul(c.as_mut(), Accum::Replace, ac.transpose(), bc, 1.0, Par::Seq);
            c
        })
        .collect();
    let mut c = Mat::<f64>::zeros(a.ncols(), b.ncols());
    for part in &parts {
        c += part;
    }
    c
}

/// Rows per chunk of the fused Gram-Schmidt passes: small enough that a chunk of the
/// basis (1024 x up to 240 f64, 2 MB) is still in cache for its second product.
const BCGS_ROWS: usize = 1024;

/// Replace `y` by an orthonormal basis of the part of range(`y`) orthogonal to range(`q`)
/// (`q` with orthonormal columns): block classical Gram-Schmidt run twice (BCGS2), each
/// pass projecting out `q` and orthonormalising the block by CholeskyQR. `qty` is `q^T y`
/// for the first pass, when the caller has it more cheaply than a pass over `q`.
///
/// The passes are fused so `q` is read twice: (1) `y -= q qty`, accumulating `y^T y = L0
/// L0^T` and `q^T y`; (2) `y <- y L0^{-T} - q (q^T y L0^{-T})`, accumulating its Gram
/// matrix `L1 L1^T`; (3) `y <- y L1^{-T}`. When a Gram matrix is too ill-conditioned for
/// Cholesky, Householder QR takes over that step. When `y` lies in range(`q`), the
/// result is still orthonormal and orthogonal to `q`, spanning round-off directions.
pub(super) fn orthonormalize_against(q: MatRef<'_, f64>, mut y: MatMut<'_, f64>, qty: Mat<f64>) {
    let b = y.ncols();
    let parts: Vec<(Mat<f64>, Mat<f64>)> = y
        .rb_mut()
        .par_row_chunks_mut(BCGS_ROWS)
        .zip(q.par_row_chunks(BCGS_ROWS))
        .map(|(mut yc, qc)| {
            matmul(yc.rb_mut(), Accum::Add, qc, qty.as_ref(), -1.0, Par::Seq);
            (mul_tn(yc.rb(), yc.rb()), mul_tn(qc, yc.rb()))
        })
        .collect();
    let (mut g, mut qtr) = (Mat::<f64>::zeros(b, b), Mat::<f64>::zeros(q.ncols(), b));
    for (pg, pq) in &parts {
        g += pg;
        qtr += pq;
    }
    let second = match cholesky_inverse_t(g) {
        Some(inv_t) => {
            let c = mul(qtr.as_ref(), inv_t.as_ref());
            let parts: Vec<Mat<f64>> = y
                .rb_mut()
                .par_row_chunks_mut(BCGS_ROWS)
                .zip(q.par_row_chunks(BCGS_ROWS))
                .map(|(mut yc, qc)| {
                    let mut t = mul(yc.rb(), inv_t.as_ref());
                    matmul(t.as_mut(), Accum::Add, qc, c.as_ref(), -1.0, Par::Seq);
                    yc.copy_from(&t);
                    mul_tn(t.as_ref(), t.as_ref())
                })
                .collect();
            let mut g = Mat::<f64>::zeros(b, b);
            for pg in &parts {
                g += pg;
            }
            g
        }
        None => {
            orthonormalize_chunked(y.rb_mut(), TSQR_ROWS);
            let c = cross(q, y.rb());
            y.rb_mut().par_row_chunks_mut(BCGS_ROWS).zip(q.par_row_chunks(BCGS_ROWS)).for_each(|(yc, qc)| {
                matmul(yc, Accum::Add, qc, c.as_ref(), -1.0, Par::Seq);
            });
            cross(y.rb(), y.rb())
        }
    };
    match cholesky_inverse_t(second) {
        Some(inv_t) => right_multiply(y, inv_t.as_ref()),
        None => orthonormalize_chunked(y, TSQR_ROWS),
    }
}

/// `a b`, sequential (small matrices).
fn mul(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let mut c = Mat::<f64>::zeros(a.nrows(), b.ncols());
    matmul(c.as_mut(), Accum::Replace, a, b, 1.0, Par::Seq);
    c
}

/// `a^T b`, sequential (small matrices).
pub(super) fn mul_tn(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let mut c = Mat::<f64>::zeros(a.ncols(), b.ncols());
    matmul(c.as_mut(), Accum::Replace, a.transpose(), b, 1.0, Par::Seq);
    c
}

/// `a <- a m` over fixed row chunks in parallel.
fn right_multiply(a: MatMut<'_, f64>, m: MatRef<'_, f64>) {
    a.par_row_chunks_mut(BCGS_ROWS).for_each(|mut c| {
        let t = mul(c.rb(), m);
        c.copy_from(&t);
    });
}

/// Smallest accepted `min(diag L) / max(diag L)` in `cholesky_inverse_t`: a condition
/// number up to about 1e6, so one CholeskyQR round leaves an orthogonality error near
/// `1e12 eps` that the next round removes.
const CHOL_MIN_PIVOT_RATIO: f64 = 1e-6;

/// `L^{-T}` for the Cholesky factor of a Gram matrix `g = L L^T` (so `a L^{-T}` has
/// orthonormal columns when `g = a^T a`); `None` when `g` is not safely positive definite.
fn cholesky_inverse_t(mut g: Mat<f64>) -> Option<Mat<f64>> {
    let l = g.nrows();
    let mut mem = MemBuffer::new(cholesky_in_place_scratch::<f64>(l, Par::Seq, Default::default()));
    cholesky_in_place(g.as_mut(), Default::default(), Par::Seq, MemStack::new(&mut mem), Default::default()).ok()?;
    let (lo, hi) = (0..l).fold((f64::INFINITY, 0.0f64), |(lo, hi), i| (lo.min(g[(i, i)]), hi.max(g[(i, i)])));
    if !(lo > CHOL_MIN_PIVOT_RATIO * hi) {
        return None;
    }
    let lower = Mat::from_fn(l, l, |i, j| if i >= j { g[(i, j)] } else { 0.0 });
    let mut inv = Mat::<f64>::zeros(l, l);
    invert_lower_triangular(inv.as_mut(), lower.as_ref(), Par::Seq);
    Some(inv.transpose().to_owned())
}

/// One CholeskyQR round, `a <- a L^{-T}` with `a^T a = L L^T`; `false`, with `a`
/// unchanged, when `a^T a` is not safely positive definite.
fn cholesky_qr(a: MatMut<'_, f64>) -> bool {
    match cholesky_inverse_t(cross(a.rb(), a.rb())) {
        Some(inv_t) => {
            right_multiply(a, inv_t.as_ref());
            true
        }
        None => false,
    }
}

/// Thin SVD `a = U diag(s) V^T` (a: m x n), singular values in non-increasing order.
pub(super) fn thin_svd(a: MatRef<'_, f64>) -> Result<(Mat<f64>, Vec<f64>, Mat<f64>)> {
    let (m, n) = a.shape();
    let k = m.min(n);
    let mut u = Mat::<f64>::zeros(m, k);
    let mut v = Mat::<f64>::zeros(n, k);
    let mut s = faer::diag::Diag::<f64>::zeros(k);
    let mut mem = MemBuffer::new(svd_scratch::<f64>(
        m,
        n,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        Par::Seq,
        Default::default(),
    ));
    svd(a, s.as_mut(), Some(u.as_mut()), Some(v.as_mut()), Par::Seq, MemStack::new(&mut mem), Default::default())
        .map_err(|e| Error::Backend { context: "faer thin SVD", message: format!("{e:?}") })?;
    let s = (0..k).map(|i| s[i]).collect();
    Ok((u, s, v))
}

/// Standard normal m x l test matrix from ChaCha8 seeded with `seed` (Box-Muller), filled
/// column by column, so adding oversampling columns leaves the first ones unchanged.
pub(super) fn gaussian(m: usize, l: usize, seed: u64) -> Mat<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut values = Vec::with_capacity(m * l + 1);
    while values.len() < m * l {
        let u1: f64 = 1.0 - rng.random::<f64>(); // (0, 1]
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let (sin, cos) = (std::f64::consts::TAU * u2).sin_cos();
        values.push(r * cos);
        values.push(r * sin);
    }
    Mat::from_fn(m, l, |i, j| values[j * m + i])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(m: usize, l: usize, seed: u64) -> Mat<f64> {
        gaussian(m, l, seed)
    }

    fn check_q(q: MatRef<'_, f64>, a: MatRef<'_, f64>) {
        let l = q.ncols();
        let qtq = q.transpose() * q;
        for i in 0..l {
            for j in 0..l {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((qtq[(i, j)] - want).abs() < 1e-12, "Q^T Q ({i},{j}) = {}", qtq[(i, j)]);
            }
        }
        // a lies in range(Q): a - Q Q^T a = 0.
        let resid = a - q * (q.transpose() * a);
        let scale = a.norm_l2();
        assert!(resid.norm_l2() <= 1e-12 * scale, "residual {}", resid.norm_l2());
    }

    #[test]
    fn tsqr_matches_range_and_is_orthonormal() {
        for &(m, l, chunk) in &[(50, 7, 10_000), (1000, 13, 100), (1001, 40, 81), (333, 33, 66)] {
            let a = random(m, l, 7);
            let mut q = a.clone();
            orthonormalize_chunked(q.as_mut(), chunk);
            check_q(q.as_ref(), a.as_ref());
        }
    }

    #[test]
    fn cholesky_qr2_and_fallback() {
        for &(m, l) in &[(50_000, 40), (3000, 7), (40, 40)] {
            let a = random(m, l, 5);
            let mut q = a.clone();
            orthonormalize(q.as_mut());
            check_q(q.as_ref(), a.as_ref());
        }
        // Rank-deficient and badly scaled inputs take the TSQR fallback.
        let base = random(20_000, 3, 1);
        let a = Mat::from_fn(20_000, 9, |i, j| base[(i, j % 3)] * (1.0 + j as f64));
        let mut q = a.clone();
        orthonormalize(q.as_mut());
        check_q(q.as_ref(), a.as_ref());
        let b = Mat::from_fn(20_000, 6, |i, j| base[(i, j % 3)] * 10f64.powi(-3 * j as i32) + 1e-9 * a[(i, j)]);
        let mut q = b.clone();
        orthonormalize(q.as_mut());
        check_q(q.as_ref(), b.as_ref());
    }

    #[test]
    fn bcgs2_extends_an_orthonormal_basis() {
        let mut q = random(30_000, 80, 2);
        orthonormalize(q.as_mut());
        // A block mostly inside range(q), with one column exactly inside it.
        let r = random(30_000, 20, 3);
        let mut y = Mat::from_fn(30_000, 20, |i, j| (0..80).map(|t| q[(i, t)] * ((t + j) % 7) as f64).sum::<f64>() + if j == 0 { 0.0 } else { 1e-7 * r[(i, j)] });
        let qty = cross(q.as_ref(), y.as_ref());
        orthonormalize_against(q.as_ref(), y.as_mut(), qty);
        let mut full = Mat::<f64>::zeros(30_000, 100);
        full.as_mut().subcols_mut(0, 80).copy_from(&q);
        full.as_mut().subcols_mut(80, 20).copy_from(&y);
        let both = full.clone();
        check_q(full.as_ref(), both.as_ref());
        // range(y) holds the new directions of the perturbation.
        let rest = r.as_ref().subcols(1, 19);
        let resid = rest - &full * (full.transpose() * rest);
        assert!(resid.norm_l2() < 1e-6 * rest.norm_l2());
    }

    #[test]
    fn tsqr_rank_deficient() {
        // 3 independent columns repeated: Q is still orthonormal and spans a.
        let base = random(500, 3, 1);
        let a = Mat::from_fn(500, 9, |i, j| base[(i, j % 3)] * (1.0 + j as f64));
        let mut q = a.clone();
        orthonormalize_chunked(q.as_mut(), 64);
        check_q(q.as_ref(), a.as_ref());
    }

    #[test]
    fn tsqr_is_thread_count_independent() {
        let a = random(5000, 20, 3);
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut q = a.clone();
                orthonormalize_chunked(q.as_mut(), 300);
                q
            })
        };
        let q1 = run(1);
        for t in [2, 3, 4] {
            assert!(q1 == run(t), "TSQR differs between 1 and {t} threads");
        }
    }

    #[test]
    fn gaussian_is_seeded_and_standard() {
        let a = gaussian(400, 50, 0);
        assert!(a == gaussian(400, 50, 0));
        assert!(a != gaussian(400, 50, 1));
        // First columns do not depend on the number of columns.
        let b = gaussian(400, 60, 0);
        assert!(a.as_ref() == b.as_ref().subcols(0, 50));
        let n = (400 * 50) as f64;
        let mean: f64 = (0..50).map(|j| a.col(j).iter().sum::<f64>()).sum::<f64>() / n;
        let var: f64 = (0..50).map(|j| a.col(j).iter().map(|x| x * x).sum::<f64>()).sum::<f64>() / n;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }
}
