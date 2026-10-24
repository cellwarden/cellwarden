//! Tests for `pca::implicit_scaled_pca` (docs/RUST_ENGINE.md section 7).
//!
//! - exactness against a dense reference (Z built explicitly, faer SVD)
//! - determinism for a seed and across thread counts
//! - parameter and input validation
//! - zero-variance genes
//! - parity with Seurat 5.0.1 on tests/fixtures/pbmc_like (fixture in tests/data/)
//! - an ignored timing harness: `SCFAST_PCA_BENCH=<dir> cargo test --release -p scfast-core
//!   --test pca -- --ignored --nocapture bench_from_env`

use faer::Mat;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use scfast_core::pca::{implicit_scaled_pca, PcaParams, PcaResult};
use scfast_core::{CsrView, Error};

// ---------------------------------------------------------------- helpers

/// Owned CSR (cells x genes) for building test inputs.
#[derive(Clone, Debug)]
struct Csr {
    n: usize,
    p: usize,
    indptr: Vec<u32>,
    indices: Vec<u32>,
    data: Vec<f64>,
}

impl Csr {
    fn view(&self) -> CsrView<'_, u32, f64> {
        CsrView::new(self.n, self.p, &self.indptr, &self.indices, &self.data).unwrap()
    }

    fn from_dense(d: &Mat<f64>) -> Self {
        let (n, p) = d.shape();
        let mut indptr = vec![0u32];
        let (mut indices, mut data) = (vec![], vec![]);
        for i in 0..n {
            for j in 0..p {
                if d[(i, j)] != 0.0 {
                    indices.push(j as u32);
                    data.push(d[(i, j)]);
                }
            }
            indptr.push(indices.len() as u32);
        }
        Self { n, p, indptr, indices, data }
    }

    fn to_dense(&self) -> Mat<f64> {
        let mut d = Mat::<f64>::zeros(self.n, self.p);
        for i in 0..self.n {
            for t in self.indptr[i] as usize..self.indptr[i + 1] as usize {
                d[(i, self.indices[t] as usize)] += self.data[t];
            }
        }
        d
    }
}

fn normal(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = 1.0 - rng.random::<f64>();
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Number of latent gene programs planted by `structured`.
const PROGRAMS: usize = 10;

/// Log-normalised-looking sparse data: `groups` cell types with their own marker genes
/// (decreasing marker strength), plus `PROGRAMS` continuous gene programs with
/// geometrically decreasing strength, on noisy background genes. Gives a decaying run of
/// leading components (groups - 1 + PROGRAMS of them) above a flat noise bulk, like real data.
fn structured(n: usize, p: usize, groups: usize, seed: u64) -> Csr {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let base: Vec<f64> = (0..p).map(|_| rng.random_range(0.05..0.45)).collect();
    let markers_per_group = (p / (3 * groups)).max(1);
    let program_of: Vec<usize> = (0..p).map(|j| j % PROGRAMS).collect();
    let mut d = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        let g = i % groups;
        let level_g = 1.5 + 2.0 * 0.8f64.powi(g as i32);
        let scores: Vec<f64> = (0..PROGRAMS).map(|f| normal(&mut rng) * 0.9 * 0.8f64.powi(f as i32)).collect();
        for j in 0..p {
            let is_marker = j / markers_per_group == g && j < groups * markers_per_group;
            let s = scores[program_of[j]];
            let prob = ((if is_marker { 3.0 } else { 1.0 }) * base[j] * s.exp()).min(0.95);
            if rng.random::<f64>() < prob {
                let level = if is_marker { level_g } else { 1.0 };
                d[(i, j)] = level * (0.5 + rng.random::<f64>()) * (0.5 * s).exp();
            }
        }
    }
    Csr::from_dense(&d)
}

/// Two-pass reference mean and sample sd (n - 1).
fn reference_stats(x: &Mat<f64>) -> (Vec<f64>, Vec<f64>) {
    let (n, p) = x.shape();
    let mean: Vec<f64> = (0..p).map(|j| (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64).collect();
    let sd: Vec<f64> = (0..p)
        .map(|j| ((0..n).map(|i| (x[(i, j)] - mean[j]).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt())
        .collect();
    (mean, sd)
}

/// Dense Z = (X - 1 mu^T) diag(1/sd), sd = 1 for constant genes (or all genes if !scale).
fn dense_z(x: &Mat<f64>, scale: bool) -> Mat<f64> {
    let (mean, sd) = reference_stats(x);
    Mat::from_fn(x.nrows(), x.ncols(), |i, j| {
        let s = if !scale || sd[j] == 0.0 { 1.0 } else { sd[j] };
        (x[(i, j)] - mean[j]) / s
    })
}

struct Exact {
    /// U diag(d), n x r
    emb: Mat<f64>,
    /// V, p x r
    load: Mat<f64>,
    d: Vec<f64>,
}

fn exact_pca(z: &Mat<f64>) -> Exact {
    let svd = z.thin_svd().unwrap();
    let r = svd.S().dim();
    let d: Vec<f64> = (0..r).map(|i| svd.S()[i]).collect();
    // Same sign convention as scfast: largest-|loading| positive.
    let mut load = svd.V().to_owned();
    let mut emb = Mat::from_fn(z.nrows(), r, |i, c| svd.U()[(i, c)] * d[c]);
    for c in 0..r {
        let col = load.col(c);
        let mut best = 0;
        for j in 1..col.nrows() {
            if col[j].abs() > col[best].abs() {
                best = j;
            }
        }
        if col[best] < 0.0 {
            for j in 0..load.nrows() {
                load[(j, c)] = -load[(j, c)];
            }
            for i in 0..emb.nrows() {
                emb[(i, c)] = -emb[(i, c)];
            }
        }
    }
    Exact { emb, load, d }
}

fn mat(r: &scfast_core::RowMajor) -> Mat<f64> {
    Mat::from_fn(r.n_rows, r.n_cols, |i, j| r.data[i * r.n_cols + j] as f64)
}

/// Mean cosine of the principal angles between the column spaces of the first `k` columns.
fn subspace_similarity(a: &Mat<f64>, b: &Mat<f64>, k: usize) -> f64 {
    let qa = a.as_ref().subcols(0, k).qr().compute_thin_Q();
    let qb = b.as_ref().subcols(0, k).qr().compute_thin_Q();
    let m = qa.transpose() * &qb;
    let s = m.thin_svd().unwrap();
    (0..k).map(|i| s.S()[i].min(1.0)).sum::<f64>() / k as f64
}

/// |Pearson correlation| per column.
fn column_correlations(a: &Mat<f64>, b: &Mat<f64>, k: usize) -> Vec<f64> {
    (0..k)
        .map(|c| {
            let n = a.nrows() as f64;
            let (ma, mb) = (a.col(c).sum() / n, b.col(c).sum() / n);
            let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
            for i in 0..a.nrows() {
                let (x, y) = (a[(i, c)] - ma, b[(i, c)] - mb);
                sab += x * y;
                saa += x * x;
                sbb += y * y;
            }
            (sab / (saa * sbb).sqrt()).abs()
        })
        .collect()
}

fn singular_values(r: &PcaResult, n: usize) -> Vec<f64> {
    r.stdev.iter().map(|&s| s as f64 * ((n - 1) as f64).sqrt()).collect()
}

fn params(k: usize) -> PcaParams {
    PcaParams { n_comps: k, ..PcaParams::default() }
}

fn assert_identical(a: &PcaResult, b: &PcaResult, what: &str) {
    assert!(a.embeddings == b.embeddings, "{what}: embeddings differ");
    assert!(a.loadings == b.loadings, "{what}: loadings differ");
    assert!(a.stdev == b.stdev, "{what}: stdev differs");
    assert!(a.gene_mean == b.gene_mean && a.gene_sd == b.gene_sd, "{what}: gene stats differ");
}

// ---------------------------------------------------------------- exactness

#[test]
fn matches_exact_svd() {
    // High-accuracy settings: every requested component converges to f64 precision.
    for &(n, p, seed) in &[(400usize, 150usize, 1u64), (257, 300, 2), (1500, 90, 3)] {
        let x = structured(n, p, 4, seed);
        let ex = exact_pca(&dense_z(&x.to_dense(), true));
        let k = 10;
        let pr = PcaParams { n_comps: k, oversample: 30, power_iters: 20, clip: None, ..PcaParams::default() };
        let r = implicit_scaled_pca(x.view(), &pr).unwrap();
        let d = singular_values(&r, n);
        for c in 0..k {
            let rel = (d[c] - ex.d[c]).abs() / ex.d[c];
            assert!(rel < 1e-6, "n={n} p={p}: sigma_{c} {} vs exact {} (rel {rel:e})", d[c], ex.d[c]);
        }
        let z = dense_z(&x.to_dense(), true);
        let ex = exact_pca(&Mat::from_fn(z.nrows(), z.ncols(), |i, j| z[(i, j)].min(1.5)));
        let r = implicit_scaled_pca(x.view(), &PcaParams { clip: Some(1.5), ..pr }).unwrap();
        let d = singular_values(&r, n);
        for c in 0..k {
            let rel = (d[c] - ex.d[c]).abs() / ex.d[c];
            assert!(rel < 1e-6, "clipped n={n} p={p}: sigma_{c} {} vs exact {} (rel {rel:e})", d[c], ex.d[c]);
        }
        let (emb, load) = (mat(&r.embeddings), mat(&r.loadings));
        let sim = subspace_similarity(&emb, &ex.emb, k);
        assert!(sim > 0.9999, "n={n} p={p}: subspace similarity {sim}");
        // With the shared sign convention, loadings and embeddings match column by column.
        for c in 0..k {
            for j in 0..p {
                assert!((load[(j, c)] - ex.load[(j, c)]).abs() < 1e-4, "loading ({j},{c})");
            }
            for i in 0..n {
                let tol = 1e-4 * ex.d[c];
                assert!((emb[(i, c)] - ex.emb[(i, c)]).abs() < tol, "embedding ({i},{c})");
            }
        }
    }
}

#[test]
fn light_and_default_settings_accuracy() {
    let (n, p) = (800, 200);
    let x = structured(n, p, 5, 11);
    let ex = exact_pca(&dense_z(&x.to_dense(), true));
    let rel = |r: &PcaResult| -> Vec<f64> {
        singular_values(r, n).iter().zip(&ex.d).map(|(a, b)| (a - b).abs() / b).collect()
    };
    // Light settings (oversample 10, 4 power iterations): the leading components (well
    // above the noise bulk) are exact; components inside the flat bulk converge slowly and
    // are slightly underestimated.
    let light = implicit_scaled_pca(x.view(), &PcaParams { oversample: 10, power_iters: 4, ..params(30) }).unwrap();
    let r_light = rel(&light);
    for c in 0..30 {
        let tol = if c < 5 { 1e-6 } else if c < 7 { 1e-5 } else { 5e-2 };
        assert!(r_light[c] < tol, "light: sigma_{c} rel error {:e}", r_light[c]);
    }
    // Defaults: tighter everywhere.
    let def = implicit_scaled_pca(x.view(), &params(30)).unwrap();
    let r_def = rel(&def);
    println!("relative sigma errors, light: {:?}", r_light.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>());
    println!("relative sigma errors, default: {:?}", r_def.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>());
    for c in 0..30 {
        let tol = if c < 8 { 1e-6 } else { 1e-2 };
        assert!(r_def[c] < tol, "default: sigma_{c} rel error {:e}", r_def[c]);
    }
    let emb = mat(&def.embeddings);
    assert!(subspace_similarity(&emb, &ex.emb, 8) > 0.9999);
    let s10 = subspace_similarity(&emb, &ex.emb, 10);
    assert!(s10 > 0.999, "top-10 similarity {s10}");
}

#[test]
fn output_conventions() {
    let (n, p, k) = (300, 120, 12);
    let x = structured(n, p, 3, 5);
    let r = implicit_scaled_pca(x.view(), &params(k)).unwrap();
    assert_eq!((r.embeddings.n_rows, r.embeddings.n_cols), (n, k));
    assert_eq!((r.loadings.n_rows, r.loadings.n_cols), (p, k));
    assert_eq!((r.stdev.len(), r.gene_mean.len(), r.gene_sd.len()), (k, p, p));
    // Decreasing stdev.
    assert!(r.stdev.windows(2).all(|w| w[0] >= w[1]));
    let (emb, load) = (mat(&r.embeddings), mat(&r.loadings));
    // Loadings orthonormal; embeddings orthogonal with squared norms d^2 = (n-1) stdev^2;
    // embeddings centred (Z is centred).
    let vtv = load.transpose() * &load;
    let ete = emb.transpose() * &emb;
    for a in 0..k {
        for b in 0..k {
            let want = if a == b { 1.0 } else { 0.0 };
            assert!((vtv[(a, b)] - want).abs() < 1e-5);
            let d2 = (r.stdev[a] as f64).powi(2) * (n - 1) as f64;
            let want = if a == b { d2 } else { 0.0 };
            assert!((ete[(a, b)] - want).abs() < 1e-4 * d2.max(1.0), "E^T E ({a},{b})");
        }
        assert!(emb.col(a).sum().abs() < 1e-3 * (r.stdev[a] as f64) * (n as f64).sqrt());
        // Sign convention.
        let col: Vec<f32> = (0..p).map(|j| r.loadings.data[j * k + a]).collect();
        let best = col.iter().copied().fold(0f32, |m, v| if v.abs() > m.abs() { v } else { m });
        assert!(best > 0.0, "largest-|loading| of component {a} is negative");
    }
}

#[test]
fn gene_stats_match_direct_computation() {
    let x = structured(500, 80, 4, 21);
    let (mean, sd) = reference_stats(&x.to_dense());
    let r = implicit_scaled_pca(x.view(), &params(5)).unwrap();
    for j in 0..80 {
        assert!(((r.gene_mean[j] as f64) - mean[j]).abs() <= 1e-6 * mean[j].abs().max(1.0));
        assert!(((r.gene_sd[j] as f64) - sd[j]).abs() <= 1e-6 * sd[j]);
    }
}

#[test]
fn scale_false_is_centred_pca() {
    let (n, p, k) = (350, 100, 8);
    let x = structured(n, p, 4, 8);
    let ex = exact_pca(&dense_z(&x.to_dense(), false));
    let pr = PcaParams { n_comps: k, scale: false, oversample: 20, power_iters: 12, clip: None, ..PcaParams::default() };
    let r = implicit_scaled_pca(x.view(), &pr).unwrap();
    assert!(r.gene_sd.iter().all(|&s| s == 1.0));
    let d = singular_values(&r, n);
    for c in 0..k {
        assert!((d[c] - ex.d[c]).abs() / ex.d[c] < 1e-6);
    }
    assert!(subspace_similarity(&mat(&r.embeddings), &ex.emb, k) > 0.9999);
}

#[test]
fn largest_possible_rank_deficient_problems() {
    // l = n_comps + oversample is capped at min(n, p); with l = n the range finder sees a
    // rank-deficient (centred) matrix. Everything requested must still be exact.
    for &(n, p, k) in &[(60usize, 40usize, 39usize), (41, 60, 40), (25, 25, 24)] {
        let x = structured(n, p, 3, 99);
        let ex = exact_pca(&dense_z(&x.to_dense(), true));
        let r = implicit_scaled_pca(x.view(), &params(k)).unwrap();
        let d = singular_values(&r, n);
        let d0 = ex.d[0];
        for c in 0..k {
            assert!((d[c] - ex.d[c]).abs() < 1e-5 * d0, "n={n} p={p}: sigma_{c} {} vs {}", d[c], ex.d[c]);
        }
        assert!(r.embeddings.data.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn generic_index_and_value_types() {
    // Values exactly representable in f32: f32 and f64 inputs give identical results.
    let x = structured(300, 70, 3, 4);
    let data32: Vec<f32> = x.data.iter().map(|&v| v as f32).collect();
    let data64: Vec<f64> = data32.iter().map(|&v| v as f64).collect();
    let ip64: Vec<i64> = x.indptr.iter().map(|&v| v as i64).collect();
    let ix64: Vec<i64> = x.indices.iter().map(|&v| v as i64).collect();
    let ip32: Vec<i32> = x.indptr.iter().map(|&v| v as i32).collect();
    let ix32: Vec<i32> = x.indices.iter().map(|&v| v as i32).collect();
    let a = implicit_scaled_pca(CsrView::new(300, 70, &ip32, &ix32, &data32).unwrap(), &params(6)).unwrap();
    let b = implicit_scaled_pca(CsrView::new(300, 70, &ip64, &ix64, &data64).unwrap(), &params(6)).unwrap();
    assert_identical(&a, &b, "i32/f32 vs i64/f64");
}

// ---------------------------------------------------------------- determinism

#[test]
fn deterministic_for_a_seed() {
    // 300 genes: the Krylov basis (width 210) does not span every gene, so the seed matters.
    let x = structured(700, 300, 4, 31);
    let a = implicit_scaled_pca(x.view(), &params(20)).unwrap();
    let b = implicit_scaled_pca(x.view(), &params(20)).unwrap();
    assert_identical(&a, &b, "same seed");
    let c = implicit_scaled_pca(x.view(), &PcaParams { seed: 1, ..params(20) }).unwrap();
    assert!(a.embeddings != c.embeddings, "a different seed should change the low-order bits");
    assert!(subspace_similarity(&mat(&a.embeddings), &mat(&c.embeddings), 4) > 0.9999);
}

#[test]
fn identical_across_thread_counts() {
    // Large enough for many reduction blocks, several TSQR chunks (> 8192 rows) and
    // several output chunks.
    let x = structured(20_000, 60, 5, 41);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| implicit_scaled_pca(x.view(), &params(15)).unwrap())
    };
    let one = run(1);
    for t in [2, 3] {
        assert_identical(&one, &run(t), &format!("1 vs {t} threads"));
    }
}

// ---------------------------------------------------------------- validation

fn expect_param_err(r: scfast_core::Result<PcaResult>, what: &str) {
    match r {
        Err(Error::InvalidParameter(_)) => {}
        other => panic!("{what}: expected InvalidParameter, got {:?}", other.map(|_| ())),
    }
}

fn expect_input_err(r: scfast_core::Result<PcaResult>, what: &str) {
    match r {
        Err(Error::InvalidInput(_)) => {}
        other => panic!("{what}: expected InvalidInput, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn rejects_invalid_parameters() {
    let x = structured(50, 20, 2, 1);
    expect_param_err(implicit_scaled_pca(x.view(), &params(0)), "n_comps = 0");
    expect_param_err(implicit_scaled_pca(x.view(), &params(20)), "n_comps = n_genes");
    expect_param_err(implicit_scaled_pca(x.view(), &params(25)), "n_comps > n_genes");
    let wide = structured(15, 40, 2, 1);
    expect_param_err(implicit_scaled_pca(wide.view(), &params(15)), "n_comps = n_cells");
    for clip in [0.0, -1.0, f64::NAN] {
        expect_param_err(implicit_scaled_pca(x.view(), &PcaParams { clip: Some(clip), ..params(5) }), "bad clip");
    }
    // Accepted edge values.
    assert!(implicit_scaled_pca(x.view(), &params(19)).is_ok());
    assert!(implicit_scaled_pca(x.view(), &PcaParams { clip: None, ..params(5) }).is_ok());
    assert!(implicit_scaled_pca(x.view(), &PcaParams { clip: Some(f64::INFINITY), ..params(5) }).is_ok());
    assert!(implicit_scaled_pca(x.view(), &PcaParams { oversample: 0, power_iters: 0, ..params(5) }).is_ok());
    let empty = Csr { n: 0, p: 0, indptr: vec![0], indices: vec![], data: vec![] };
    expect_param_err(implicit_scaled_pca(empty.view(), &params(1)), "empty matrix");
}

#[test]
fn rejects_invalid_input() {
    let x = structured(50, 20, 2, 1);
    // Views built directly (bypassing CsrView::new) are still checked.
    let raw = |indptr: &[u32], indices: &[u32], data: &[f64]| -> scfast_core::Result<PcaResult> {
        let n = indptr.len().saturating_sub(1);
        let v = CsrView { n_rows: n, n_cols: 20, indptr, indices, data };
        implicit_scaled_pca(v, &params(3))
    };
    let mut bad_idx = x.indices.clone();
    bad_idx[7] = 20;
    expect_input_err(raw(&x.indptr, &bad_idx, &x.data), "column index out of range");
    let mut dup = x.indices.clone();
    let s = x.indptr[3] as usize;
    dup[s + 1] = dup[s];
    expect_input_err(raw(&x.indptr, &dup, &x.data), "duplicate column index");
    let mut bad_ptr = x.indptr.clone();
    bad_ptr.swap(10, 11);
    expect_input_err(raw(&bad_ptr, &x.indices, &x.data), "decreasing indptr");
    expect_input_err(raw(&x.indptr[..40], &x.indices, &x.data), "indptr not ending at nnz");
    let short = &x.data[..x.data.len() - 1];
    let v = CsrView { n_rows: 50, n_cols: 20, indptr: &x.indptr, indices: &x.indices, data: short };
    expect_input_err(implicit_scaled_pca(v, &params(3)), "indices/data length mismatch");
    for bad in [f64::NAN, f64::INFINITY] {
        let mut data = x.data.clone();
        data[5] = bad;
        expect_input_err(raw(&x.indptr, &x.indices, &data), "non-finite value");
    }
}

// ---------------------------------------------------------------- zero variance

#[test]
fn zero_variance_genes() {
    let (n, p) = (400, 60);
    let base = structured(n, p, 4, 77).to_dense();
    // Insert an all-zero gene at 0, a constant non-zero gene at 10 (1.7 in every cell; not
    // exactly representable, so a naive variance would be rounding noise) and one with a
    // single non-zero cell at the end (variable: must be kept).
    let extra = [(0usize, 0.0f64), (10, 1.7)];
    let mut d = Mat::<f64>::zeros(n, p + 3);
    let mut src = 0;
    for j in 0..p + 2 {
        if let Some(&(_, v)) = extra.iter().find(|e| e.0 == j) {
            for i in 0..n {
                d[(i, j)] = v;
            }
        } else {
            for i in 0..n {
                d[(i, j)] = base[(i, src)];
            }
            src += 1;
        }
    }
    d[(123, p + 2)] = 2.0;
    let x = Csr::from_dense(&d);
    let k = 8;
    let pr = PcaParams { n_comps: k, oversample: 20, power_iters: 12, clip: None, ..PcaParams::default() };
    let r = implicit_scaled_pca(x.view(), &pr).unwrap();
    assert_eq!(r.gene_sd[0], 1.0);
    assert_eq!(r.gene_sd[10], 1.0);
    assert_eq!(r.gene_mean[0], 0.0);
    assert_eq!(r.gene_mean[10], 1.7f32);
    for c in 0..k {
        assert_eq!(r.loadings.data[c], 0.0, "all-zero gene loading");
        assert_eq!(r.loadings.data[10 * k + c], 0.0, "constant gene loading");
    }
    // The single-cell gene is variable: sd = sqrt(4/n - 4/n^2 ... ) computed directly.
    let (_, sd) = reference_stats(&d);
    assert!(((r.gene_sd[p + 2] as f64) - sd[p + 2]).abs() < 1e-6 * sd[p + 2]);
    // Everything else equals the PCA of the matrix with the constant genes left out.
    let keep: Vec<usize> = (0..p + 3).filter(|&j| j != 0 && j != 10).collect();
    let reduced = Mat::from_fn(n, keep.len(), |i, j| d[(i, keep[j])]);
    let ex = exact_pca(&dense_z(&reduced, true));
    let dv = singular_values(&r, n);
    for c in 0..k {
        assert!((dv[c] - ex.d[c]).abs() / ex.d[c] < 1e-6, "sigma_{c}");
    }
    assert!(subspace_similarity(&mat(&r.embeddings), &ex.emb, k) > 0.9999);
    // An all-constant matrix has no variance at all: every component is zero.
    // 200 x 50 also runs several Krylov blocks, whose `Z g` are exactly zero.
    for (n, p) in [(30, 10), (200, 50)] {
        let flat = Csr::from_dense(&Mat::from_fn(n, p, |_, j| j as f64));
        let r = implicit_scaled_pca(flat.view(), &params(3)).unwrap();
        assert!(r.stdev.iter().all(|&s| s == 0.0));
        assert!(r.embeddings.data.iter().all(|&v| v == 0.0));
        assert!(r.gene_sd.iter().all(|&s| s == 1.0));
    }
}

// ---------------------------------------------------------------- Seurat parity

fn data_lines(path: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(String::from)
        .collect()
}

/// tests/fixtures/pbmc_like counts, log-normalised as Seurat's NormalizeData
/// (log1p(count / cell total * 1e4)), restricted to the HVGs in Seurat's order.
fn pbmc_like_lognorm(hvg: &[usize]) -> Csr {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/pbmc_like");
    let text = std::fs::read_to_string(root.join("matrix.mtx")).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('%'));
    let dims: Vec<usize> = lines.next().unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
    let (genes, cells) = (dims[0], dims[1]);
    let mut per_cell: Vec<Vec<(usize, f64)>> = vec![vec![]; cells];
    for l in lines {
        let f: Vec<&str> = l.split_whitespace().collect();
        let (g, c, v): (usize, usize, f64) = (f[0].parse().unwrap(), f[1].parse().unwrap(), f[2].parse().unwrap());
        per_cell[c - 1].push((g - 1, v));
    }
    let mut col_of = vec![usize::MAX; genes];
    for (k, &g) in hvg.iter().enumerate() {
        col_of[g] = k;
    }
    let mut x = Csr { n: cells, p: hvg.len(), indptr: vec![0], indices: vec![], data: vec![] };
    for entries in &per_cell {
        let total: f64 = entries.iter().map(|e| e.1).sum();
        let mut row: Vec<(u32, f64)> = entries
            .iter()
            .filter(|e| col_of[e.0] != usize::MAX && e.1 != 0.0)
            .map(|e| (col_of[e.0] as u32, (e.1 / total * 1e4).ln_1p()))
            .collect();
        row.sort_by_key(|e| e.0);
        x.indices.extend(row.iter().map(|e| e.0));
        x.data.extend(row.iter().map(|e| e.1));
        x.indptr.push(x.indices.len() as u32);
    }
    x
}

/// Parity with Seurat 5.0.1 ScaleData + RunPCA(npcs = 30, seed.use = 0) on the 500-cell
/// fixture (reference in tests/data, regenerated by tests/data/make_seurat_pca_fixture.R).
///
/// The fixture has four cell types, so PCs 5-30 lie in a flat noise bulk. Seurat's default
/// `scale.max = 10` clips 0.05% of values, which reorders those noise PCs (the exact
/// unclipped SVD is only 0.978 similar to clipped RunPCA), so the default (clipped) run is
/// checked against clipped Seurat and `clip: None` against `scale.max = Inf`.
#[test]
fn seurat_parity_pbmc_like() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let hvg: Vec<usize> =
        data_lines(&dir.join("pbmc_like_hvg.txt"))[0].split_whitespace().map(|v| v.parse().unwrap()).collect();
    let pcs: Vec<Vec<f64>> = data_lines(&dir.join("pbmc_like_seurat_pcs.tsv"))[1..]
        .iter()
        .map(|l| l.split('\t').map(|v| v.trim().parse().unwrap()).collect())
        .collect();
    let stdev: Vec<Vec<f64>> = data_lines(&dir.join("pbmc_like_seurat_stdev.tsv"))[1..]
        .iter()
        .map(|l| l.split('\t').skip(1).map(|v| v.trim().parse().unwrap()).collect())
        .collect();
    let x = pbmc_like_lognorm(&hvg);
    assert_eq!((x.n, x.p), (500, 800));
    let seurat_clip = Mat::from_fn(500, 10, |i, c| pcs[i][c]);
    let seurat_inf = Mat::from_fn(500, 10, |i, c| pcs[i][10 + c]);

    let r = implicit_scaled_pca(x.view(), &PcaParams { clip: None, ..PcaParams::default() }).unwrap();
    let emb = mat(&r.embeddings);
    let rc = implicit_scaled_pca(x.view(), &PcaParams::default()).unwrap();
    let emb_c = mat(&rc.embeddings);
    let sim_inf = subspace_similarity(&emb, &seurat_inf, 10);
    let sim_clip = subspace_similarity(&emb_c, &seurat_clip, 10);
    let corr_inf = column_correlations(&emb, &seurat_inf, 10);
    let corr_clip = column_correlations(&emb_c, &seurat_clip, 10);
    let sd_err: Vec<f64> = (0..10).map(|c| (r.stdev[c] as f64 - stdev[c][1]).abs() / stdev[c][1]).collect();
    let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(" ");
    println!("top-10 subspace similarity vs Seurat scale.max=Inf (clip None): {sim_inf:.6}, vs scale.max=10 (clip 10): {sim_clip:.6}");
    println!("|corr| PC1-10 vs Inf: {}", fmt(&corr_inf));
    println!("|corr| PC1-10 vs 10:  {}", fmt(&corr_clip));
    println!("stdev rel. error PC1-10 vs Inf: {}", sd_err.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>().join(" "));
    assert!(sim_inf > 0.999, "vs Seurat scale.max = Inf: {sim_inf}");
    assert!(corr_inf.iter().all(|&c| c > 0.99), "per-PC correlation vs Inf");
    assert!(sd_err.iter().all(|&e| e < 1e-3), "stdev vs Inf");
    assert!(sim_clip > 0.999, "vs Seurat scale.max = 10: {sim_clip}");
    assert!(corr_clip.iter().all(|&c| c > 0.99), "per-PC correlation vs 10");
}

// ---------------------------------------------------------------- timing harness (ignored)

fn status_kb(key: &str) -> f64 {
    let st = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    st.lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(f64::NAN)
}

fn read_bin<T: Copy>(path: &std::path::Path, from: fn(&[u8]) -> T, size: usize) -> Vec<T> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(size).map(from).collect()
}

fn write_f32(path: &std::path::Path, v: &[f32]) {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

/// Times `implicit_scaled_pca` on a matrix exported by Seurat (CSR arrays as written by
/// `writeBin`: X_indptr.i32, X_indices.i32, X_data.f64 and dims.txt = "n p nnz"), records
/// peak RSS per run (VmHWM after resetting it through /proc/self/clear_refs), and writes the
/// result next to the input as scfast_q<power_iters>_os<oversample>_{emb,load,stdev}.f32 (row-major).
/// Env: SCFAST_PCA_BENCH=<dir>, SCFAST_PCA_REPS (3), SCFAST_PCA_THREADS ("1,2"),
/// SCFAST_PCA_POWER_ITERS and SCFAST_PCA_OVERSAMPLE (core defaults), SCFAST_PCA_NCOMPS (30).
#[test]
#[ignore]
fn bench_from_env() {
    let Ok(dir) = std::env::var("SCFAST_PCA_BENCH") else {
        eprintln!("SCFAST_PCA_BENCH not set; nothing to do");
        return;
    };
    let env = |k: &str, d: &str| std::env::var(k).unwrap_or_else(|_| d.to_string());
    let reps: usize = env("SCFAST_PCA_REPS", "3").parse().unwrap();
    let threads: Vec<usize> = env("SCFAST_PCA_THREADS", "1,2").split(',').map(|t| t.trim().parse().unwrap()).collect();
    let power_iters: usize = env("SCFAST_PCA_POWER_ITERS", &PcaParams::default().power_iters.to_string()).parse().unwrap();
    let k: usize = env("SCFAST_PCA_NCOMPS", "30").parse().unwrap();
    let oversample: usize = env("SCFAST_PCA_OVERSAMPLE", &PcaParams::default().oversample.to_string()).parse().unwrap();
    let dir = std::path::PathBuf::from(dir);
    let dims: Vec<usize> =
        std::fs::read_to_string(dir.join("dims.txt")).unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
    let (n, p) = (dims[0], dims[1]);
    let i32le = |b: &[u8]| i32::from_le_bytes(b.try_into().unwrap());
    let f64le = |b: &[u8]| f64::from_le_bytes(b.try_into().unwrap());
    let indptr = read_bin(&dir.join("X_indptr.i32"), i32le, 4);
    let indices = read_bin(&dir.join("X_indices.i32"), i32le, 4);
    let data = read_bin(&dir.join("X_data.f64"), f64le, 8);
    let view = CsrView::new(n, p, &indptr, &indices, &data).unwrap();
    let pr = PcaParams { n_comps: k, power_iters, oversample, ..PcaParams::default() };
    println!("input: {n} cells x {p} genes, nnz {} ({:.1} MB as i32 + f64)", data.len(), (data.len() * 12) as f64 / 1e6);
    let mut rows = vec!["threads\trep\twall_s\trss_before_mb\tpeak_mb".to_string()];
    let mut last = None;
    for &t in &threads {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
        for rep in 0..=reps {
            let _ = std::fs::write("/proc/self/clear_refs", "5");
            let rss0 = status_kb("VmRSS:");
            let t0 = std::time::Instant::now();
            let r = pool.install(|| implicit_scaled_pca(view, &pr).unwrap());
            let wall = t0.elapsed().as_secs_f64();
            let peak = status_kb("VmHWM:");
            println!("threads {t} rep {rep}: {wall:.4} s, RSS before {:.1} MB, peak {:.1} MB (+{:.1})", rss0 / 1024.0, peak / 1024.0, (peak - rss0) / 1024.0);
            rows.push(format!("{t}\t{rep}\t{wall:.6}\t{:.2}\t{:.2}", rss0 / 1024.0, peak / 1024.0));
            last = Some(r);
        }
    }
    let r = last.unwrap();
    let tag = format!("scfast_q{power_iters}_os{oversample}");
    write_f32(&dir.join(format!("{tag}_emb.f32")), &r.embeddings.data);
    write_f32(&dir.join(format!("{tag}_load.f32")), &r.loadings.data);
    write_f32(&dir.join(format!("{tag}_stdev.f32")), &r.stdev);
    std::fs::write(dir.join(format!("{tag}_timing.tsv")), rows.join("\n") + "\n").unwrap();
}
