//! Tests for `knn::knn`: exact search against a naive O(n^2) reference, the output
//! contract for every method, HNSW recall, and parameter validation.
//!
//! The timing bench is an ignored test (run in release):
//!   cargo test -p scfast-core --release --test knn -- --ignored --nocapture bench_knn
//! With SCFAST_KNN_BENCH_DIR set, it also writes each dataset and the exact / HNSW
//! indices there (raw little-endian f32 / i32) for the Seurat comparison.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use scfast_core::knn::{knn, KnnMethod, KnnParams};
use scfast_core::{Error, Knn, RowMajor};

// ---------------------------------------------------------------------------------------
// Data generators

fn normal(rng: &mut ChaCha8Rng) -> f64 {
    // Box-Muller; 1 - u keeps the log argument in (0, 1].
    let u: f64 = 1.0 - rng.random::<f64>();
    let v: f64 = rng.random::<f64>();
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

fn gaussian(n: usize, d: usize, offset: f64, seed: u64) -> RowMajor {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let data = (0..n * d).map(|_| (offset + normal(&mut rng)) as f32).collect();
    RowMajor::from_vec(n, d, data).unwrap()
}

/// PC-like data: clusters of unequal size whose centres spread less along later
/// dimensions (as PC variances decay), plus isotropic within-cluster noise.
fn clustered(n: usize, d: usize, n_clusters: usize, seed: u64) -> RowMajor {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let centres: Vec<Vec<f64>> = (0..n_clusters)
        .map(|_| (0..d).map(|t| 12.0 * 0.85f64.powi(t as i32) * normal(&mut rng)).collect())
        .collect();
    let weights: Vec<f64> = (0..n_clusters).map(|c| 1.0 / (1.0 + c as f64)).collect();
    let total: f64 = weights.iter().sum();
    let mut data = Vec::with_capacity(n * d);
    for _ in 0..n {
        let mut r = rng.random::<f64>() * total;
        let mut c = 0;
        while c + 1 < n_clusters && r >= weights[c] {
            r -= weights[c];
            c += 1;
        }
        let spread = 0.6 + 0.4 * c as f64 / n_clusters as f64;
        for t in 0..d {
            data.push((centres[c][t] + spread * normal(&mut rng)) as f32);
        }
    }
    RowMajor::from_vec(n, d, data).unwrap()
}

/// Integer grid points (many exactly equal distances) plus exact duplicates, one of them
/// repeated more than k times, in shuffled order.
fn ties_and_duplicates(seed: u64) -> RowMajor {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let d = 3;
    let mut rows: Vec<[f32; 3]> = Vec::new();
    for a in 0..8 {
        for b in 0..8 {
            for c in 0..4 {
                rows.push([a as f32, b as f32, c as f32]);
            }
        }
    }
    for _ in 0..30 {
        rows.push([3.0, 3.0, 1.0]); // 31 copies in total
    }
    for _ in 0..120 {
        let p = rows[rng.random_range(0..256)];
        rows.push(p);
    }
    for i in (1..rows.len()).rev() {
        rows.swap(i, rng.random_range(0..=i));
    }
    let n = rows.len();
    RowMajor::from_vec(n, d, rows.into_iter().flatten().collect()).unwrap()
}

// ---------------------------------------------------------------------------------------
// Reference and checks

fn d2(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(&u, &v)| (u as f64 - v as f64).powi(2)).sum()
}

/// Naive O(n^2 d) kNN with the contract: self first, then by (distance, index).
fn naive(x: &RowMajor, k: usize) -> (Vec<u32>, Vec<f64>) {
    let n = x.n_rows;
    let mut idx = Vec::with_capacity(n * k);
    let mut dist = Vec::with_capacity(n * k);
    for i in 0..n {
        let mut all: Vec<(f64, usize)> = (0..n).filter(|&j| j != i).map(|j| (d2(x.row(i), x.row(j)), j)).collect();
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        idx.push(i as u32);
        dist.push(0.0);
        for &(dd, j) in &all[..k - 1] {
            idx.push(j as u32);
            dist.push(dd.sqrt());
        }
    }
    (idx, dist)
}

fn params(k: usize, method: KnnMethod) -> KnnParams {
    KnnParams { k, method, auto_threshold: 50_000, seed: 0 }
}

fn exact(x: &RowMajor, k: usize) -> Knn {
    knn(x, &params(k, KnnMethod::Exact)).unwrap()
}

fn assert_matches_naive(x: &RowMajor, k: usize, label: &str) {
    let got = exact(x, k);
    let (ri, rd) = naive(x, k);
    assert_eq!((got.n, got.k), (x.n_rows, k), "{label}: shape");
    for i in 0..x.n_rows {
        let row = i * k..(i + 1) * k;
        assert_eq!(got.indices[row.clone()], ri[row.clone()], "{label}: indices of cell {i}");
        for p in row {
            let (a, b) = (got.distances[p] as f64, rd[p]);
            assert!((a - b).abs() <= 1e-5 * b.max(1.0), "{label}: cell {i} distance {a} vs {b}");
        }
    }
}

/// The output contract every method must satisfy.
fn assert_contract(x: &RowMajor, r: &Knn, label: &str) {
    let (n, k) = (x.n_rows, r.k);
    assert_eq!(r.n, n, "{label}");
    assert_eq!(r.indices.len(), n * k, "{label}");
    assert_eq!(r.distances.len(), n * k, "{label}");
    for i in 0..n {
        let idx = &r.indices[i * k..(i + 1) * k];
        let dist = &r.distances[i * k..(i + 1) * k];
        assert_eq!(idx[0] as usize, i, "{label}: cell {i} not first");
        assert_eq!(dist[0], 0.0, "{label}: cell {i} self distance");
        let mut seen = idx.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), k, "{label}: cell {i} has repeated neighbours");
        for t in 1..k {
            let j = idx[t] as usize;
            assert!(j < n, "{label}: index out of range");
            let want = d2(x.row(i), x.row(j)).sqrt();
            assert!((dist[t] as f64 - want).abs() <= 1e-5 * want.max(1.0), "{label}: cell {i} distance to {j}");
            if t > 1 {
                let (pd, pj) = (dist[t - 1], idx[t - 1]);
                assert!(pd < dist[t] || (pd == dist[t] && pj < idx[t]), "{label}: cell {i} not sorted at {t}");
            }
        }
    }
}

/// Fraction of the exact non-self neighbours that `approx` also found.
fn recall(approx: &Knn, exact: &Knn) -> f64 {
    let k = exact.k;
    let mut hit = 0usize;
    for i in 0..exact.n {
        let truth = &exact.indices[i * k + 1..(i + 1) * k];
        let got = &approx.indices[i * k + 1..(i + 1) * k];
        hit += got.iter().filter(|j| truth.contains(j)).count();
    }
    hit as f64 / (exact.n * (k - 1)) as f64
}

// ---------------------------------------------------------------------------------------
// Exact search

#[test]
fn exact_matches_naive_gaussian() {
    assert_matches_naive(&gaussian(700, 30, 0.0, 1), 20, "700x30 k20");
    assert_matches_naive(&gaussian(300, 2, 0.0, 2), 2, "300x2 k2");
    assert_matches_naive(&gaussian(400, 64, 0.0, 3), 31, "400x64 k31");
    assert_matches_naive(&gaussian(50, 5, 0.0, 4), 50, "50x5 k=n");
    assert_matches_naive(&gaussian(37, 1, 0.0, 5), 7, "37x1 k7");
}

#[test]
fn exact_matches_naive_across_chunks() {
    // More cells than one reference chunk (2048) and not a multiple of the block sizes.
    assert_matches_naive(&gaussian(2300, 10, 0.0, 6), 20, "2300x10 k20");
}

#[test]
fn exact_matches_naive_clustered() {
    assert_matches_naive(&clustered(1200, 30, 8, 7), 20, "clustered 1200x30");
    // k-d tree path (<= 16 dims), several levels deep
    assert_matches_naive(&clustered(9000, 10, 12, 11), 20, "clustered 9000x10");
}

#[test]
fn exact_matches_naive_uncentred() {
    // A large common offset: the GEMM works on centred data, and distances still come out
    // exactly as the naive computation on the original values.
    assert_matches_naive(&gaussian(600, 30, 1.0e4, 8), 20, "offset 1e4");
}

#[test]
fn exact_ties_and_duplicates_break_by_index() {
    let x = ties_and_duplicates(9);
    assert_matches_naive(&x, 20, "grid + duplicates k20");
    assert_matches_naive(&x, 40, "grid + duplicates k40");
    // A cell with 30 duplicates: self first, then 19 duplicates in index order at distance 0.
    let r = exact(&x, 20);
    let target = [3.0f32, 3.0, 1.0];
    let copies: Vec<u32> = (0..x.n_rows).filter(|&i| x.row(i) == target).map(|i| i as u32).collect();
    assert_eq!(copies.len(), 31);
    let i = copies[17] as usize;
    let row = &r.indices[i * 20..(i + 1) * 20];
    let others: Vec<u32> = copies.iter().copied().filter(|&j| j as usize != i).take(19).collect();
    assert_eq!(row[0] as usize, i);
    assert_eq!(&row[1..], &others[..]);
    assert!(r.distances[i * 20..(i + 1) * 20].iter().all(|&dd| dd == 0.0));
}

#[test]
fn exact_is_independent_of_thread_count() {
    let x = clustered(1500, 30, 6, 10);
    let many = exact(&x, 20);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let one = pool.install(|| exact(&x, 20));
    assert_eq!(many.indices, one.indices);
    assert_eq!(many.distances, one.distances);
}

// ---------------------------------------------------------------------------------------
// HNSW and Auto

#[test]
fn hnsw_recall_clustered_5k() {
    let x = clustered(5000, 30, 12, 11);
    let truth = exact(&x, 20);
    let approx = knn(&x, &params(20, KnnMethod::Hnsw { m: 16, ef_construction: 200, ef_search: 64 })).unwrap();
    assert_contract(&x, &approx, "hnsw 5k");
    let rc = recall(&approx, &truth);
    println!("HNSW recall@20 (non-self), 5000 x 30 clustered: {rc:.4}");
    assert!(rc >= 0.95, "recall {rc}");
}

#[test]
fn auto_switches_at_threshold() {
    let x = clustered(3000, 30, 10, 12);
    let truth = exact(&x, 20);

    let below = KnnParams { k: 20, method: KnnMethod::Auto, auto_threshold: 3001, seed: 0 };
    let r = knn(&x, &below).unwrap();
    assert_eq!(r.indices, truth.indices);
    assert_eq!(r.distances, truth.distances);

    let above = KnnParams { k: 20, method: KnnMethod::Auto, auto_threshold: 3000, seed: 0 };
    let r = knn(&x, &above).unwrap();
    assert_contract(&x, &r, "auto (hnsw)");
    let rc = recall(&r, &truth);
    assert!(rc >= 0.95, "auto HNSW recall {rc}");

    let x = clustered(3000, 10, 10, 12);
    let r = knn(&x, &above).unwrap();
    assert_eq!(r.indices, knn(&x, &params(20, KnnMethod::KdTree { eps: 0.7 })).unwrap().indices);
}

#[test]
fn kdtree_eps_recall_and_contract() {
    let x = clustered(20_000, 10, 12, 15);
    let truth = exact(&x, 20);
    let r = knn(&x, &params(20, KnnMethod::KdTree { eps: 0.7 })).unwrap();
    assert_contract(&x, &r, "kdtree eps 0.7");
    let rc = recall(&r, &truth);
    println!("k-d tree (eps 0.7) recall@20 (non-self), 20000 x 10 clustered: {rc:.4}");
    assert!(rc >= 0.9, "recall {rc}");
    let all = knn(&x, &params(20, KnnMethod::KdTree { eps: 0.0 })).unwrap();
    assert_eq!(all.indices, truth.indices);
    assert_eq!(all.distances, truth.distances);
    let r = knn(&ties_and_duplicates(16), &params(20, KnnMethod::KdTree { eps: 5.0 })).unwrap();
    assert_contract(&ties_and_duplicates(16), &r, "kdtree eps 5 duplicates");
    for eps in [-0.1, f64::NAN, f64::INFINITY] {
        assert!(knn(&x, &params(20, KnnMethod::KdTree { eps })).is_err(), "eps {eps}");
    }
}

#[test]
fn hnsw_contract_with_duplicates_and_small_n() {
    let x = ties_and_duplicates(13);
    let r = knn(&x, &params(20, KnnMethod::Hnsw { m: 8, ef_construction: 100, ef_search: 40 })).unwrap();
    assert_contract(&x, &r, "hnsw duplicates");

    let x = gaussian(25, 4, 0.0, 14);
    let r = knn(&x, &params(25, KnnMethod::Hnsw { m: 4, ef_construction: 20, ef_search: 1 })).unwrap();
    assert_contract(&x, &r, "hnsw k=n");
    assert_eq!(r.indices, exact(&x, 25).indices);
}

// ---------------------------------------------------------------------------------------
// Validation

fn is_param_err(r: Result<Knn, Error>) -> bool {
    matches!(r, Err(Error::InvalidParameter(_)))
}
fn is_input_err(r: Result<Knn, Error>) -> bool {
    matches!(r, Err(Error::InvalidInput(_)))
}

#[test]
fn rejects_invalid_parameters() {
    let x = gaussian(10, 3, 0.0, 15);
    for method in [KnnMethod::Exact, KnnMethod::Auto, KnnMethod::Hnsw { m: 16, ef_construction: 200, ef_search: 64 }] {
        assert!(is_param_err(knn(&x, &params(0, method))), "k = 0");
        assert!(is_param_err(knn(&x, &params(1, method))), "k = 1");
        assert!(is_param_err(knn(&x, &params(11, method))), "k > n");
        assert!(knn(&x, &params(10, method)).is_ok(), "k = n");
    }
    let hnsw = |m, ef_construction, ef_search| params(5, KnnMethod::Hnsw { m, ef_construction, ef_search });
    assert!(is_param_err(knn(&x, &hnsw(0, 200, 64))));
    assert!(is_param_err(knn(&x, &hnsw(1, 200, 64))));
    assert!(is_param_err(knn(&x, &hnsw(257, 200, 64))));
    assert!(is_param_err(knn(&x, &hnsw(16, 0, 64))));
    assert!(is_param_err(knn(&x, &hnsw(16, 200, 0))));
    let empty = RowMajor { n_rows: 0, n_cols: 3, data: vec![] };
    assert!(is_param_err(knn(&empty, &params(2, KnnMethod::Exact))));
}

#[test]
fn rejects_invalid_input() {
    let p = params(3, KnnMethod::Exact);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut x = gaussian(10, 3, 0.0, 16);
        x.data[7] = bad;
        assert!(is_input_err(knn(&x, &p)), "{bad}");
    }
    let wrong_len = RowMajor { n_rows: 10, n_cols: 3, data: vec![0.0; 29] };
    assert!(is_input_err(knn(&wrong_len, &p)));
    let no_dims = RowMajor { n_rows: 10, n_cols: 0, data: vec![] };
    assert!(is_input_err(knn(&no_dims, &p)));
}

// ---------------------------------------------------------------------------------------
// Bench (ignored; run in release)

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time<F: FnMut() -> Knn>(reps: usize, mut f: F) -> (f64, Knn) {
    let mut out = f(); // warm-up
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = std::time::Instant::now();
        out = f();
        times.push(t.elapsed().as_secs_f64());
    }
    (median(times), out)
}

fn write_f32(path: &std::path::Path, v: &[f32]) {
    std::fs::write(path, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}
fn write_idx(path: &std::path::Path, v: &[u32]) {
    std::fs::write(path, v.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

#[test]
#[ignore]
fn bench_knn() {
    let sizes: Vec<usize> = std::env::var("SCFAST_KNN_BENCH_SIZES")
        .map(|s| s.split(',').map(|t| t.trim().parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![2000, 5000]);
    let reps: usize = std::env::var("SCFAST_KNN_BENCH_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let out_dir = std::env::var("SCFAST_KNN_BENCH_DIR").ok().map(std::path::PathBuf::from);
    let (d, k) = (30, 20);
    let hnsw = KnnMethod::Hnsw { m: 16, ef_construction: 200, ef_search: (2 * k).max(64) };
    println!("threads = {}, d = {d}, k = {k}, reps = {reps} (median after 1 warm-up)", rayon::current_num_threads());
    println!("{:>8} {:>10} {:>10} {:>10}", "n", "exact_s", "hnsw_s", "recall");
    for &n in &sizes {
        let x = clustered(n, d, 12, 42 + n as u64);
        let (te, re) = time(reps, || knn(&x, &params(k, KnnMethod::Exact)).unwrap());
        let (th, rh) = time(reps, || knn(&x, &params(k, hnsw)).unwrap());
        println!("{n:>8} {te:>10.4} {th:>10.4} {:>10.4}", recall(&rh, &re));
        if let Some(dir) = &out_dir {
            std::fs::create_dir_all(dir).unwrap();
            write_f32(&dir.join(format!("x_{n}.f32")), &x.data);
            write_idx(&dir.join(format!("exact_{n}.i32")), &re.indices);
            write_idx(&dir.join(format!("hnsw_{n}.i32")), &rh.indices);
        }
    }
}
