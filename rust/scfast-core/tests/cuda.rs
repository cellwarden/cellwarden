//! CUDA backend tests (feature `cuda`, docs/scfast-integration.md Phase 6).
//!
//! The parity tests need a CUDA device and are `#[ignore]`d; everything else here runs on
//! any machine with `cargo test --features cuda` and checks the behaviour of the backend
//! when no device is present (clear `Error::Gpu`, no panic, CPU untouched).
//!
//! On a GPU node (e.g. the L4):
//!
//!   cd rust && cargo test -p scfast-core --release --features cuda --test cuda -- --ignored --nocapture
//!
//! Thresholds are the Phase 3 / 4 ones: top-10 PC subspace similarity > 0.99 against the
//! CPU result, kNN neighbour overlap >= 90% (in practice ~100%: the GPU search is exact up
//! to f32 candidate ranking).
#![cfg(feature = "cuda")]

use faer::Mat;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use scfast_core::knn::{knn, knn_on, KnnMethod, KnnParams};
use scfast_core::pca::{implicit_scaled_pca, implicit_scaled_pca_device, implicit_scaled_pca_on, DeviceCsr, PcaParams};
use scfast_core::{device_info, CsrView, Device, Error, RowMajor};

const GPU_SKIP: &str = "needs a CUDA device (run on the L4 with `cargo test --features cuda -- --ignored`)";

// ---------------------------------------------------------------- helpers

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
}

fn normal(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = 1.0 - rng.random::<f64>();
    let u2: f64 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Sparse log-normalised-looking data with `groups` cell types and continuous programs
/// (same construction as tests/pca.rs).
fn structured(n: usize, p: usize, groups: usize, seed: u64) -> Csr {
    const PROGRAMS: usize = 10;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let base: Vec<f64> = (0..p).map(|_| rng.random_range(0.05..0.45)).collect();
    let markers_per_group = (p / (3 * groups)).max(1);
    let mut indptr = vec![0u32];
    let (mut indices, mut data) = (Vec::new(), Vec::new());
    for i in 0..n {
        let g = i % groups;
        let level_g = 1.5 + 2.0 * 0.8f64.powi(g as i32);
        let scores: Vec<f64> = (0..PROGRAMS).map(|f| normal(&mut rng) * 0.9 * 0.8f64.powi(f as i32)).collect();
        for j in 0..p {
            let is_marker = j / markers_per_group == g && j < groups * markers_per_group;
            let s = scores[j % PROGRAMS];
            let prob = ((if is_marker { 3.0 } else { 1.0 }) * base[j] * s.exp()).min(0.95);
            if rng.random::<f64>() < prob {
                let level = if is_marker { level_g } else { 1.0 };
                indices.push(j as u32);
                data.push(level * (0.5 + rng.random::<f64>()) * (0.5 * s).exp());
            }
        }
        indptr.push(indices.len() as u32);
    }
    Csr { n, p, indptr, indices, data }
}

fn clustered(n: usize, d: usize, n_clusters: usize, seed: u64) -> RowMajor {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let centres: Vec<Vec<f64>> = (0..n_clusters).map(|_| (0..d).map(|t| 12.0 * 0.85f64.powi(t as i32) * normal(&mut rng)).collect()).collect();
    let mut data = Vec::with_capacity(n * d);
    for i in 0..n {
        let c = i % n_clusters;
        for t in 0..d {
            data.push((centres[c][t] + 0.8 * normal(&mut rng)) as f32);
        }
    }
    RowMajor::from_vec(n, d, data).unwrap()
}

fn mat(r: &RowMajor) -> Mat<f64> {
    Mat::from_fn(r.n_rows, r.n_cols, |i, j| r.data[i * r.n_cols + j] as f64)
}

fn subspace_similarity(a: &Mat<f64>, b: &Mat<f64>, k: usize) -> f64 {
    let qa = a.as_ref().subcols(0, k).qr().compute_thin_Q();
    let qb = b.as_ref().subcols(0, k).qr().compute_thin_Q();
    let m = qa.transpose() * &qb;
    let s = m.thin_svd().unwrap();
    (0..k).map(|i| s.S()[i].min(1.0)).sum::<f64>() / k as f64
}

fn overlap(a: &scfast_core::Knn, b: &scfast_core::Knn) -> f64 {
    assert_eq!((a.n, a.k), (b.n, b.k));
    let mut shared = 0usize;
    for i in 0..a.n {
        let ra = &a.indices[i * a.k..(i + 1) * a.k];
        let rb = &b.indices[i * b.k..(i + 1) * b.k];
        shared += ra.iter().filter(|j| rb.contains(j)).count();
    }
    shared as f64 / (a.n * a.k) as f64
}

// ---------------------------------------------------------------- no-device behaviour

#[test]
fn info_reports_cuda_built() {
    let info = device_info();
    assert!(info.cuda_built);
    if !info.cuda_available {
        eprintln!("CUDA unavailable here: {}", info.cuda_error.as_deref().unwrap_or("?"));
        assert_eq!(Device::auto(), Device::Cpu);
    } else {
        eprintln!("CUDA device: {:?}, {:.1} GiB free", info.device_name, info.free_gb().unwrap_or(0.0));
    }
}

#[test]
fn cuda_request_without_device_is_a_gpu_error_and_cpu_still_works() {
    if device_info().cuda_available {
        return;
    }
    let x = structured(300, 80, 3, 1);
    let pr = PcaParams { n_comps: 5, ..Default::default() };
    match implicit_scaled_pca_on(x.view(), &pr, Device::Cuda(0)) {
        Err(Error::Gpu(m)) => assert!(m.contains("CUDA"), "{m}"),
        other => panic!("expected Error::Gpu, got {:?}", other.map(|_| ())),
    }
    // The CPU path is untouched and identical to the device-less API.
    let a = implicit_scaled_pca(x.view(), &pr).unwrap();
    let b = implicit_scaled_pca_on(x.view(), &pr, Device::Cpu).unwrap();
    assert!(a.embeddings == b.embeddings);

    let e = clustered(200, 10, 3, 2);
    let kp = KnnParams { k: 10, method: KnnMethod::Exact, ..Default::default() };
    match knn_on(&e, &kp, Device::Cuda(0)) {
        Err(Error::Gpu(m)) => assert!(m.contains("CUDA"), "{m}"),
        other => panic!("expected Error::Gpu, got {:?}", other.map(|_| ())),
    }
    assert!(knn_on(&e, &kp, Device::Cpu).unwrap().indices == knn(&e, &kp).unwrap().indices);
}

#[test]
fn input_validation_precedes_device_use() {
    // Bad parameters are rejected before any CUDA call, whatever the device.
    let x = structured(50, 20, 2, 3);
    let bad = PcaParams { n_comps: 0, ..Default::default() };
    assert!(matches!(implicit_scaled_pca_on(x.view(), &bad, Device::Cuda(0)), Err(Error::InvalidParameter(_))));
    let e = clustered(20, 4, 2, 4);
    let kp = KnnParams { k: 1, ..Default::default() };
    assert!(matches!(knn_on(&e, &kp, Device::Cuda(0)), Err(Error::InvalidParameter(_))));
}

// ---------------------------------------------------------------- GPU parity (ignored)

#[test]
#[ignore = "needs a CUDA device (run on the L4 with `cargo test --features cuda -- --ignored`)"]
fn gpu_pca_matches_cpu() {
    let info = device_info();
    assert!(info.cuda_available, "{GPU_SKIP}: {}", info.cuda_error.as_deref().unwrap_or("?"));
    for &(n, p, seed) in &[(2000usize, 300usize, 1u64), (5000, 1000, 2), (20_000, 2000, 3)] {
        let x = structured(n, p, 6, seed);
        let pr = PcaParams { n_comps: 30, ..Default::default() };
        let cpu = implicit_scaled_pca(x.view(), &pr).unwrap();
        let t = std::time::Instant::now();
        let gpu = implicit_scaled_pca_on(x.view(), &pr, Device::Cuda(0)).unwrap();
        eprintln!("GPU PCA {n} x {p}: {:.2} s", t.elapsed().as_secs_f64());
        let sim = subspace_similarity(&mat(&cpu.embeddings), &mat(&gpu.embeddings), 10);
        assert!(sim > 0.99, "n={n} p={p}: top-10 subspace similarity {sim}");
        for c in 0..10 {
            let (a, b) = (cpu.stdev[c] as f64, gpu.stdev[c] as f64);
            assert!((a - b).abs() / a < 1e-3, "n={n} p={p}: stdev {c}: {a} vs {b}");
        }
        assert_eq!(cpu.gene_mean, gpu.gene_mean);
        assert_eq!(cpu.gene_sd, gpu.gene_sd);
    }
}

#[test]
#[ignore = "needs a CUDA device (run on the L4 with `cargo test --features cuda -- --ignored`)"]
fn gpu_knn_matches_cpu_exact() {
    let info = device_info();
    assert!(info.cuda_available, "{GPU_SKIP}: {}", info.cuda_error.as_deref().unwrap_or("?"));
    for &(n, d, k, seed) in &[(3000usize, 30usize, 20usize, 1u64), (20_000, 30, 20, 2), (50_000, 50, 31, 3), (1000, 10, 100, 4)] {
        let x = clustered(n, d, 9, seed);
        let kp = KnnParams { k, method: KnnMethod::Exact, ..Default::default() };
        let cpu = knn(&x, &kp).unwrap();
        let t = std::time::Instant::now();
        let gpu = knn_on(&x, &kp, Device::Cuda(0)).unwrap();
        eprintln!("GPU kNN {n} x {d}, k = {k}: {:.2} s", t.elapsed().as_secs_f64());
        assert_eq!((gpu.n, gpu.k), (n, k));
        for i in 0..n {
            assert_eq!(gpu.indices[i * k], i as u32, "self first");
            assert_eq!(gpu.distances[i * k], 0.0);
        }
        let ov = overlap(&cpu, &gpu);
        assert!(ov >= 0.9, "n={n} d={d} k={k}: overlap {ov}");
        // Where the neighbour sets agree the distances are the same exact f64 values.
        let mut same = 0usize;
        for t in 0..n * k {
            if cpu.indices[t] == gpu.indices[t] {
                assert_eq!(cpu.distances[t], gpu.distances[t]);
                same += 1;
            }
        }
        eprintln!("  identical entries: {:.4}", same as f64 / (n * k) as f64);
    }
}

#[test]
#[ignore = "needs a CUDA device (run on the L4 with `cargo test --features cuda -- --ignored`)"]
fn gpu_knn_ties_and_duplicates_follow_the_contract() {
    let info = device_info();
    assert!(info.cuda_available, "{GPU_SKIP}: {}", info.cuda_error.as_deref().unwrap_or("?"));
    // Integer grid (many exact ties) plus a point duplicated 25 times (> k).
    let mut data = Vec::new();
    for a in 0..8 {
        for b in 0..8 {
            for c in 0..4 {
                data.extend_from_slice(&[a as f32, b as f32, c as f32]);
            }
        }
    }
    for _ in 0..25 {
        data.extend_from_slice(&[3.0, 3.0, 1.0]);
    }
    let n = data.len() / 3;
    let x = RowMajor::from_vec(n, 3, data).unwrap();
    let kp = KnnParams { k: 20, method: KnnMethod::Exact, ..Default::default() };
    let cpu = knn(&x, &kp).unwrap();
    let gpu = knn_on(&x, &kp, Device::Cuda(0)).unwrap();
    assert!(overlap(&cpu, &gpu) >= 0.9);
    for i in 0..n {
        let row = &gpu.indices[i * 20..(i + 1) * 20];
        assert_eq!(row[0], i as u32);
        let d = &gpu.distances[i * 20..(i + 1) * 20];
        assert!(d.windows(2).skip(1).all(|w| w[0] <= w[1]), "row {i} not sorted");
    }
}

#[test]
#[ignore = "needs a CUDA device (run on the L4 with `cargo test --features cuda -- --ignored`)"]
fn gpu_auto_resolves_to_cuda_and_parse_accepts_ordinals() {
    let info = device_info();
    assert!(info.cuda_available, "{GPU_SKIP}: {}", info.cuda_error.as_deref().unwrap_or("?"));
    assert_eq!(Device::auto(), Device::Cuda(0));
    assert_eq!(Device::parse("cuda").unwrap(), Device::Cuda(0));
    assert_eq!(Device::parse("cuda:0").unwrap(), Device::Cuda(0));
    assert!(matches!(Device::parse("cuda:99"), Err(Error::Gpu(_))));
    assert!(info.device_name.is_some());
    assert!(info.free_gb().unwrap_or(0.0) > 0.0);
}

#[test]
#[ignore = "needs a CUDA device (run on the L4 with `cargo test --features cuda -- --ignored`)"]
fn gpu_pca_from_device_memory_matches_host_input() {
    use cudarc::driver::{CudaContext, DevicePtr};
    let info = device_info();
    assert!(info.cuda_available, "{GPU_SKIP}: {}", info.cuda_error.as_deref().unwrap_or("?"));
    let x = structured(20_000, 2000, 6, 3);
    let pr = PcaParams { n_comps: 30, ..Default::default() };
    let host = implicit_scaled_pca_on(x.view(), &pr, Device::Cuda(0)).unwrap();
    let stream = CudaContext::new(0).unwrap().default_stream();
    let ip = stream.clone_htod(&x.indptr.iter().map(|&v| v as i32).collect::<Vec<i32>>()).unwrap();
    let ix = stream.clone_htod(&x.indices).unwrap();
    let dx = stream.clone_htod(&x.data.iter().map(|&v| v as f32).collect::<Vec<f32>>()).unwrap();
    stream.synchronize().unwrap();
    let ((ip_p, _a), (ix_p, _b), (dx_p, _c)) = (ip.device_ptr(&stream), ix.device_ptr(&stream), dx.device_ptr(&stream));
    let dev = DeviceCsr { n: x.n, p: x.p, nnz: x.data.len(), indptr: ip_p, indptr_i64: false, indices: ix_p, data: dx_p };
    // SAFETY: the three buffers above hold the CSR and outlive the call.
    let gpu = unsafe { implicit_scaled_pca_device(&dev, &pr, 0) }.unwrap();
    let sim = subspace_similarity(&mat(&host.embeddings), &mat(&gpu.embeddings), 10);
    assert!(sim > 0.999, "top-10 subspace similarity {sim}");
    for (a, b) in host.stdev.iter().zip(&gpu.stdev).take(10) {
        assert!((a - b).abs() <= 1e-3 * a.abs(), "stdev {a} vs {b}");
    }
    for (a, b) in host.gene_mean.iter().zip(&gpu.gene_mean) {
        assert!((a - b).abs() <= 1e-5 * a.abs().max(1e-3), "gene mean {a} vs {b}");
    }
}

#[test]
#[ignore = "needs a CUDA device (run on the L4 with `cargo test --features cuda -- --ignored`)"]
fn gpu_pca_clip_matches_cpu_clip() {
    use cudarc::driver::{CudaContext, DevicePtr};
    let info = device_info();
    assert!(info.cuda_available, "{GPU_SKIP}: {}", info.cuda_error.as_deref().unwrap_or("?"));
    let mut x = structured(8000, 800, 6, 5);
    // Outliers: every 50th stored entry x 25, so many scaled values exceed scale.max = 10.
    for v in x.data.iter_mut().step_by(50) {
        *v *= 25.0;
    }
    let pr = PcaParams { n_comps: 30, ..Default::default() };
    let unclipped = implicit_scaled_pca(x.view(), &PcaParams { clip: None, ..pr.clone() }).unwrap();
    let cpu = implicit_scaled_pca(x.view(), &pr).unwrap();
    let clipped = (0..x.n)
        .flat_map(|i| (x.indptr[i] as usize..x.indptr[i + 1] as usize).map(move |t| (i, t)))
        .filter(|&(_, t)| {
            let j = x.indices[t] as usize;
            (x.data[t] - cpu.gene_mean[j] as f64) / cpu.gene_sd[j] as f64 > 10.0
        })
        .count();
    let sim_clip = subspace_similarity(&mat(&cpu.embeddings), &mat(&unclipped.embeddings), 10);
    eprintln!("{clipped} clipped entries; CPU clip vs no clip top-10 similarity {sim_clip}");
    assert!(clipped > 1000 && sim_clip < 0.999, "the clip must matter here");

    let stream = CudaContext::new(0).unwrap().default_stream();
    let ip = stream.clone_htod(&x.indptr.iter().map(|&v| v as i64).collect::<Vec<i64>>()).unwrap();
    let ix = stream.clone_htod(&x.indices).unwrap();
    let dx = stream.clone_htod(&x.data.iter().map(|&v| v as f32).collect::<Vec<f32>>()).unwrap();
    stream.synchronize().unwrap();
    let ((ip_p, _a), (ix_p, _b), (dx_p, _c)) = (ip.device_ptr(&stream), ix.device_ptr(&stream), dx.device_ptr(&stream));
    let dev = DeviceCsr { n: x.n, p: x.p, nnz: x.data.len(), indptr: ip_p, indptr_i64: true, indices: ix_p, data: dx_p };
    let host_input = implicit_scaled_pca_on(x.view(), &pr, Device::Cuda(0)).unwrap();
    // SAFETY: the three buffers above hold the CSR and outlive the call.
    let device_input = unsafe { implicit_scaled_pca_device(&dev, &pr, 0) }.unwrap();
    for (name, gpu) in [("host input", &host_input), ("device input", &device_input)] {
        let sim = subspace_similarity(&mat(&cpu.embeddings), &mat(&gpu.embeddings), 10);
        let load = subspace_similarity(&mat(&cpu.loadings), &mat(&gpu.loadings), 10);
        eprintln!("{name}: GPU clip vs CPU clip top-10 similarity {sim}, loadings {load}");
        assert!(sim > 0.9999 && load > 0.9999, "{name}: similarity {sim} / {load}");
        for (c, (a, b)) in cpu.stdev.iter().zip(&gpu.stdev).take(10).enumerate() {
            assert!((a - b).abs() <= 1e-4 * a.abs(), "{name}: stdev {c}: {a} vs {b}");
        }
    }
}
