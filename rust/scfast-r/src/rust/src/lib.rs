//! extendr bindings for scfast-core: the native side of the R package scfastR.
//!
//! Each `rs_*` function is a thin adapter: it borrows R memory, calls one scfast-core kernel
//! and converts the result to plain R vectors / matrices / lists. All Seurat-specific logic
//! (DimReduc, Graph, meta data) lives in R (R/scfast.R).
//!
//! Copies (docs/RUST_ENGINE.md section 3):
//! - PCA input: the dgCMatrix `@p`, `@i`, `@x` arrays are borrowed in place as a cells x genes
//!   `CsrView<i32, f64>`. No copy, no transpose.
//! - kNN input: the R embedding matrix (column-major f64) is borrowed, then written once into
//!   the row-major f32 `RowMajor` that `knn::knn` takes (cells x dims, small).
//! - SNN input: the 1-based integer index matrix is converted to the 0-based row-major `u32`
//!   `Knn` (n x k, small).
//! - Leiden input: the dgCMatrix arrays are converted to the owned `usize`/`u32`/`f32`
//!   `SparseGraph` that the clusterer takes (one copy of the graph, as planned in section 4).
//! - Harmony input: the R embedding matrix (column-major f64) is borrowed in place; the
//!   kernel makes its own d x cells working copy.
//! - Outputs are always new R vectors (f32 -> f64, row-major -> column-major).
//!
//! Errors: scfast-core `Error`s and Rust panics (including `unimplemented!()` in stub
//! kernels) are caught here and returned as `Err`, which the extendr wrapper turns into an
//! R error. Nothing unwinds into R.

use extendr_api::prelude::*;
use scfast_core::integrate::{self, HarmonyParams, Layout};
use scfast_core::{cluster, knn, pca, preprocess, snn, umap, CsrView, Device, Knn, RowMajor, SparseGraph};
use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

type RResult<T> = std::result::Result<T, extendr_api::Error>;

fn err<T>(msg: impl Into<String>) -> RResult<T> {
    Err(extendr_api::Error::Other(msg.into()))
}

// ------------------------------------------------------------------ panic handling

static QUIET_HOOK: Once = Once::new();

/// Replace the default panic hook (which prints "thread panicked at ..." to stderr) with a
/// silent one, because every panic is reported as an R error instead. Set the environment
/// variable SCFAST_DEBUG=1 to keep the default hook (and RUST_BACKTRACE to get backtraces).
fn install_quiet_hook() {
    QUIET_HOOK.call_once(|| {
        if std::env::var_os("SCFAST_DEBUG").is_none() {
            panic::set_hook(Box::new(|_| {}));
        }
    });
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Run one kernel, turning a scfast-core error or a panic into an R-facing error message
/// that names the kernel. Stub kernels (`unimplemented!()`) produce
/// "scfast <kernel>: not implemented: ...", which the R side maps to the condition class
/// `scfast_not_implemented`.
fn run_kernel<T>(kernel: &str, f: impl FnOnce() -> scfast_core::Result<T>) -> RResult<T> {
    install_quiet_hook();
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => err(format!("scfast {kernel}: {e}")),
        Err(payload) => err(format!("scfast {kernel}: {}", panic_message(payload.as_ref()))),
    }
}

// ------------------------------------------------------------------ argument helpers

fn int_slice<'a>(x: &'a Robj, what: &str) -> RResult<&'a [i32]> {
    match x.as_integer_slice() {
        Some(s) => Ok(s),
        None => err(format!("{what} must be an integer vector, got {:?}", x.rtype())),
    }
}

fn real_slice<'a>(x: &'a Robj, what: &str) -> RResult<&'a [f64]> {
    match x.as_real_slice() {
        Some(s) => Ok(s),
        None => err(format!("{what} must be a double vector, got {:?}", x.rtype())),
    }
}

fn non_negative(v: i32, what: &str) -> RResult<usize> {
    if v < 0 {
        return err(format!("{what} must be >= 0, got {v}"));
    }
    Ok(v as usize)
}

fn seed_u64(seed: f64) -> RResult<u64> {
    if !seed.is_finite() || seed < 0.0 || seed.fract() != 0.0 || seed > 9.007_199_254_740_992e15 {
        return err(format!("seed must be a non-negative whole number, got {seed}"));
    }
    Ok(seed as u64)
}

/// Borrow a genes x cells dgCMatrix (`@p`, `@i`, `@x`) as a cells x genes CSR view, in
/// place. Index validation (O(nnz)) is done by `CsrView::new`; values are checked for
/// NA/NaN/Inf here.
fn csc_as_csr<'a>(p: &'a Robj, i: &'a Robj, x: &'a Robj, n_genes: i32, n_cells: i32) -> RResult<CsrView<'a, i32, f64>> {
    let n_genes = non_negative(n_genes, "number of genes")?;
    let n_cells = non_negative(n_cells, "number of cells")?;
    let p = int_slice(p, "@p")?;
    let i = int_slice(i, "@i")?;
    let x = real_slice(x, "@x")?;
    if p.iter().any(|&v| v < 0) || i.iter().any(|&v| v < 0) {
        return err("invalid input: negative (or NA) entries in @p or @i");
    }
    if let Some(pos) = x.iter().position(|v| !v.is_finite()) {
        return err(format!("invalid input: @x contains a non-finite value (NA, NaN or Inf) at position {}", pos + 1));
    }
    CsrView::new(n_cells, n_genes, p, i, x).map_err(|e| extendr_api::Error::Other(format!("scfast: {e}")))
}

/// Row-major f32 -> new R double matrix (column-major).
fn rowmajor_to_rmatrix(m: &RowMajor) -> Robj {
    let (nr, nc) = (m.n_rows, m.n_cols);
    let mut out = RMatrix::<f64>::new(nr, nc);
    let dst = out.data_mut();
    for c in 0..nc {
        let col = &mut dst[c * nr..(c + 1) * nr];
        for (r, v) in col.iter_mut().enumerate() {
            *v = m.data[r * nc + c] as f64;
        }
    }
    out.into()
}

fn f32_to_r(v: &[f32]) -> Robj {
    v.iter().map(|&x| x as f64).collect::<Vec<f64>>().into()
}

// ------------------------------------------------------------------ PCA

/// Implicitly scaled PCA of a genes x cells dgCMatrix, read in place as cells x genes CSR.
/// `oversample` / `power_iters` < 0 keep the scfast-core defaults. `device` is "cpu",
/// "cuda" / "cuda:<i>" or "auto" (see `rs_device`); the device used is returned as `device`.
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_pca_csc(p: Robj, i: Robj, x: Robj, n_genes: i32, n_cells: i32, npcs: i32, seed: f64, oversample: i32, power_iters: i32, device: &str) -> RResult<Robj> {
    let view = csc_as_csr(&p, &i, &x, n_genes, n_cells)?;
    if npcs < 1 {
        return err(format!("npcs must be >= 1, got {npcs}"));
    }
    let dev = parse_device(device)?;
    let mut params = pca::PcaParams { n_comps: npcs as usize, seed: seed_u64(seed)?, ..Default::default() };
    if oversample >= 0 {
        params.oversample = oversample as usize;
    }
    if power_iters >= 0 {
        params.power_iters = power_iters as usize;
    }
    let res = run_kernel("pca", || pca::implicit_scaled_pca_on(view, &params, dev))?;
    if res.embeddings.n_rows != view.n_rows || res.loadings.n_rows != view.n_cols {
        return err(format!(
            "scfast pca: kernel returned embeddings {}x{} and loadings {}x{} for {} cells x {} genes",
            res.embeddings.n_rows, res.embeddings.n_cols, res.loadings.n_rows, res.loadings.n_cols, view.n_rows, view.n_cols
        ));
    }
    Ok(list!(
        embeddings = rowmajor_to_rmatrix(&res.embeddings),
        loadings = rowmajor_to_rmatrix(&res.loadings),
        stdev = f32_to_r(&res.stdev),
        gene_mean = f32_to_r(&res.gene_mean),
        gene_sd = f32_to_r(&res.gene_sd),
        device = dev.name()
    )
    .into())
}

// ------------------------------------------------------------------ kNN

fn knn_method(method: &str, hnsw_m: i32, ef_construction: i32, ef_search: i32) -> RResult<knn::KnnMethod> {
    match method {
        "auto" => Ok(knn::KnnMethod::Auto),
        "exact" => Ok(knn::KnnMethod::Exact),
        "hnsw" => {
            if hnsw_m < 1 || ef_construction < 1 || ef_search < 1 {
                return err("hnsw parameters (m, ef_construction, ef_search) must be >= 1");
            }
            Ok(knn::KnnMethod::Hnsw { m: hnsw_m as usize, ef_construction: ef_construction as usize, ef_search: ef_search as usize })
        }
        other => err(format!("method must be one of \"auto\", \"exact\", \"hnsw\"; got \"{other}\"")),
    }
}

/// Exact / HNSW k nearest neighbours (self included) of every row of a double matrix
/// (cells x dims). Returns 1-based indices and Euclidean distances, both n x k, and the
/// `device` used. `auto_threshold` < 0 keeps the scfast-core default. On a CUDA device
/// `method` is ignored (brute force GEMM + top-k).
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_knn(mat: Robj, n: i32, d: i32, k: i32, method: &str, seed: f64, hnsw_m: i32, ef_construction: i32, ef_search: i32, auto_threshold: f64, device: &str) -> RResult<Robj> {
    let dev = parse_device(device)?;
    let n = non_negative(n, "number of cells")?;
    let d = non_negative(d, "number of dimensions")?;
    let x = real_slice(&mat, "embedding matrix")?;
    if x.len() != n * d {
        return err(format!("embedding matrix has {} values, expected {n} x {d}", x.len()));
    }
    if let Some(pos) = x.iter().position(|v| !v.is_finite()) {
        return err(format!("embedding matrix contains a non-finite value (NA, NaN or Inf) at position {}", pos + 1));
    }
    if k < 2 || k as usize > n {
        return err(format!("k must be between 2 (it counts the cell itself) and the number of cells ({n}), got {k}"));
    }
    let mut params = knn::KnnParams { k: k as usize, method: knn_method(method, hnsw_m, ef_construction, ef_search)?, seed: seed_u64(seed)?, ..Default::default() };
    if auto_threshold >= 0.0 {
        params.auto_threshold = auto_threshold as usize;
    }
    // Column-major f64 (R) -> row-major f32 (scfast). The only copy, n x d.
    let mut rm = vec![0f32; n * d];
    for c in 0..d {
        let col = &x[c * n..(c + 1) * n];
        for (r, &v) in col.iter().enumerate() {
            rm[r * d + c] = v as f32;
        }
    }
    let rm = RowMajor::from_vec(n, d, rm).map_err(|e| extendr_api::Error::Other(format!("scfast: {e}")))?;
    let res = run_kernel("knn", || knn::knn_on(&rm, &params, dev))?;
    let kk = res.k;
    if res.n != n || res.indices.len() != n * kk || res.distances.len() != n * kk {
        return err(format!("scfast knn: kernel returned n = {}, k = {kk} with {} indices for {n} cells", res.n, res.indices.len()));
    }
    let mut idx = RMatrix::<i32>::new(n, kk);
    let mut dist = RMatrix::<f64>::new(n, kk);
    {
        let di = idx.data_mut();
        for c in 0..kk {
            for r in 0..n {
                di[c * n + r] = res.indices[r * kk + c] as i32 + 1;
            }
        }
    }
    {
        let dd = dist.data_mut();
        for c in 0..kk {
            for r in 0..n {
                dd[c * n + r] = res.distances[r * kk + c] as f64;
            }
        }
    }
    Ok(list!(idx = idx, dist = dist, device = dev.name()).into())
}

// ------------------------------------------------------------------ SNN

fn graph_to_r(g: &SparseGraph) -> RResult<Robj> {
    if g.nnz() > i32::MAX as usize {
        return err(format!("graph has {} edges, more than an R dgCMatrix can hold", g.nnz()));
    }
    if g.indptr.len() != g.n + 1 || g.weights.len() != g.indices.len() {
        return err("scfast: kernel returned an inconsistent graph");
    }
    let p: Vec<i32> = g.indptr.iter().map(|&v| v as i32).collect();
    let i: Vec<i32> = g.indices.iter().map(|&v| v as i32).collect();
    let x: Vec<f64> = g.weights.iter().map(|&v| v as f64).collect();
    Ok(list!(p = p, i = i, x = x, n = g.n as i32).into())
}

/// 1-based column-major n x k R index matrix -> 0-based row-major `u32`.
fn knn_indices(ix: &[i32], n: usize, k: usize) -> RResult<Vec<u32>> {
    let mut indices = vec![0u32; n * k];
    for c in 0..k {
        for r in 0..n {
            let v = ix[c * n + r];
            if v < 1 || v as usize > n {
                return err(format!("neighbour index {v} (row {}, column {}) is outside 1..{n}", r + 1, c + 1));
            }
            indices[r * k + c] = (v - 1) as u32;
        }
    }
    Ok(indices)
}

/// SNN (Jaccard) graph from an n x k matrix of 1-based neighbour indices (self first).
/// Returns the full symmetric adjacency as CSR parts (0-based), which are also its CSC parts.
/// @keywords internal
#[extendr]
fn rs_snn(idx: Robj, n: i32, k: i32, prune: f64) -> RResult<Robj> {
    let n = non_negative(n, "number of cells")?;
    let k = non_negative(k, "k")?;
    let ix = int_slice(&idx, "neighbour index matrix")?;
    if ix.len() != n * k || k == 0 {
        return err(format!("neighbour index matrix has {} values, expected {n} x {k} (k >= 1)", ix.len()));
    }
    if !prune.is_finite() || !(0.0..=1.0).contains(&prune) {
        return err(format!("prune must be in [0, 1], got {prune}"));
    }
    let indices = knn_indices(ix, n, k)?;
    // snn_from_knn reads only the indices; distances may be empty.
    let knn = Knn { n, k, indices, distances: Vec::new() };
    let params = snn::SnnParams { prune, ..Default::default() };
    let g = run_kernel("snn", || snn::snn_from_knn(&knn, &params))?;
    if g.n != n {
        return err(format!("scfast snn: kernel returned a graph of {} nodes for {n} cells", g.n));
    }
    graph_to_r(&g)
}

// ------------------------------------------------------------------ Leiden

/// Leiden (RB objective) on a symmetric n x n dgCMatrix given by its `@p`, `@i`, `@x`, on the
/// CPU. Returns 0-based labels numbered by decreasing cluster size.
/// @keywords internal
#[extendr]
fn rs_leiden(p: Robj, i: Robj, x: Robj, n: i32, resolution: f64, seed: f64, n_iter: i32) -> RResult<Robj> {
    leiden_impl(p, i, x, n, resolution, seed, n_iter, cluster::LeidenDevice::Cpu)
}

/// `rs_leiden` with a device: "cpu", "auto" (cuGraph when built with the `cugraph` feature,
/// loadable and a CUDA device is usable, else CPU), "cuda" or "cuda:<index>" (an error when
/// cuGraph is not available). Phase 7, docs/scfast-integration.md.
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_leiden_on(p: Robj, i: Robj, x: Robj, n: i32, resolution: f64, seed: f64, n_iter: i32, device: &str) -> RResult<Robj> {
    let device = cluster::LeidenDevice::parse(device).map_err(|e| extendr_api::Error::Other(e.to_string()))?;
    leiden_impl(p, i, x, n, resolution, seed, n_iter, device)
}

/// Resolves a Leiden device string to the device `rs_leiden_on` would run on: "cpu" or
/// "cuda:<index>". "auto" becomes "cuda:0" only when cuGraph is usable; an explicit "cuda"
/// that is not usable is an error naming the reason. Third element: whether the `cugraph`
/// feature was compiled in at all.
/// @keywords internal
#[extendr]
fn rs_leiden_device(device: &str) -> RResult<Robj> {
    let requested = cluster::LeidenDevice::parse(device).map_err(|e| extendr_api::Error::Other(e.to_string()))?;
    let resolved = run_kernel("leiden", || cluster::resolve_leiden_device(requested))?;
    Ok(list!(requested = requested.as_string(), resolved = resolved.as_string(), cugraph = cluster::cugraph_compiled()).into())
}

#[allow(clippy::too_many_arguments)]
fn leiden_impl(p: Robj, i: Robj, x: Robj, n: i32, resolution: f64, seed: f64, n_iter: i32, device: cluster::LeidenDevice) -> RResult<Robj> {
    let n = non_negative(n, "number of nodes")?;
    let p = int_slice(&p, "@p")?;
    let i = int_slice(&i, "@i")?;
    let x = real_slice(&x, "@x")?;
    if p.len() != n + 1 || i.len() != x.len() || p.first() != Some(&0) || p[n] as usize != x.len() {
        return err("invalid input: graph arrays are inconsistent (need length(p) = n + 1, p[1] = 0, p[n + 1] = length(i) = length(x))");
    }
    if p.windows(2).any(|w| w[1] < w[0]) {
        return err("invalid input: graph @p is not non-decreasing");
    }
    if i.iter().any(|&v| v < 0 || v as usize >= n) {
        return err(format!("invalid input: graph @i has entries outside 0..{}", n.saturating_sub(1)));
    }
    if x.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return err("invalid input: graph weights must be finite and non-negative");
    }
    if !resolution.is_finite() || resolution < 0.0 {
        return err(format!("resolution must be >= 0, got {resolution}"));
    }
    if n_iter < 1 {
        return err(format!("n_iter must be >= 1, got {n_iter}"));
    }
    let graph = SparseGraph {
        n,
        indptr: p.iter().map(|&v| v as usize).collect(),
        indices: i.iter().map(|&v| v as u32).collect(),
        weights: x.iter().map(|&v| v as f32).collect(),
    };
    let params = cluster::ClusterParams { resolution, seed: seed_u64(seed)?, n_iterations: n_iter as usize };
    let labels = run_kernel("leiden", || cluster::leiden_on(&graph, &params, device))?;
    if labels.len() != n {
        return err(format!("scfast leiden: kernel returned {} labels for {n} nodes", labels.len()));
    }
    Ok(labels.iter().map(|&l| l as i32).collect::<Vec<i32>>().into())
}

/// Is the n x n dgCMatrix (`@p`, `@i`, `@x`, row indices sorted within columns, as in any
/// valid dgCMatrix) symmetric to a relative tolerance? O(nnz log degree), no allocation,
/// unlike `isSymmetric()` in R, which builds the transpose.
/// @keywords internal
#[extendr]
fn rs_is_symmetric(p: Robj, i: Robj, x: Robj, n: i32, tol: f64) -> RResult<bool> {
    let n = non_negative(n, "number of nodes")?;
    let p = int_slice(&p, "@p")?;
    let i = int_slice(&i, "@i")?;
    let x = real_slice(&x, "@x")?;
    if p.len() != n + 1 || i.len() != x.len() || p[n] as usize != x.len() {
        return err("invalid input: graph arrays are inconsistent");
    }
    for col in 0..n {
        let (s, e) = (p[col] as usize, p[col + 1] as usize);
        for k in s..e {
            let row = i[k];
            if row < 0 || row as usize >= n {
                return err("invalid input: graph @i out of range");
            }
            let row = row as usize;
            if row == col {
                continue;
            }
            // Look for (col, row), i.e. row index `col` inside column `row`.
            let (s2, e2) = (p[row] as usize, p[row + 1] as usize);
            match i[s2..e2].binary_search(&(col as i32)) {
                Ok(pos) => {
                    let (a, b) = (x[k], x[s2 + pos]);
                    if (a - b).abs() > tol * a.abs().max(b.abs()).max(f64::MIN_POSITIVE) {
                        return Ok(false);
                    }
                }
                Err(_) => {
                    if x[k] != 0.0 {
                        return Ok(false);
                    }
                }
            }
        }
    }
    Ok(true)
}

// ------------------------------------------------------------------ Harmony

/// Harmony (harmony 2.x algorithm) on a double matrix (cells x dims) with integer batch
/// codes 1..n_batches (one per cell, every level used). `lambda` < 0 is the automatic
/// ridge penalty (alpha E, harmony's `lambda = NULL`), `nclust` < 0 is
/// `min(round(n / 30), 100)`, `epsilon_harmony` = -Inf disables early stopping and
/// `init_centroids` is NULL or an nclust x dims matrix. Returns the corrected matrix and
/// the convergence record.
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_harmony(
    mat: Robj,
    n: i32,
    d: i32,
    batches: Robj,
    n_batches: i32,
    theta: f64,
    lambda: f64,
    sigma: f64,
    nclust: i32,
    max_iter: i32,
    max_iter_cluster: i32,
    epsilon_cluster: f64,
    epsilon_harmony: f64,
    block_size: f64,
    tau: f64,
    alpha: f64,
    batch_prop_cutoff: f64,
    kmeans_nstart: i32,
    kmeans_iter: i32,
    seed: f64,
    device: &str,
    init_centroids: Robj,
) -> RResult<Robj> {
    let dev = parse_device(device)?;
    let n = non_negative(n, "number of cells")?;
    let d = non_negative(d, "number of dimensions")?;
    let x = real_slice(&mat, "embedding matrix")?;
    if x.len() != n * d {
        return err(format!("embedding matrix has {} values, expected {n} x {d}", x.len()));
    }
    let n_batches = non_negative(n_batches, "number of batches")?;
    let codes = int_slice(&batches, "batch codes")?;
    let mut b = Vec::with_capacity(codes.len());
    for (i, &c) in codes.iter().enumerate() {
        if c < 1 || c as usize > n_batches {
            return err(format!("batch code {c} of cell {} is not in 1..{n_batches} (NA batch labels are not allowed)", i + 1));
        }
        b.push((c - 1) as u32);
    }
    let init_centroids = if init_centroids.is_null() {
        None
    } else {
        let c = real_slice(&init_centroids, "init_centroids")?;
        let k = if d == 0 { 0 } else { c.len() / d };
        if k * d != c.len() || k == 0 {
            return err(format!("init_centroids has {} values, not a multiple of {d} dimensions", c.len()));
        }
        // column-major k x d (R) -> row-major
        let mut rm = vec![0f32; k * d];
        for j in 0..d {
            for i in 0..k {
                rm[i * d + j] = c[j * k + i] as f32;
            }
        }
        Some(RowMajor::from_vec(k, d, rm).map_err(|e| extendr_api::Error::Other(format!("scfast: {e}")))?)
    };
    let params = HarmonyParams {
        theta: vec![theta],
        lambda: if lambda < 0.0 { None } else { Some(lambda) },
        sigma,
        nclust: if nclust < 0 { None } else { Some(nclust as usize) },
        max_iter: non_negative(max_iter, "max_iter")?,
        max_iter_cluster: non_negative(max_iter_cluster, "max_iter_cluster")?,
        epsilon_cluster,
        epsilon_harmony,
        block_size,
        tau,
        alpha,
        batch_prop_cutoff,
        kmeans_nstart: non_negative(kmeans_nstart, "kmeans_nstart")?,
        kmeans_iter: non_negative(kmeans_iter, "kmeans_iter")?,
        seed: seed_u64(seed)?,
        init_centroids,
    };
    let res = run_kernel("harmony", || integrate::harmony_dense(x, n, d, Layout::ColMajor, &b, n_batches, &params, dev))?;
    let (z, info) = res.split_z();
    let mut out = RMatrix::<f64>::new(n, d);
    out.data_mut().copy_from_slice(&z);
    Ok(list!(
        embeddings = out,
        objective = info.objective,
        objective_kmeans = info.objective_kmeans,
        kmeans_rounds = info.kmeans_rounds.iter().map(|&v| v as i32).collect::<Vec<i32>>(),
        iters = info.iters as i32,
        converged = info.converged,
        nclust = info.n_clusters as i32,
        device = dev.name()
    )
    .into())
}

// ------------------------------------------------------------------ diagnostics

/// Build information: crate version, rayon thread count, compiled backends and the CUDA
/// device state (`cuda` = `cuda_built`, kept for older callers).
/// @keywords internal
#[extendr]
fn rs_info() -> Robj {
    let info = scfast_core::device_info();
    list!(
        version = env!("CARGO_PKG_VERSION"),
        threads = rayon::current_num_threads() as i32,
        cuda = info.cuda_built,
        cuda_built = info.cuda_built,
        cuda_available = info.cuda_available,
        device_count = info.device_count as i32,
        device_name = info.device_name.clone().map(Robj::from).unwrap_or_else(|| Robj::from(())),
        free_gb = info.free_gb().map(Robj::from).unwrap_or_else(|| Robj::from(())),
        total_gb = info.total_bytes.map(|b| Robj::from(b as f64 / (1u64 << 30) as f64)).unwrap_or_else(|| Robj::from(())),
        cuda_error = info.cuda_error.clone().map(Robj::from).unwrap_or_else(|| Robj::from(())),
        extendr = "0.9"
    )
    .into()
}

/// Parse a PCA / kNN device string with scfast-core's rules; errors are prefixed so the R
/// side can tell a GPU problem ("scfast device: GPU: ...") from a bad string.
fn parse_device(device: &str) -> RResult<Device> {
    Device::parse(device).map_err(|e| extendr_api::Error::Other(format!("scfast device: {e}")))
}

/// Resolves a PCA / kNN device string ("auto", "cpu", "cuda", "cuda:<i>") to the device the
/// kernels would run on: "cpu" or "cuda:<i>". "auto" is "cuda:0" only when scfastR was
/// built with the `cuda` feature and a device is usable; an explicit "cuda" that is not
/// usable is an error naming the reason.
/// @keywords internal
#[extendr]
fn rs_device(device: &str) -> RResult<String> {
    Ok(parse_device(device)?.name())
}

/// Address of an R vector's data, as a hex string (zero-copy checks).
/// @keywords internal
#[extendr]
fn rs_dataptr(x: Robj) -> RResult<String> {
    let addr = if let Some(s) = x.as_real_slice() {
        s.as_ptr() as usize
    } else if let Some(s) = x.as_integer_slice() {
        s.as_ptr() as usize
    } else {
        return err("rs_dataptr: x must be a double or integer vector");
    };
    Ok(format!("{addr:#x}"))
}

/// The data addresses inside the CsrView that `rs_pca_csc` builds from the same arguments
/// (zero-copy checks: they must equal `rs_dataptr()` of `@p`, `@i`, `@x`).
/// @keywords internal
#[extendr]
fn rs_csr_view_ptrs(p: Robj, i: Robj, x: Robj, n_genes: i32, n_cells: i32) -> RResult<Robj> {
    let v = csc_as_csr(&p, &i, &x, n_genes, n_cells)?;
    Ok(list!(
        indptr = format!("{:#x}", v.indptr.as_ptr() as usize),
        indices = format!("{:#x}", v.indices.as_ptr() as usize),
        data = format!("{:#x}", v.data.as_ptr() as usize),
        n_rows = v.n_rows as i32,
        n_cols = v.n_cols as i32,
        nnz = v.nnz() as i32
    )
    .into())
}

/// Panics on purpose (checks that panics become R errors).
/// @keywords internal
#[extendr]
fn rs_debug_panic(msg: &str) -> RResult<Robj> {
    let m = msg.to_string();
    run_kernel("debug", move || -> scfast_core::Result<Robj> { panic!("{}", m) })
}

// ------------------------------------------------------------------ preprocessing (Phase 9)

/// Fused NormalizeData + FindVariableFeatures(vst) + ScaleData statistics of a genes x cells
/// counts dgCMatrix, read in place as cells x genes CSR. Returns the log-normalised `@x`
/// (same `@p`/`@i` as the counts), per-cell library sizes, the per-gene vst columns, and the
/// HVGs (1-based, most variable first) with the mean / sd of their log-normalised values.
/// `clip` <= 0 means sqrt(n_cells).
/// @keywords internal
#[extendr]
fn rs_preprocess_csc(p: Robj, i: Robj, x: Robj, n_genes: i32, n_cells: i32, n_hvg: i32, scale_factor: f64, span: f64, clip: f64) -> RResult<Robj> {
    let view = csc_as_csr(&p, &i, &x, n_genes, n_cells)?;
    if n_hvg < 1 {
        return err(format!("n_hvg must be >= 1, got {n_hvg}"));
    }
    let params = preprocess::PreprocessParams {
        n_hvg: n_hvg as usize,
        scale_factor,
        span,
        clip: if clip > 0.0 { Some(clip) } else { None },
    };
    let res: preprocess::PreprocessResult<f64> = run_kernel("preprocess", || preprocess::preprocess(view, &params))?;
    if res.data.len() != view.nnz() || res.library_size.len() != view.n_rows || res.mean.len() != view.n_cols {
        return err(format!(
            "scfast preprocess: kernel returned {} values, {} library sizes and {} gene means for {} non-zeros, {} cells, {} genes",
            res.data.len(), res.library_size.len(), res.mean.len(), view.nnz(), view.n_rows, view.n_cols
        ));
    }
    Ok(list!(
        data = res.data,
        library_size = res.library_size,
        mean = res.mean,
        variance = res.variance,
        variance_expected = res.variance_expected,
        variance_standardized = res.variance_standardized,
        clip = res.clip,
        hvg = res.hvg.iter().map(|&j| j as i32 + 1).collect::<Vec<i32>>(),
        hvg_mean = res.hvg_mean,
        hvg_sd = res.hvg_sd
    )
    .into())
}

/// Local quadratic loess of y on x evaluated at every x (R's loess with surface = "direct").
/// For the parity tests of the loess used by rs_preprocess_csc.
/// @keywords internal
#[extendr]
fn rs_loess_direct(x: Robj, y: Robj, span: f64) -> RResult<Robj> {
    let xs = real_slice(&x, "x")?;
    let ys = real_slice(&y, "y")?;
    let f = run_kernel("loess", || preprocess::loess_direct(xs, ys, span))?;
    Ok(f.into())
}

// ------------------------------------------------------------------ UMAP

/// UMAP layout from an n x k kNN graph: 1-based indices and distances (self first), an
/// n x 2 starting layout and the curve parameters `a`, `b`. Returns the n x 2 embedding.
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn rs_umap(idx: Robj, dist: Robj, init: Robj, n: i32, k: i32, a: f64, b: f64, n_epochs: i32, negative_sample_rate: i32, learning_rate: f64, seed: f64) -> RResult<Robj> {
    let n = non_negative(n, "number of cells")?;
    let k = non_negative(k, "k")?;
    let ix = int_slice(&idx, "neighbour index matrix")?;
    let dx = real_slice(&dist, "neighbour distance matrix")?;
    let x0 = real_slice(&init, "init")?;
    if ix.len() != n * k || dx.len() != n * k || x0.len() != n * 2 {
        return err(format!("idx ({}), dist ({}) and init ({}) values do not match {n} cells x {k} neighbours", ix.len(), dx.len(), x0.len()));
    }
    if let Some(pos) = dx.iter().chain(x0).position(|v| !v.is_finite()) {
        return err(format!("dist / init contain a non-finite value at position {}", pos + 1));
    }
    let n_epochs = non_negative(n_epochs, "n_epochs")?;
    let negative_sample_rate = non_negative(negative_sample_rate, "negative_sample_rate")?;
    let indices = knn_indices(ix, n, k)?;
    let mut distances = vec![0f32; n * k];
    for c in 0..k {
        for r in 0..n {
            distances[r * k + c] = dx[c * n + r] as f32;
        }
    }
    let mut start = vec![0f32; n * 2];
    for c in 0..2 {
        for r in 0..n {
            start[r * 2 + c] = x0[c * n + r] as f32;
        }
    }
    let params = umap::UmapParams { a: a as f32, b: b as f32, n_epochs, negative_sample_rate, learning_rate: learning_rate as f32, seed: seed_u64(seed)? };
    let emb = run_kernel("umap", || umap::umap(&indices, &distances, n, k, &start, &params))?;
    let m = RowMajor::from_vec(n, 2, emb).map_err(|e| extendr_api::Error::Other(format!("scfast: {e}")))?;
    Ok(rowmajor_to_rmatrix(&m))
}

extendr_module! {
    mod scfastR;
    fn rs_pca_csc;
    fn rs_knn;
    fn rs_snn;
    fn rs_leiden;
    fn rs_leiden_on;
    fn rs_leiden_device;
    fn rs_is_symmetric;
    fn rs_info;
    fn rs_device;
    fn rs_dataptr;
    fn rs_csr_view_ptrs;
    fn rs_debug_panic;
    fn rs_preprocess_csc;
    fn rs_loess_direct;
    fn rs_umap;
    fn rs_harmony;
}
