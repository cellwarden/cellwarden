//! SNN (`snn::snn_from_knn`) and Leiden (`cluster::leiden`) tests.
//!
//! - Exact parity with Seurat 5.0.1 `ComputeSNN` on Seurat's own kNN indices, from the
//!   fixture `tests/data/seurat_snn_pbmc_like.bin` (layout in `SeuratFixture::load`). It was
//!   made from `tests/fixtures/pbmc_like` (500 cells): NormalizeData, FindVariableFeatures,
//!   ScaleData, RunPCA, then `FindNeighbors(dims = 1:20, k.param = 20, return.neighbor = TRUE)`
//!   for the indices, `Seurat:::ComputeSNN(Indices(nn), prune = 1/15)` for the graph (identical
//!   to `FindNeighbors(compute.SNN = TRUE)`'s `RNA_snn`), and
//!   `FindClusters(algorithm = 4, resolution = 0.5, random.seed = 0, n.iter = 2)` for labels.
//! - Brute-force SNN reference on random kNN sets, threading determinism, edge cases.
//! - Leiden: planted partitions, seed determinism, label order, Seurat-label agreement.
//! - `external_dataset_report` (ignored): timings and parity on an exported Seurat dataset.

use std::collections::HashMap;
use std::time::Instant;

use scfast_core::cluster::{leiden, ClusterParams};
use scfast_core::snn::{snn_from_knn, SnnParams};
use scfast_core::{Knn, SparseGraph};

// ------------------------------------------------------------------------------- helpers

/// SplitMix64: small, deterministic, no dependency.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u = self.uniform().max(1e-300);
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * self.uniform()).cos()
    }
}

fn knn_from(n: usize, k: usize, indices: Vec<u32>) -> Knn {
    Knn { n, k, indices, distances: Vec::new() }
}

/// Random neighbour sets of size k, distinct per row, with the cell itself first if `self_first`.
fn random_knn(n: usize, k: usize, self_first: bool, seed: u64) -> Knn {
    let mut rng = Rng(seed);
    let mut idx = Vec::with_capacity(n * k);
    for i in 0..n {
        let mut row: Vec<u32> = Vec::with_capacity(k);
        if self_first {
            row.push(i as u32);
        }
        while row.len() < k {
            // bias toward nearby ids so overlaps (and so surviving edges) are common
            let j = if rng.uniform() < 0.8 { (i + rng.below(3 * k)) % n } else { rng.below(n) };
            if !row.contains(&(j as u32)) && (self_first || j != i || rng.uniform() < 0.5) {
                row.push(j as u32);
            }
        }
        idx.extend(row);
    }
    knn_from(n, k, idx)
}

/// Exact kNN (Euclidean) by brute force, self first, ties by lower index.
fn brute_knn(x: &[Vec<f64>], k: usize) -> Knn {
    let n = x.len();
    let mut idx = Vec::with_capacity(n * k);
    let mut d: Vec<(f64, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        d.clear();
        for (j, xj) in x.iter().enumerate() {
            let dist: f64 = x[i].iter().zip(xj).map(|(a, b)| (a - b) * (a - b)).sum();
            d.push((dist, j as u32));
        }
        let cmp = |a: &(f64, u32), b: &(f64, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
        d.select_nth_unstable_by(k - 1, cmp);
        d[..k].sort_unstable_by(cmp);
        assert_eq!(d[0].1 as usize, i);
        idx.extend(d[..k].iter().map(|p| p.1));
    }
    knn_from(n, k, idx)
}

/// Gaussian blobs in `dim` dimensions: blob c centred at `sep * e_c`, unit variance.
fn blobs(sizes: &[usize], dim: usize, sep: f64, seed: u64) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut rng = Rng(seed);
    let (mut x, mut truth) = (Vec::new(), Vec::new());
    for (c, &size) in sizes.iter().enumerate() {
        for _ in 0..size {
            let mut p: Vec<f64> = (0..dim).map(|_| rng.normal()).collect();
            p[c % dim] += sep * (1 + c / dim) as f64;
            x.push(p);
            truth.push(c);
        }
    }
    // shuffle so cluster membership isn't aligned with node index
    let mut order: Vec<usize> = (0..x.len()).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
    }
    (order.iter().map(|&i| x[i].clone()).collect(), order.iter().map(|&i| truth[i]).collect())
}

fn row(g: &SparseGraph, i: usize) -> (&[u32], &[f32]) {
    let (s, e) = (g.indptr[i], g.indptr[i + 1]);
    (&g.indices[s..e], &g.weights[s..e])
}

fn weight(g: &SparseGraph, i: usize, j: usize) -> Option<f32> {
    let (c, w) = row(g, i);
    c.binary_search(&(j as u32)).ok().map(|p| w[p])
}

/// CSR well-formed, rows strictly increasing, exactly symmetric (pattern and weight).
fn assert_valid_symmetric(g: &SparseGraph) {
    assert_eq!(g.indptr.len(), g.n + 1);
    assert_eq!(g.indptr[0], 0);
    assert_eq!(g.indptr[g.n], g.indices.len());
    assert_eq!(g.indices.len(), g.weights.len());
    for i in 0..g.n {
        let (c, w) = row(g, i);
        assert!(c.windows(2).all(|p| p[0] < p[1]), "row {i} not strictly sorted");
        for (&j, &wij) in c.iter().zip(w) {
            assert!(wij > 0.0 && wij <= 1.0, "weight {wij} at ({i}, {j})");
            assert_eq!(weight(g, j as usize, i), Some(wij), "({i}, {j}) has no identical mirror");
        }
    }
}

/// Seurat's definition, pair by pair: Jaccard of the neighbour sets, pruned below `prune`.
fn reference_snn(knn: &Knn, prune: f64, keep_diagonal: bool) -> Vec<Vec<(u32, f32)>> {
    let (n, k) = (knn.n, knn.k);
    let sets: Vec<Vec<u32>> = (0..n)
        .map(|i| {
            let mut r = knn.indices[i * k..(i + 1) * k].to_vec();
            r.sort_unstable();
            r
        })
        .collect();
    let kf = k as f64;
    (0..n)
        .map(|i| {
            (0..n)
                .filter(|&j| keep_diagonal || j != i)
                .filter_map(|j| {
                    let s = sets[i].iter().filter(|x| sets[j].binary_search(x).is_ok()).count() as f64;
                    let w = s / (kf + (kf - s));
                    (s > 0.0 && w >= prune).then_some((j as u32, w as f32))
                })
                .collect()
        })
        .collect()
}

fn assert_matches_reference(g: &SparseGraph, reference: &[Vec<(u32, f32)>]) {
    assert_eq!(g.n, reference.len());
    for (i, want) in reference.iter().enumerate() {
        let (c, w) = row(g, i);
        let got: Vec<(u32, f32)> = c.iter().copied().zip(w.iter().copied()).collect();
        assert_eq!(&got, want, "row {i}");
    }
}

/// Adjusted Rand index.
fn ari<A: Copy + Eq + std::hash::Hash, B: Copy + Eq + std::hash::Hash>(a: &[A], b: &[B]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut joint: HashMap<(A, B), f64> = HashMap::new();
    let mut ma: HashMap<A, f64> = HashMap::new();
    let mut mb: HashMap<B, f64> = HashMap::new();
    for (&x, &y) in a.iter().zip(b) {
        *joint.entry((x, y)).or_default() += 1.0;
        *ma.entry(x).or_default() += 1.0;
        *mb.entry(y).or_default() += 1.0;
    }
    let c2 = |v: f64| v * (v - 1.0) / 2.0;
    let index: f64 = joint.values().map(|&v| c2(v)).sum();
    let (sa, sb): (f64, f64) = (ma.values().map(|&v| c2(v)).sum(), mb.values().map(|&v| c2(v)).sum());
    let expected = sa * sb / c2(a.len() as f64);
    let max = 0.5 * (sa + sb);
    if max == expected {
        1.0
    } else {
        (index - expected) / (max - expected)
    }
}

fn n_clusters(labels: &[u32]) -> usize {
    labels.iter().map(|&l| l as usize + 1).max().unwrap_or(0)
}

/// Labels are 0..n_clusters-1, all used, sizes non-increasing, ties ordered by first node.
fn assert_label_contract(labels: &[u32]) {
    let nc = n_clusters(labels);
    let mut size = vec![0usize; nc];
    let mut first = vec![usize::MAX; nc];
    for (v, &l) in labels.iter().enumerate() {
        size[l as usize] += 1;
        first[l as usize] = first[l as usize].min(v);
    }
    assert!(size.iter().all(|&s| s > 0), "unused label");
    for c in 1..nc {
        assert!(
            size[c - 1] > size[c] || (size[c - 1] == size[c] && first[c - 1] < first[c]),
            "labels {} and {c} out of order: sizes {:?}",
            c - 1,
            size
        );
    }
}

// --------------------------------------------------------------------- Seurat fixture

struct SeuratFixture {
    knn: Knn,
    prune: f64,
    /// Upper triangle (col >= row, diagonal included) of Seurat's SNN, CSR.
    up_indptr: Vec<usize>,
    up_indices: Vec<u32>,
    up_weights: Vec<f32>,
    labels: Vec<u8>,
}

impl SeuratFixture {
    /// Little-endian: b"SCFSNN01" | n u32 | k u32 | nnz_upper u32 | has_labels u32 | prune f64 |
    /// knn n*k u16 | indptr n+1 u32 | indices nnz u16 | weights nnz f32 | labels n u8.
    fn load() -> Self {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/seurat_snn_pbmc_like.bin");
        let b = std::fs::read(path).expect("fixture");
        assert_eq!(&b[..8], b"SCFSNN01");
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize;
        let u16_at = |o: usize| u16::from_le_bytes(b[o..o + 2].try_into().unwrap()) as u32;
        let (n, k, nnz) = (u32_at(8), u32_at(12), u32_at(16));
        let prune = f64::from_le_bytes(b[24..32].try_into().unwrap());
        let mut o = 32;
        let knn: Vec<u32> = (0..n * k).map(|t| u16_at(o + 2 * t)).collect();
        o += 2 * n * k;
        let up_indptr: Vec<usize> = (0..=n).map(|t| u32_at(o + 4 * t)).collect();
        o += 4 * (n + 1);
        let up_indices: Vec<u32> = (0..nnz).map(|t| u16_at(o + 2 * t)).collect();
        o += 2 * nnz;
        let up_weights: Vec<f32> = (0..nnz).map(|t| f32::from_le_bytes(b[o + 4 * t..o + 4 * t + 4].try_into().unwrap())).collect();
        o += 4 * nnz;
        let labels = b[o..o + n].to_vec();
        assert_eq!(o + n, b.len());
        Self { knn: knn_from(n, k, knn), prune, up_indptr, up_indices, up_weights, labels }
    }
}

// ------------------------------------------------------------------------------ SNN tests

#[test]
fn snn_matches_seurat_compute_snn_exactly() {
    let fx = SeuratFixture::load();
    let n = fx.knn.n;
    let g = snn_from_knn(&fx.knn, &SnnParams { prune: fx.prune, keep_diagonal: true }).unwrap();
    assert_valid_symmetric(&g);
    // full graph = mirrored upper triangle: 2 * upper - diagonal
    assert_eq!(g.nnz(), 2 * fx.up_indices.len() - n);
    let mut max_diff = 0.0f32;
    for i in 0..n {
        let (c, w) = row(&g, i);
        let start = c.partition_point(|&j| (j as usize) < i);
        let (want_c, want_w) = (&fx.up_indices[fx.up_indptr[i]..fx.up_indptr[i + 1]], &fx.up_weights[fx.up_indptr[i]..fx.up_indptr[i + 1]]);
        assert_eq!(&c[start..], want_c, "nonzero pattern differs from Seurat in row {i}");
        for (a, b) in w[start..].iter().zip(want_w) {
            max_diff = max_diff.max((a - b).abs());
        }
    }
    assert!(max_diff <= 1e-6, "max |w - w_seurat| = {max_diff}");
    // Seurat's diagonal is 1 for every cell
    assert!((0..n).all(|i| weight(&g, i, i) == Some(1.0)));
}

#[test]
fn snn_without_diagonal_drops_only_the_diagonal() {
    let fx = SeuratFixture::load();
    let with = snn_from_knn(&fx.knn, &SnnParams::default()).unwrap();
    let without = snn_from_knn(&fx.knn, &SnnParams { keep_diagonal: false, ..Default::default() }).unwrap();
    assert_valid_symmetric(&without);
    assert_eq!(without.nnz(), with.nnz() - with.n);
    for i in 0..with.n {
        let (c, w) = row(&with, i);
        let kept: Vec<(u32, f32)> = c.iter().copied().zip(w.iter().copied()).filter(|&(j, _)| j as usize != i).collect();
        let (c2, w2) = row(&without, i);
        assert_eq!(kept, c2.iter().copied().zip(w2.iter().copied()).collect::<Vec<_>>());
    }
}

#[test]
fn snn_prune_only_removes_entries_below_the_threshold() {
    let fx = SeuratFixture::load();
    let all = snn_from_knn(&fx.knn, &SnnParams { prune: 0.0, keep_diagonal: true }).unwrap();
    let pruned = snn_from_knn(&fx.knn, &SnnParams::default()).unwrap();
    assert_valid_symmetric(&all);
    assert!(all.nnz() > pruned.nnz());
    for i in 0..all.n {
        let (c, w) = row(&all, i);
        let kept: Vec<(u32, f32)> = c.iter().copied().zip(w.iter().copied()).filter(|&(_, w)| w as f64 >= 1.0 / 15.0).collect();
        let (c2, w2) = row(&pruned, i);
        assert_eq!(kept, c2.iter().copied().zip(w2.iter().copied()).collect::<Vec<_>>(), "row {i}");
    }
}

#[test]
fn snn_matches_brute_force_definition() {
    for &(n, k, self_first, seed) in &[(300, 10, true, 1), (257, 20, true, 2), (200, 7, false, 3), (60, 1, true, 4), (40, 40, true, 5)] {
        let knn = random_knn(n, k, self_first, seed);
        for prune in [0.0, 1.0 / 15.0, 0.2, 1.0] {
            for keep_diagonal in [true, false] {
                let g = snn_from_knn(&knn, &SnnParams { prune, keep_diagonal }).unwrap();
                assert_valid_symmetric(&g);
                assert_matches_reference(&g, &reference_snn(&knn, prune, keep_diagonal));
            }
        }
    }
}

#[test]
fn snn_prune_boundary_keeps_weights_equal_to_prune() {
    // k = 8, overlap 1: weight 1 / (8 + 7) == 1/15 exactly, which Seurat keeps (`< prune` drops)
    let n = 16;
    let mut idx = Vec::new();
    for i in 0..n {
        // rows 0 and 1 share exactly one neighbour (cell 15); others are disjoint blocks
        let row: Vec<u32> = match i {
            0 => vec![0, 2, 3, 4, 5, 6, 7, 15],
            1 => vec![1, 8, 9, 10, 11, 12, 13, 15],
            _ => (0..8).map(|t| ((i + t) % n) as u32).collect(),
        };
        idx.extend(row);
    }
    let knn = knn_from(n, 8, idx);
    let g = snn_from_knn(&knn, &SnnParams { prune: 1.0 / 15.0, keep_diagonal: true }).unwrap();
    assert_eq!(weight(&g, 0, 1), Some((1.0f64 / 15.0) as f32));
    assert_matches_reference(&g, &reference_snn(&knn, 1.0 / 15.0, true));
}

#[test]
fn snn_is_identical_across_thread_counts() {
    let knn = random_knn(6000, 20, true, 11);
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| snn_from_knn(&knn, &SnnParams::default()).unwrap())
    };
    let (a, b) = (run(1), run(4));
    assert_eq!(a.indptr, b.indptr);
    assert_eq!(a.indices, b.indices);
    assert_eq!(a.weights, b.weights);
    assert_valid_symmetric(&a);
}

#[test]
fn snn_edge_cases() {
    // k = 1 (only the cell itself): the identity, or nothing without the diagonal
    let knn = knn_from(5, 1, (0..5).collect());
    let g = snn_from_knn(&knn, &SnnParams::default()).unwrap();
    assert_eq!((g.indptr.clone(), g.indices.clone(), g.weights.clone()), ((0..=5).collect(), (0..5).collect(), vec![1.0; 5]));
    let g = snn_from_knn(&knn, &SnnParams { keep_diagonal: false, ..Default::default() }).unwrap();
    assert_eq!(g.nnz(), 0);
    // k = n: every pair shares every neighbour
    let knn = knn_from(3, 3, vec![0, 1, 2, 1, 2, 0, 2, 0, 1]);
    let g = snn_from_knn(&knn, &SnnParams::default()).unwrap();
    assert_eq!(g.nnz(), 9);
    assert!(g.weights.iter().all(|&w| w == 1.0));
    // empty
    let g = snn_from_knn(&knn_from(0, 20, vec![]), &SnnParams::default()).unwrap();
    assert_eq!((g.n, g.indptr.len(), g.nnz()), (0, 1, 0));
    // malformed
    assert!(snn_from_knn(&knn_from(3, 2, vec![0, 1, 1, 2, 2, 3]), &SnnParams::default()).is_err());
    assert!(snn_from_knn(&knn_from(3, 2, vec![0, 1, 1, 1, 2, 0]), &SnnParams::default()).is_err());
}

// ---------------------------------------------------------------------------- Leiden tests

fn blob_graph(sizes: &[usize], sep: f64, seed: u64) -> (SparseGraph, Vec<usize>) {
    let (x, truth) = blobs(sizes, 10, sep, seed);
    let knn = brute_knn(&x, 20);
    (snn_from_knn(&knn, &SnnParams::default()).unwrap(), truth)
}

#[test]
fn leiden_recovers_planted_blobs_of_unequal_size() {
    let sizes = [400, 300, 250, 200, 150, 100, 80, 60, 40, 30];
    for seed in [1, 2] {
        let (g, truth) = blob_graph(&sizes, 6.0, seed);
        let labels = leiden(&g, &ClusterParams::default()).unwrap();
        assert_label_contract(&labels);
        let score = ari(&labels, &truth);
        assert_eq!(n_clusters(&labels), sizes.len(), "data seed {seed}: ARI {score}");
        assert!(score > 0.99, "data seed {seed}: ARI {score}");
    }
}

#[test]
fn leiden_recovers_planted_partition_graph() {
    // weighted stochastic block model, given directly as a symmetric graph with a unit diagonal
    let sizes = [120usize, 90, 60, 40, 25];
    let n: usize = sizes.iter().sum();
    let truth: Vec<usize> = sizes.iter().enumerate().flat_map(|(c, &s)| std::iter::repeat_n(c, s)).collect();
    let mut rng = Rng(7);
    let mut rows: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
    for i in 0..n {
        rows[i].push((i as u32, 1.0));
        for j in (i + 1)..n {
            let p = if truth[i] == truth[j] { 0.3 } else { 0.01 };
            if rng.uniform() < p {
                let w = 0.1 + 0.9 * rng.uniform() as f32;
                rows[i].push((j as u32, w));
                rows[j].push((i as u32, w));
            }
        }
    }
    let mut g = SparseGraph { n, indptr: vec![0], indices: Vec::new(), weights: Vec::new() };
    for mut r in rows {
        r.sort_unstable_by_key(|e| e.0);
        g.indices.extend(r.iter().map(|e| e.0));
        g.weights.extend(r.iter().map(|e| e.1));
        g.indptr.push(g.indices.len());
    }
    let labels = leiden(&g, &ClusterParams { resolution: 1.0, ..Default::default() }).unwrap();
    assert_label_contract(&labels);
    assert_eq!(n_clusters(&labels), sizes.len());
    assert!(ari(&labels, &truth) > 0.99);
}

#[test]
fn leiden_same_seed_gives_identical_labels() {
    let (g, truth) = blob_graph(&[300, 200, 150, 100, 50], 3.0, 3);
    let p = ClusterParams::default();
    let first = leiden(&g, &p).unwrap();
    for _ in 0..3 {
        assert_eq!(leiden(&g, &p).unwrap(), first);
    }
    // overlapping blobs: the RB optimum is not the truth (leidenalg also scores ~0.85 vs
    // truth here), so other seeds are checked against each other rather than the truth
    assert!(ari(&first, &truth) > 0.8);
    for seed in 1..5 {
        let other = leiden(&g, &ClusterParams { seed, ..p.clone() }).unwrap();
        assert!(ari(&other, &first) > 0.95, "seed {seed}: ARI vs seed 0 {}", ari(&other, &first));
    }
}

#[test]
fn leiden_is_identical_across_thread_counts() {
    // large enough that local moving runs in multi-node parallel batches
    let (g, truth) = blob_graph(&[6000, 5000, 4000, 3000, 2000], 6.0, 5);
    let p = ClusterParams::default();
    let many = leiden(&g, &p).unwrap();
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap().install(|| leiden(&g, &p).unwrap());
    assert_eq!(many, one);
    assert!(ari(&many, &truth) > 0.99, "ARI {}", ari(&many, &truth));
}

#[test]
fn leiden_on_seurat_snn_agrees_with_seurat_labels() {
    let fx = SeuratFixture::load();
    let g = snn_from_knn(&fx.knn, &SnnParams::default()).unwrap();
    let labels = leiden(&g, &ClusterParams::default()).unwrap();
    assert_label_contract(&labels);
    let score = ari(&labels, &fx.labels);
    assert!(score >= 0.9, "ARI vs Seurat FindClusters(algorithm = 4): {score}");
}

#[test]
fn leiden_resolution_controls_granularity() {
    let (g, _) = blob_graph(&[300, 200, 150, 100, 50], 3.0, 4);
    let counts: Vec<usize> = [0.05, 0.5, 2.0, 8.0]
        .iter()
        .map(|&resolution| n_clusters(&leiden(&g, &ClusterParams { resolution, ..Default::default() }).unwrap()))
        .collect();
    assert!(counts.windows(2).all(|w| w[0] <= w[1]), "{counts:?}");
    assert!(counts[0] < counts[3], "{counts:?}");
}

#[test]
fn leiden_edge_cases() {
    let p = ClusterParams::default();
    let empty = SparseGraph { n: 0, indptr: vec![0], indices: vec![], weights: vec![] };
    assert!(leiden(&empty, &p).unwrap().is_empty());
    // no edges at all: every node alone, numbered by node index (all ties)
    let isolated = SparseGraph { n: 4, indptr: vec![0; 5], indices: vec![], weights: vec![] };
    assert_eq!(leiden(&isolated, &p).unwrap(), vec![0, 1, 2, 3]);
    // only the diagonal (SNN with k = 1)
    let knn = knn_from(3, 1, vec![0, 1, 2]);
    let diag = snn_from_knn(&knn, &SnnParams::default()).unwrap();
    assert_eq!(leiden(&diag, &p).unwrap(), vec![0, 1, 2]);
    // two disconnected cliques, the larger listed second: never merged, larger gets label 0
    let mut g = SparseGraph { n: 9, indptr: vec![0], indices: vec![], weights: vec![] };
    for (lo, hi) in [(0u32, 4u32), (4, 9)] {
        for _ in lo..hi {
            g.indices.extend(lo..hi);
            g.weights.extend(std::iter::repeat_n(1.0, (hi - lo) as usize));
            g.indptr.push(g.indices.len());
        }
    }
    assert_eq!(leiden(&g, &p).unwrap(), vec![1, 1, 1, 1, 0, 0, 0, 0, 0]);
}

// ------------------------------------------------------------------- external (ignored)

fn read_vec<T, const N: usize>(path: &std::path::Path, f: fn([u8; N]) -> T) -> Vec<T> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    b.chunks_exact(N).map(|c| f(c.try_into().unwrap())).collect()
}

fn json_usize(meta: &str, key: &str) -> usize {
    let at = meta.find(&format!("\"{key}\":")).unwrap_or_else(|| panic!("{key} missing")) + key.len() + 3;
    meta[at..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap()
}

fn median_secs(reps: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm-up
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64()
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[reps / 2]
}

/// Parity and timing on a dataset exported by Seurat (knn.u32, snn_*.{u32,f64}, meta.json
/// with "n" and "k"), e.g. from the bench synthetic data. Run in release:
/// `SCFAST_EXT_DIR=<dir> [SCFAST_EXT_OUT=<dir>] [SCFAST_EXT_SEEDS=10] [SCFAST_EXT_RES=0.5] cargo test --release
/// --test cluster_snn -- --ignored --nocapture`. Writes Leiden labels per diagonal variant
/// and seed to SCFAST_EXT_OUT for comparison against leidenalg.
#[test]
#[ignore = "needs SCFAST_EXT_DIR with a Seurat export"]
fn external_dataset_report() {
    let dir = std::path::PathBuf::from(std::env::var("SCFAST_EXT_DIR").expect("SCFAST_EXT_DIR"));
    let meta = std::fs::read_to_string(dir.join("meta.json")).unwrap();
    let (n, k) = (json_usize(&meta, "n"), json_usize(&meta, "k"));
    let knn = knn_from(n, k, read_vec(&dir.join("knn.u32"), u32::from_le_bytes));
    let s_ptr: Vec<usize> = read_vec(&dir.join("snn_indptr.u32"), u32::from_le_bytes).into_iter().map(|x| x as usize).collect();
    let s_idx: Vec<u32> = read_vec(&dir.join("snn_indices.u32"), u32::from_le_bytes);
    let s_w: Vec<f64> = read_vec(&dir.join("snn_weights.f64"), f64::from_le_bytes);

    let params = SnnParams::default();
    let mut g = snn_from_knn(&knn, &params).unwrap();
    let snn_s = median_secs(5, || g = snn_from_knn(&knn, &params).unwrap());
    assert_valid_symmetric(&g);
    let same_pattern = g.indptr == s_ptr && g.indices == s_idx;
    let max_diff = g.weights.iter().zip(&s_w).map(|(&a, &b)| (a as f64 - b).abs()).fold(0.0, f64::max);
    println!("n={n} k={k} nnz={} threads={} snn_median_s={snn_s:.5} seurat_pattern_identical={same_pattern} max_weight_diff={max_diff:.2e}",
        g.nnz(), rayon::current_num_threads());
    assert!(same_pattern && max_diff <= 1e-6);

    // diagonal variants seen by single-clustering: cluster() halves the diagonal, so input
    // diag 1 -> 0.5 (default), input diag 2 -> 1.0 ("full"), no diagonal -> 0 ("drop")
    let scaled = |factor: f32| {
        let mut h = g.clone();
        for i in 0..n {
            for e in h.indptr[i]..h.indptr[i + 1] {
                if h.indices[e] as usize == i {
                    h.weights[e] *= factor;
                }
            }
        }
        h
    };
    let no_diag = snn_from_knn(&knn, &SnnParams { keep_diagonal: false, ..params.clone() }).unwrap();
    let variants = [("half", g.clone()), ("full", scaled(2.0)), ("drop", no_diag)];
    let seeds: u64 = std::env::var("SCFAST_EXT_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let out = std::env::var("SCFAST_EXT_OUT").ok().map(std::path::PathBuf::from);
    let resolution: f64 = std::env::var("SCFAST_EXT_RES").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5);
    for (name, graph) in &variants {
        let p = ClusterParams { resolution, ..Default::default() };
        let mut labels = Vec::new();
        let secs = median_secs(3, || labels = leiden(graph, &p).unwrap());
        println!("leiden variant={name} seed=0 clusters={} median_s={secs:.5}", n_clusters(&labels));
        if let Some(out) = &out {
            std::fs::create_dir_all(out).unwrap();
            for seed in 0..seeds {
                let l = leiden(graph, &ClusterParams { seed, ..p.clone() }).unwrap();
                let text: String = l.iter().map(|x| format!("{x}\n")).collect();
                std::fs::write(out.join(format!("rust_{name}_seed{seed}.txt")), text).unwrap();
            }
        }
    }
}

/// RB quality at `resolution` with igraph's loop convention on the graph as given (a stored
/// diagonal entry is a full adjacency entry): `sum_c [in_c - resolution * K_c^2 / 2m]`.
fn rb_quality(g: &SparseGraph, labels: &[u32], resolution: f64) -> f64 {
    let n_c = labels.iter().copied().max().map_or(0, |m| m as usize + 1);
    let (mut internal, mut kc, mut two_m) = (0f64, vec![0f64; n_c], 0f64);
    for v in 0..g.n {
        for p in g.indptr[v]..g.indptr[v + 1] {
            let w = g.weights[p] as f64;
            two_m += w;
            kc[labels[v] as usize] += w;
            if labels[g.indices[p] as usize] == labels[v] {
                internal += w;
            }
        }
    }
    internal - resolution * kc.iter().map(|k| k * k).sum::<f64>() / two_m
}

/// Timings and RB quality on a raw CSR graph: `SCFAST_LEIDEN_GRAPH=/path/prefix` with
/// `prefix.indptr` (i64), `prefix.indices` (i32), `prefix.data` (f32);
/// `SCFAST_LEIDEN_SEEDS` (default 1); labels of seed 0 go to `prefix.labels` (u32) when
/// `SCFAST_LEIDEN_SAVE` is set.
#[test]
#[ignore]
fn external_graph_report() {
    let Some(prefix) = std::env::var_os("SCFAST_LEIDEN_GRAPH") else { return };
    let prefix = prefix.into_string().unwrap();
    let read = |ext: &str| std::fs::read(format!("{prefix}.{ext}")).unwrap();
    let indptr: Vec<usize> = read("indptr").chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as usize).collect();
    let indices: Vec<u32> = read("indices").chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as u32).collect();
    let weights: Vec<f32> = read("data").chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    let g = SparseGraph { n: indptr.len() - 1, indptr, indices, weights };
    let seeds: u64 = std::env::var("SCFAST_LEIDEN_SEEDS").ok().map_or(1, |s| s.parse().unwrap());
    for seed in 0..seeds {
        let p = ClusterParams { seed, ..Default::default() };
        let t = Instant::now();
        let labels = leiden(&g, &p).unwrap();
        let secs = t.elapsed().as_secs_f64();
        let k = labels.iter().max().unwrap() + 1;
        println!("seed {seed}: {secs:.3} s, {k} clusters, quality {:.3}", rb_quality(&g, &labels, p.resolution));
        if seed == 0 && std::env::var_os("SCFAST_LEIDEN_SAVE").is_some() {
            let bytes: Vec<u8> = labels.iter().flat_map(|l| l.to_le_bytes()).collect();
            std::fs::write(format!("{prefix}.labels"), bytes).unwrap();
        }
    }
}
