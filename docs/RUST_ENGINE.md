# scfast: Rust engine for the singlecell pipeline — plan

Status: **phases 0–7 and 9 implemented** (see [scfast-integration.md](scfast-integration.md)); the CUDA backend (6) and cuGraph Leiden (7) compile and pass CPU-side tests but are **not yet validated on a GPU**; phase 8 (scale runs, `bench/hpc/`) waits for the L4 · Version 0.5 · 17 Oct 2026
Scope: plan; the CPU crates exist under `rust/`. Companion to [`DESIGN.md`](DESIGN.md) and [`exchange-spec.md`](exchange-spec.md).

## 1. Goal

Make the core Seurat workflow as fast and memory-lean as possible by moving its numerical core into a Rust library, **scfast**, with a CPU backend that runs anywhere (including a Mac) and a CUDA backend for the L4, A100 and H200 nodes.

Seurat stays the API. Objects stay Seurat objects. scfast writes standard `DimReduc` and `Graph` objects back into them, so `RunUMAP`, `FindAllMarkers`, plotting and everything downstream work unchanged. The exchange format and `gpu_core.py` keep working alongside it.

### In scope

| Step | Replaces | Why it matters |
|---|---|---|
| Implicitly scaled PCA | `ScaleData` + `RunPCA` | `ScaleData` builds a dense cells × HVGs matrix (about 8 KB per cell at 2,000 HVGs, DESIGN §5.1). Implicit scaling never builds it |
| kNN on PCs | Annoy in `FindNeighbors` | Exact on GPU, parallel on CPU |
| SNN graph (Jaccard + prune) | `FindNeighbors` SNN step | Same graph Seurat clusters on |
| **Leiden clustering** | `FindClusters` | Via **single-clustering** on CPU (section 4), cuGraph on GPU |
| Fused QC / normalise / HVG statistics | `PercentageFeatureSet`, `NormalizeData`, `FindVariableFeatures` (vst) | One streaming pass instead of several copies. Later phase |

### Out of scope (for now)

UMAP (uwot and rapids are already fast), marker testing (presto is already fast), SCTransform, anchor integration, WNN, spatial. Harmony runs in scfast (`integrate::harmony`, CPU; harmony 2.x algorithm), with R harmony / rapids as fallbacks. These can follow once the core is proven.

## 2. Architecture

```
          Seurat (R)                                cellwarden (Python)
              │ extendr                                 │ PyO3 / maturin
              ▼                                         ▼
  ┌──────────────────────────── scfast (Rust) ────────────────────────────┐
  │  Backend trait ── CpuBackend   : rayon + faer          (any machine)  │
  │                └─ CudaBackend  : cudarc → cuSPARSE, cuBLAS, NVRTC     │
  │  Clusterer trait ─ SingleClusteringLeiden (CPU)                       │
  │                  └ CugraphLeiden (GPU, FFI to libcugraph_c)           │
  │  Kernels: implicit PCA · kNN · SNN · (later) fused preprocessing      │
  └────────────────────────────────────────────────────────────────────────┘
              │ embeddings, graphs, labels written back as Seurat objects
              ▼
  RunUMAP · FindAllMarkers (presto) · plots · exchange format · agents
```

### Repository layout

```
rust/
  Cargo.toml             workspace; pins single-clustering = "=0.7.0"
  scfast-core/           CSR views, LinOp, randomised SVD, kNN, SNN,
                         Backend + Clusterer traits, CPU implementations
  scfast-cuda/           feature "cuda": cudarc kernels, cuGraph Leiden FFI
  scfast-r/              extendr R package (scfastR)
  scfast-py/             PyO3 module (scfast), built with maturin
bench/                   timing and parity harness (R and Python), JSON output
tests/rust/              parity tests against Seurat / leidenalg outputs
docs/RUST_ENGINE.md      this document
```

### Core traits

```rust
pub trait Backend {
    type Csr;    // device-resident sparse matrix (cells × genes)
    type Dense;  // device-resident dense matrix
    fn upload_csr(&self, m: CsrView<f32>) -> Self::Csr;
    fn randomized_svd(&self, op: &dyn LinOp<Self>, k: usize, seed: u64) -> Svd<Self::Dense>;
    fn knn(&self, x: &Self::Dense, k: usize) -> KnnGraph;   // indices + distances
    fn download(&self, d: &Self::Dense) -> Mat<f32>;
}

pub trait Clusterer {
    fn cluster(&self, g: &SnnGraph, p: &ClusterParams) -> Result<Vec<u32>>;
}

pub struct ClusterParams {
    pub resolution: f64,   // pipeline default 0.5
    pub seed: u64,         // pipeline default 0
    pub n_iterations: usize,
}
```

`CpuBackend` is the reference implementation. Every GPU result is tested against it.

## 3. Data flow and copies

- **R → Rust:** Seurat's `data` layer is a `dgCMatrix` of genes × cells (CSC). Its `p`, `i`, `x` arrays are exactly the CSR arrays of cells × genes (exchange-spec §1 already relies on this), so scfast reads them as a borrowed `CsrView` with no transpose and no copy.
- **Python → Rust:** scipy CSR `indptr`, `indices`, `data` are borrowed through PyO3 numpy views.
- **Types:** `u32` indices and `f32` values internally. Results return as `f32` (embeddings) and `f64` (graph weights, for Seurat's `Graph` class).
- **Rust → R:** PCA returns cell embeddings, loadings and standard deviations as a `DimReduc` named `pca` (or a name passed by the caller). The SNN returns as a `Graph` named `RNA_snn`. Clusters go into `seurat_clusters` and `Idents()`, matching what `FindClusters` writes.

## 4. Clustering: single-clustering

### Decision

CPU Leiden uses **single-clustering** (SingleRust), version **pinned to 0.7.0**, with the **RB objective at resolution 0.5**. RB (Reichardt–Bornholdt with a configuration null model) is the objective Seurat's Leiden (`algorithm = 4`, `RBConfigurationVertexPartition`) and Scanpy's Leiden optimise, and 0.5 is the resolution used in `seurat_pbmc3k.R` and `gpu_core.py`.

GPU Leiden uses **cuGraph's C API** (`cugraph_leiden`) through FFI, behind the same `Clusterer` trait. No native Rust GPU Leiden exists.

### Evidence (17 Oct 2026, synthetic kNN-style graph, 10 planted clusters of unequal size, 2-core sandbox)

| Nodes | Crate | Leiden time | Clusters | Modularity | ARI vs truth |
|---|---|---|---|---|---|
| 3k | single-clustering 0.7.0 | 0.03 s | 10 | 0.742 | 1.00 |
| 3k | leiden-rs 0.8.1 | 0.04 s | 10 | 0.742 | 1.00 |
| 100k | single-clustering 0.7.0 | 2.96 s | 10 | 0.742 | 1.00 |
| 100k | leiden-rs 0.8.1 | 3.22 s | 6 | 0.635 | 0.68 |

leiden-rs merged clusters at 100k nodes with the same seed and resolution, so it was ruled out. graphrs was ruled out because it has no seed parameter. Still to confirm in Phase 0: the 1M-node run, a comparison against leidenalg on the same graph, and results on real PBMC graphs (the synthetic graph is easier than a real one).

### Parameter mapping

| Pipeline / Seurat | single-clustering `LeidenConfig` | Value |
|---|---|---|
| `resolution = 0.5` | `objective: ObjectiveKind::Rb { resolution }` | 0.5 |
| `random.seed` / `--seed` | `seed: Some(seed)` | 0 |
| `n.iter` / `n_iterations = 2` | `n_iterations` | 2 |
| Leiden (not Louvain) | `refine: true` | true |
| leidenalg randomness | `refine_randomness` | 0.01 (crate default) |
| — | `patience`, `tolerance`, `max_levels` | crate defaults |

### Graph hand-off

single-clustering needs the **full symmetric adjacency** in CSR via `CSRNetwork::from_csr_parts(node_ptrs, neighbors, weights, None)`. The SNN kernel produces exactly that (both directions of every edge, `u32` neighbours, `f32` weights). The constructor takes owned `Vec`s, so there is one copy of the graph per clustering call; at k = 20 that is small next to the expression matrix.

### Known limits

- Single-threaded. Its README describes parallel local moving as deliberately deferred. At 1M cells, expect seconds to tens of seconds, which is acceptable because Leiden isn't the main bottleneck once PCA and kNN are fixed.
- The API is still changing (0.7.0 was a breaking rewrite), hence the exact version pin.
- Works in `f32`, so quality values match leidenalg only to `f32` precision. Parity is judged by ARI, not identical labels.

## 5. Interfaces

### R (scfastR, via extendr)

```r
pbmc <- NormalizeData(pbmc) |> FindVariableFeatures(nfeatures = 2000)
pbmc <- scfastR::RunPCAFast(pbmc, npcs = 20, seed = 0, device = "auto")      # no ScaleData
pbmc <- scfastR::FindNeighborsFast(pbmc, dims = 1:10, k.param = 20, prune.SNN = 1/15)
pbmc <- scfastR::FindClustersFast(pbmc, resolution = 0.5, seed = 0, n.iter = 2)
pbmc <- RunUMAP(pbmc, dims = 1:10)                                          # unchanged
```

Argument names follow Seurat's own, so switching is a one-word change per call.

### Python (scfast, via PyO3)

```python
import scfast
pcs, loadings, sdev = scfast.pca(X_csr, n_comps=30, seed=0, device="auto")
knn_idx, knn_dist   = scfast.knn(pcs, k=15, device="auto")
snn                 = scfast.snn(knn_idx, prune=1/15)            # scipy CSR
labels              = scfast.leiden(snn, resolution=0.5, seed=0, n_iterations=2)
```

`gpu_core.py` gains `--engine {rapids,scanpy,rust}`. The exchange contract (slim in, results out) does not change.

## 6. Build plan

Each phase is one pull request into `feat/rust-engine`, and is only merged when its "done when" check passes.

| Phase | Work | Done when |
|---|---|---|
| **0. Baseline** | `bench/` harness (section 8): per-step time, peak RSS, peak GPU memory, parity outputs, written as JSONL. Baselines for Seurat CPU, Scanpy and rapids `gpu_core` on PBMC 3k, the 500-cell fixture and one large dataset. Finish the Leiden checks from section 4 | Baseline `results.jsonl` and report committed under `bench/results/`; single-clustering ARI vs leidenalg ≥ 0.9 on PBMC 3k |
| **1. Scaffold** | Cargo workspace, `Backend` and `Clusterer` traits, `CsrView` borrowed from R and Python, no-op round trip | `cargo test` passes; R and Python can call into scfast |
| **2. Leiden (CPU)** | `SingleClusteringLeiden` behind `Clusterer`; `FindClustersFast` and `scfast.leiden` taking Seurat's existing `RNA_snn` graph | Same SNN graph, ARI vs Seurat `algorithm = 4` ≥ 0.9 on PBMC 3k; same seed gives identical labels |
| **3. Implicit PCA (CPU)** | Randomised SVD on the implicit scaled operator, faer for dense steps | Top 10 PC subspace similarity > 0.99 vs Seurat; no dense cells × HVG matrix allocated |
| **4. kNN + SNN (CPU)** | Parallel exact kNN on PCs; SNN with Jaccard weights and Seurat's prune rule | kNN overlap with Seurat ≥ 90%; end-to-end cluster ARI ≥ 0.9; the nine PBMC cell types still found |
| **5. Integration** | `RunPCAFast`, `FindNeighborsFast`, `FindClustersFast` in `seurat_pbmc3k.R` as an option; `gpu_core --engine rust` | Full PBMC 3k run produces the same 9 annotated cell types; bench shows time and memory vs baseline |
| **6. CUDA PCA + kNN** | `CudaBackend`: cuSPARSE SpMM for the implicit operator, cuBLAS GEMM + top-k kernel for kNN | Same parity checks with `device = "cuda"` on the L4 |
| **7. GPU Leiden** | `CugraphLeiden` via FFI to `libcugraph_c` (`cugraph_leiden` with resolution, theta, RNG state) | ARI vs CPU Leiden ≥ 0.9 on the same graph |
| **8. Scale runs** | Large dataset on the L4: Seurat vs scfast CPU vs scfast CUDA vs rapids | Results table in the README |
| **9. Fused preprocessing** (stretch) | One-pass QC, normalise, log1p and vst HVG statistics, streamed in chunks | HVG overlap with Seurat ≥ 95% |

Phases 2 and 3 are independent and can run in parallel. Phase 2 comes first in the order because it plugs into Seurat's existing SNN graph and gives a usable result on day one.

## 7. Parity and testing

Following the decision not to use GitHub Actions, tests run locally and on the node through one script, `tests/run_all.sh`, which runs the existing exchange tests and adds:

| Check | Threshold | Reference |
|---|---|---|
| PC subspace (principal angles, top 10 PCs) | similarity > 0.99 vs `RunPCA` run with `scale.max = Inf`; signal PCs correlate ≥ 0.999 | Seurat `RunPCA` |
| kNN overlap (k = 20) | ≥ 90% | Seurat `FindNeighbors` |
| Cluster ARI at resolution 0.5 | ≥ leidenalg's own seed-to-seed ARI on the same graph, and partition quality ≥ leidenalg's | Seurat `FindClusters(algorithm = 4)` and leidenalg |
| Determinism | identical labels for the same seed, CPU, with exact kNN (HNSW is not bit-reproducible: its library seeds its own RNG) | Previous run |
| CPU vs CUDA | ARI ≥ 0.9; PC similarity > 0.99 | `CpuBackend` |
| Cell types | all 9 PBMC types recovered | `seurat_pbmc3k.R` |

Datasets: the 500-cell fixture (`tests/fixtures/pbmc_like`, fast, four known cell types), PBMC 3k, and one dataset of 100k cells or more for scale. Results will not be identical to Seurat (f32 vs f64, exact vs Annoy kNN, randomised SVD); the thresholds above define "correct".

Findings from Phase 1-4 (17 Oct 2026) that shaped these thresholds:

- **PCA:** Seurat's `scale.max = 10` clipping reorders noise PCs; even an exact unclipped SVD reaches only 0.98-0.99 top-10 similarity vs clipped `RunPCA`. The implicit operator cannot clip, so parity is judged against unclipped Seurat, where scfast reaches 0.9998 (fixture), 0.994 (2k) and 0.986 (5k) with the defaults oversample 30 / power_iters 10; the signal PCs correlate at 0.9995 or better everywhere. The shortfall is in near-degenerate noise PCs.
- **kNN:** exact search is bit-identical to a naive reference; Seurat's Annoy neighbours overlap it 99% (2k) and 98% (5k), so the 90% bar has margin.
- **SNN:** bit-identical to Seurat's `ComputeSNN` on the fixture, synthetic 2k/5k and PBMC 3k.
- **Leiden:** on PBMC 3k, leidenalg run Seurat's way agrees with itself across seeds at ARI 0.86, so a fixed 0.9 bar vs Seurat's labels was unattainable; scfast scores 0.87 vs Seurat with higher seed-to-seed stability (0.92) and equal or better quality. On the fixture and synthetic data ARI is 1.0. The SNN diagonal is passed at half weight so the objective matches the directed graph Seurat gives leidenalg (see `cluster.rs`).

## 8. CPU/GPU split and benchmarking

### 8.1 What runs where

Rule of thumb: **GPU for heavy, regular arithmetic on data that can stay on the GPU; CPU for I/O, irregular graph work, small data and R-ecosystem steps.** Every CPU⇄GPU copy costs time, so the target is one upload (the HVG matrix) and one download (PCs, graph, labels).

| Step | Device | Engine | Why |
|---|---|---|---|
| Read 10x / h5 / h5ad | CPU | Rust or R | Limited by disk, not compute |
| QC metrics | CPU | Rust, fused pass | Simple sums; uploading the full raw matrix costs more than computing them |
| Doublets (scDblFinder) | CPU | R | Out of scope |
| Normalise, log1p, HVG statistics | CPU | Rust, one streaming pass | Works on all genes; only the HVG subset (roughly 5–25% of the data) is worth uploading |
| **Implicitly scaled PCA** | **GPU** | Rust CUDA (cuSPARSE + cuBLAS); CPU fallback | The largest GPU win: sparse and dense matrix products |
| Harmony | CPU (GPU next) | scfast `integrate::harmony` | harmony 2.x algorithm; 6-40x faster than R harmony |
| **kNN** | **GPU** | Rust CUDA (GEMM + top-k); CPU fallback | Brute-force kNN is matrix multiplication, and it's exact |
| SNN graph | CPU first, GPU later | Rust | Light either way; moves to GPU only to avoid a copy once kNN is there |
| **Leiden** | **CPU** | single-clustering | Irregular graph work on a small graph (about 20 edges per cell). GPU (cuGraph) only for multi-million cells |
| UMAP | GPU (CPU: uwot) | rapids | Already fast; not worth rewriting |
| Markers, pseudobulk DGE, plots, agents | CPU | R / Python | Not compute-bound, or already fast (presto) |

In Rust, the GPU work is **PCA and kNN**, with SNN and Leiden as later options. Small datasets don't benefit from the GPU because of start-up and transfer costs, so each GPU step falls back to CPU below a cell-count threshold. The threshold is set from the scaling curve (8.2), not guessed; 50k–100k cells is the starting estimate.

### 8.2 How performance is judged

The harness in `bench/` (Phase 0) implements everything below.

**Per-step measurements**, for every engine and dataset size:

- Wall time: median of the measured repeats after one warm-up repeat.
- Cold start, recorded separately: R package loading, Python imports and CUDA context creation.
- Transfer time between CPU and GPU, recorded separately.
- Peak host memory (RSS) per step, and peak GPU memory.
- Throughput in cells per second.

GPU work runs asynchronously, so every GPU step is timed only after the device has finished (device synchronise or CUDA events).

**Correctness alongside speed.** Every run saves PCs, kNN indices and cluster labels, and the parity checks in section 7 are computed against the Seurat run at the same size. A faster step that fails parity doesn't count.

**Fair comparisons:**

- Same machine, same cells and genes after QC (one shared QC step), same parameters (resolution 0.5, seed 0, same PCs and neighbours).
- Thread count fixed and recorded, because Seurat, BLAS and rayon each pick their own defaults.
- System load and `nvidia-smi` recorded with each run, since the L4 login node is shared; repeat runs if another user is active.
- Precision noted: scfast uses f32, Seurat f64.

**Engines compared:** Seurat CPU (baseline), Scanpy CPU, rapids-singlecell GPU (`gpu_core.py`), scfast CPU, scfast GPU.

**Scaling curve:** one large dataset subsampled to 10k, 50k, 100k, 250k, 500k and 1M cells. Time and memory against cell count, per step and engine, shows where the GPU starts to win (the fallback threshold), where Seurat runs out of memory, and any step that scales worse than expected.

**Profiling** when a number needs explaining: `cargo flamegraph` / `perf` for Rust, `profvis` for Seurat, NVIDIA Nsight Systems (`nsys`) for kernel-versus-copy time on the GPU.

**Adoption rule:** a Rust step replaces the existing one only if parity passes **and**, at 100k cells or more, it is at least 1.5× faster or uses at least 2× less peak memory than the Seurat baseline. Otherwise it stays optional.

### 8.3 Record format

One JSON object per line (`results.jsonl`), one line per step per repeat:

```json
{"engine": "scfast-gpu", "step": "pca", "cells": 250000, "genes": 2000, "rep": 1, "warmup": false,
 "wall_s": 1.84, "transfer_s": 0.31, "peak_rss_gb": 3.1, "peak_gpu_gb": 2.4, "threads": 8,
 "status": "ok", "node": "iridisx-l4", "commit": "29b6e58"}
```

`bench/report.py` turns these into the results table (median times, speed-up and memory ratio against Seurat, parity, verdict) and the scaling plots.

## 9. Build environment

| Where | What builds | Notes |
|---|---|---|
| Mac (Apple Silicon) | CPU backend, R and Python bindings | Default features only |
| Iridis X L4 node | CPU + `--features cuda` | `rustup` installs in the home directory without root; CUDA 12 from the container |
| Container | Everything | Add the Rust toolchain to the Apptainer recipe (DESIGN §7.5) so extendr can build inside the image |

The `cuda` feature is off by default, so a machine without CUDA always builds. At runtime, `device = "auto"` picks CUDA when a device is visible and falls back to CPU otherwise, mirroring `gpu_core._resolve_device`.

## 10. Risks

| Risk | Mitigation |
|---|---|
| single-clustering API changes | Exact version pin; all calls go through the `Clusterer` trait, so swapping crates touches one file |
| single-clustering diverges from leidenalg on real graphs | Phase 0 check before any integration; fall back to leidenalg via R if ARI < 0.9 |
| Single-threaded Leiden too slow at multi-million cells | cuGraph Leiden on GPU (Phase 7); Leiden is not the main bottleneck after PCA and kNN |
| CUDA build problems on Iridis | Test `cargo build --features cuda` on the node early, before Phase 6 |
| extendr build friction in the container | Build the R package once into the image, not at run time |
| Results differ from Seurat | Thresholds defined up front; differences reported, not hidden |
| Scope creep toward a full Seurat port | In/out scope fixed in section 1; new steps need a bench result showing the gain |

## 11. Open questions

1. Which large dataset to use for scale runs (100k+ human cells closest to the real data, or the 10x 1.3M mouse brain)?
2. Is building and running Rust on the Iridis X L4 node permitted, and can the container recipe include the Rust toolchain?
3. Should `FindClustersFast` be the default in the pipeline once Phase 2 passes, or stay opt-in until Phase 5?
