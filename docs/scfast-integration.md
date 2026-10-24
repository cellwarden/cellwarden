# scfast integration spec

Goal: use the Rust engine (`scfast`, see [RUST_ENGINE.md](RUST_ENGINE.md)) in the pipeline wherever it
replaces Scanpy, so no Seurat ⇄ AnnData exchange is needed on that path.

## Config

Top-level `engine: auto | scfast | seurat` (default `auto`), validated in `python/cellwarden/config.py`.

| value | meaning |
|---|---|
| `scfast` | in-process Rust engine; error with install hint if `scfastR` is missing |
| `seurat` | previous behaviour (Seurat on CPU, exchange + rapids/scanpy on GPU) |
| `auto` | `scfast` if `scfastR` is installed and the device plan says CPU, else `seurat` |

`device` keeps its meaning. `auto` never selects scfast when the plan is GPU (scfast has no CUDA backend yet):
GPU plans keep rapids via exchange. `engine: scfast` with `device: gpu` is a config error.
`engine` is added to the fingerprint keys of S1a, S1b and S2.

## S2 (R, `R/stages/s2_core.R`)

New path name `cpu_scfast`, in process, no exchange files:
normalise -> HVG -> scale (Seurat) -> `RunPCAFast` -> (Harmony via R `harmony` if `integration: harmony`;
if absent, fall back to the `exchange_cpu` path with a warning) -> `FindNeighborsFast` -> `FindClustersFast`
-> `RunUMAPFast` on the PCs. Same `S2/object.rds` layout, same reductions/graph names as the Seurat path.
`S2/summary.json` gets `path: "cpu_scfast"` and `engine: "scfast"`, plus `scfast_version` from `scfast_info()`.

## S1b doublets and pre-clustering

Python `scrublet`/scanpy and `S1/metrics_counts.h5ad` are replaced when the engine is scfast:
S1a (R) computes doublet scores and `pre_cluster` itself with a scrublet-style detector
(`R/stages/doublets_scfast.R`: per sample, simulate doublets by summing random cell pairs, log-normalise,
HVG, `RunPCAFast`, kNN via `scfast_knn`, score = fraction of simulated neighbours with a threshold from
the bimodal score distribution; pre-cluster = PCA -> kNN -> SNN -> Leiden res 0.3, seed fixed).
It writes `doublet_score`, `doublet_call`, `pre_cluster` straight into `S1/qc_metrics.tsv.gz`,
S1b becomes a pass-through (`method` recorded, nothing to do) and no `.h5ad` is written. S1c is unchanged.
`doublets.method: scrublet` still works with `engine: seurat`.

## gpu_core

`python -m cellwarden.gpu_core --engine {rapids,scanpy,rust}` (default unchanged). `rust` uses `scfast`
(Python) for PCA, kNN, SNN, Leiden; UMAP still uses `umap-learn` / rapids. For the Python/exchange route only.

## Install and checks

- R: `R CMD INSTALL rust/scfast-r` (needs cargo >= 1.80). Python: `maturin develop --release -m rust/scfast-py/Cargo.toml`.
- `cellwarden doctor` reports scfastR / scfast presence and version, and the engine that `auto` would choose.
- `containers/` recipes install the Rust toolchain and both packages.

## Tests

- Unit: config validation, engine resolution, summary fields.
- Integration (skip cleanly when `scfastR` is absent): `run_stages.sh` variant with `engine: scfast`
  asserts no `*.h5ad` under the run dir, clusters vs planted types ARI >= the Seurat path - 0.05,
  doublet recall on planted doublets, and the full `run_full.sh` passes with `engine: scfast`.

## Phases 6–9 (GPU backend, scale runs, fused preprocessing)

Constraint: the development workspace has no GPU and no CUDA toolkit. Everything below must
**compile without CUDA installed** and be **verifiable on the Iridis X L4**; GPU parity tests are
`#[ignore]` locally and run on the node with `cargo test --features cuda -- --ignored`.

### Phase 6: `scfast-core` feature `cuda` (cudarc, dynamic loading)

- `scfast-core/src/device.rs`: `enum Device { Cpu, Cuda(usize) }`, `Device::auto()` = CUDA if the
  driver loads and a device is visible, else CPU; `device_info()` for R/Python (`scfast_info()` gains
  `cuda_built`, `cuda_available`, `device_name`, `free_gb`).
- `scfast-core/src/cuda/`: `cudarc` with `dynamic-loading` (no link-time dependency on libcuda).
  Kernels as CUDA C strings compiled with NVRTC at first use: CSR SpMM (`X·V` and `Xᵀ·U` for the
  implicit scaled operator, with the mean/inv-sd correction applied on device), top-k row selection
  for kNN; cuBLAS GEMM for the dense products. Randomised SVD's small dense steps (QR, SVD of the
  k×k core) stay on CPU via faer.
- `pca::implicit_scaled_pca` and `knn::knn` take `device: Device`; CUDA results must match CPU to
  the Phase 3/4 thresholds (subspace similarity > 0.99, kNN overlap ≥ 90%). Out-of-memory → clear
  `Error::Gpu`, callers fall back to CPU and say so.
- R `RunPCAFast/FindNeighborsFast(device="auto"|"cpu"|"cuda")`, Python `scfast.pca/knn(device=)`.

### Phase 7: GPU Leiden via cuGraph C API (feature `cugraph`)

- `scfast-core/src/cuda/cugraph.rs`: `libloading` of `libcugraph_c.so` at runtime; hand-written
  `extern "C"` declarations for `cugraph_create_resource_handle`, `cugraph_sg_graph_create` (COO
  src/dst/weights), `cugraph_leiden`, result accessors, `cugraph_free_*`. `CugraphLeiden`
  implements `Clusterer`; resolution, theta=1.0, seed → RNG state, `max_level = n_iterations`.
- ARI vs CPU Leiden ≥ 0.9 on the same graph (test ignored locally).
- `FindClustersFast(device=)`, `scfast.leiden(device=)`.

### Pipeline wiring

- `engine: auto` on a GPU plan: `gpu_scfast` (S2 in R, in process, device cuda) when `scfastR` was
  built with `cuda` **and** a device is visible; else `gpu_exchange` (rapids) as now. New config
  `engine_gpu: auto|scfast|rapids` (default `auto`) to force either. `S2/summary.json` records
  `path: gpu_scfast`, `device: gpu`, `gpu_core_device: cuda`, `scfast_device`.
- S1a doublets: same device choice. UMAP runs on the CPU (`RunUMAPFast`) on the scfast GPU path.
- `gpu_core --engine rust --device gpu` uses scfast CUDA when available.
- `cellwarden doctor`: `scfastR cuda built / device visible` lines.

### Phase 8: scale runs on the L4

`bench/hpc/`: `make_scale_inputs.py` (50k, 100k, 200k, 500k synthetic, refused outside HPC unless
`CELLWARDEN_ALLOW_LARGE=1`), `run_scale.sh` (Seurat CPU, scfast CPU, scfast CUDA, rapids exchange; each
S0–S2 with `cellwarden run --until S2`), `slurm_scale.sh`, `summarise.py` → `bench/results/scale/`
table (time, peak RSS, peak GPU memory, ARI vs Seurat) and the adoption verdict per size against
the rule in RUST_ENGINE.md §8.2. Not run here.

### Phase 9: fused preprocessing (CPU, testable here)

`scfast-core/src/preprocess.rs`: one pass over CSR counts per chunk of cells computing library
size, log1p-normalised values on the fly, and vst HVG statistics (mean, variance, standardised
variance with the loess fit done on the per-gene summaries), returning HVG indices and per-gene
mean/sd for the implicit PCA so `NormalizeData + FindVariableFeatures + ScaleData` become one call.
R `PreprocessFast(object, n_hvg)` / Python `scfast.preprocess(X_counts, n_hvg)`. Done when HVG
overlap with Seurat `FindVariableFeatures(selection.method="vst")` ≥ 95% on the fixture and PBMC-like
data, and the S2 scfast path uses it (`timings_s.preprocess` replaces normalize+hvg+scale).
