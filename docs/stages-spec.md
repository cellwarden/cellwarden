# CellWarden stages S0–S2 — interface specification v1.0

The contract for build step 2: ingest (S0), QC metrics and rules-based QC decisions (S1), and core processing with the CPU/GPU switch (S2). It sits alongside `docs/exchange-spec.md` (data hand-off between R and Python) and `docs/DESIGN.md` (architecture). Every stage reads only the files listed as its inputs and writes only its listed outputs, so stages can be rerun, resumed and tested independently.

## 1. Orchestration: built-in runner (replaces Nextflow)

With everything on one node (DESIGN v0.4), Nextflow's main benefits (spreading jobs across nodes) no longer apply, and it would add a Java dependency on every site. Stages are run by a small built-in runner, `cellwarden` (Python):

| Command | Behaviour |
|---|---|
| `cellwarden run --config config.yaml [--run-dir DIR] [--until STAGE] [--detach]` | Creates or reuses the run directory, validates inputs, runs stages in order, skips stages already completed with identical inputs |
| `cellwarden resume --run-dir DIR` | Continues from the first incomplete stage |
| `cellwarden status --run-dir DIR` | Prints stage states, timings and peak memory |
| `cellwarden doctor [--config config.yaml]` | Basic preflight: R and Python packages present, optional packages listed as available/missing, GPU detected or not, write access to the run directory |

`--detach` re-launches the run inside `tmux` (or `nohup` if tmux is missing) and prints how to reattach.

**Checkpoints.** `state.json` in the run directory records, per stage: status (`pending` / `running` / `done` / `failed`), start and end times, wall time, peak RSS, exit code, and an **input fingerprint** (SHA-256 over the config sections the stage uses plus the checksums of its input files). A `done` stage is skipped on rerun only if its fingerprint is unchanged; otherwise it and every later stage are rerun.

**Stage invocation.** Each stage is a separate process:

- R stages: `Rscript R/stages/run_stage.R --stage <ID> --run-dir <DIR>`
- Python stages: `python -m cellwarden.stages.<module> --run-dir <DIR>`

A stage exits 0 on success, non-zero on failure, and writes `logs/<STAGE>.log`. Stages read their settings from `<run-dir>/config/config.resolved.yaml`, never from the user's original file.

**Resource caps** (applied by the runner to every stage via environment variables, from `resources` in config): `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMBA_NUM_THREADS`, `CELLWARDEN_THREADS`, `CELLWARDEN_MAX_MEMORY_GB`, `CELLWARDEN_GPU_MEM_CAP_GB`. The R side calls `setThreads()` when the Seurat version provides it. Stages run under `nice -n <resources.nice>`.

## 2. Run directory layout

```
<run-dir>/
  config/
    config.yaml               copy of the user's file
    samples.csv               copy
    qc_policy.yaml            copy
    config.resolved.yaml      defaults filled in, paths made absolute, validated
    hashes.json               SHA-256 of each config file
  state.json
  logs/<STAGE>.log
  S0/object.rds               Seurat object: counts layer only, all samples merged
  S0/ingest_report.json
  S1/metrics_counts.h5ad      exchange file (full mode) for the Python QC steps; deleted when S1c is done
  S1/qc_metrics.tsv.gz        one row per cell (section 4)
  S1/qc_decisions.tsv.gz      one row per cell (section 5)
  S1/qc_summary.json
  S2/object.rds               filtered, normalised, reductions, graphs, clusters
  S2/exchange/                slim and results files when the GPU path is used
  S2/summary.json
```

Object files are `.rds` (always available). If `qs2` is installed and `object_format: qs2` is set, `.qs2` is used instead; readers accept either.

## 3. Stages

| ID | Lang | Inputs | Outputs | What it does |
|---|---|---|---|---|
| `S0` | R | `samples.csv`, input files | `S0/object.rds`, `S0/ingest_report.json` | Reads every sample, harmonises genes, merges into one Seurat v5 object |
| `S1a` | R | `S0/object.rds` | `S1/metrics_counts.h5ad`, partial `S1/qc_metrics.tsv.gz` | Per-cell QC metrics and protected-population scores; writes the counts exchange file |
| `S1b` | Python | `S1/metrics_counts.h5ad`, partial metrics | `S1/qc_metrics.tsv.gz` (complete) | Doublet scores and a quick pre-clustering of all cells |
| `S1c` | Python | `S1/qc_metrics.tsv.gz`, `qc_policy.yaml` | `S1/qc_decisions.tsv.gz`, `S1/qc_summary.json` | Rules-based QC engine (no LLM; the QC agent in build step 3 calls the same functions) |
| `S2` | R (+ Python on the GPU path) | `S0/object.rds`, `S1/qc_decisions.tsv.gz` | `S2/object.rds`, `S2/summary.json` | Filter, normalise, variable genes; then scale, PCA, integration, neighbours, clustering, UMAP on the chosen device |

### 3.1 S0 ingest

**`samples.csv`** (required columns: `sample_id`, `path`; optional `format`; any other columns become cell metadata):

**Relative `path` values are resolved against the folder that holds `config.yaml`, not the folder that holds `samples.csv`.** (The runner copies `samples.csv` into the run directory, and the resolved paths are recorded in `config.resolved.yaml` as `samples_resolved`.)

```csv
sample_id,path,format,donor,condition,batch
S1,data/S1/filtered_feature_bc_matrix.h5,,D1,control,B1
```

**Formats** (`format` column, or auto-detected from the path when blank):

| `format` | Detection | Reader |
|---|---|---|
| `10x_h5` | `.h5` file with a `matrix` group | hdf5r (Seurat::Read10X_h5) |
| `10x_mtx` | directory with `matrix.mtx(.gz)` | Seurat::Read10X (gene column 2 = symbol, column 1 = Ensembl ID) |
| `h5ad` | `.h5ad` | `read_exchange()` for exchange files; otherwise counts from `layers["counts"]`, `raw/X`, or `X` if it holds integers |
| `seurat` | `.rds` / `.qs2` | Seurat object; v4 assays converted to v5; split layers joined |
| `loom` | `.loom` | hdf5r: `/matrix` (genes × cells), `/row_attrs/Gene`, `/col_attrs/CellID` |
| `table` | `.csv`, `.tsv`, `.txt` (optionally `.gz`) | dense genes × cells table, first column gene names |

**Rules:**

- **Counts must be raw.** If a matrix has non-integer values, S0 fails with a message naming the sample and suggesting where raw counts usually live.
- **Barcodes are prefixed** with `<sample_id>_` so they are unique across samples.
- **Genes are harmonised** across samples:
  - Ensembl IDs are used as the key when every sample has them; otherwise gene symbols.
  - The original symbols go in feature metadata `gene_symbol_original`, and the IDs in `gene_id` (empty if unknown).
  - Genes missing from a sample count as zero there. Duplicate symbols within a sample are made unique with `make.unique`, and reported.
- **Organism check:** human. If fewer than 50% of the top 1,000 most-expressed genes look human (Ensembl `ENSG…` IDs or symbols matching a bundled list of human symbols), S0 warns that the data may not be human.
- **`ingest_report.json`:** per sample, the format, cells, genes, total UMIs and any warnings, plus totals and the gene-key choice.

### 3.2 S1a QC metrics (R)

Per cell: `cell_id`, `sample_id`, `n_genes`, `n_umis`, `pct_mito` (`^MT-`), `pct_ribo` (`^RP[SL]`), `pct_hb` (`^HB[ABDEGMQZ]\d*$`, excluding `HBP1`, `HBEGF` and similar non-haemoglobin genes), `log10_genes_per_umi`, and one column per protected population, `protected_<name>` = the fraction of that cell's UMIs in the population's marker genes. The protected populations come from `qc_policy.yaml`.

S1a writes `S1/metrics_counts.h5ad` (exchange full mode, counts plus log-normalised X, all cells) and a partial `qc_metrics.tsv.gz` without the S1b columns.

### 3.3 S1b doublets and pre-clustering (Python)

**Doublets:**

| `doublets.method` | Behaviour |
|---|---|
| `scrublet` (default) | Per sample; rapids-singlecell `rsc.pp.scrublet` on GPU, `scanpy.pp.scrublet` on CPU |
| `scdblfinder` | Run in R instead, if installed |
| `none` | Skip doublet detection |

Adds `doublet_score` (float) and `doublet_call` (bool).

**Pre-clustering:** a quick, coarse clustering of all cells before filtering, used only to spot QC rules that would wipe out a whole population. It runs on HVGs with 20 PCs and Leiden at resolution 0.3, reusing `cellwarden.gpu_core`, and adds `pre_cluster` (string). Pre-clusters are never used downstream.

### 3.4 S1c rules-based QC engine (Python, `cellwarden.qc`)

Public functions, reused unchanged by the QC agent in build step 3:

```python
load_policy(path) -> Policy
compute_thresholds(metrics: DataFrame, policy) -> Thresholds        # per sample
simulate_filter(metrics, thresholds, policy) -> SimulationReport     # per sample + per pre_cluster losses
check_protected(metrics, thresholds, policy) -> list[ProtectedLoss]
decide(metrics, thresholds, policy) -> Decisions                    # keep / exclude / review + reason codes
```

**`qc_policy.yaml`:**

```yaml
hard_limits:
  min_genes: 200
  max_mito_pct: 25
  min_umis: 400
adaptive:
  n_mads: 3
  metrics: [log_genes, log_umis, mito_pct]   # mito: upper bound only
  min_cells_for_mad: 100                      # fewer cells → hard limits only
doublets:
  exclude_called: true
protected_populations:
  - name: Platelet
    markers: [PPBP, PF4, GNG11]
    min_fraction: 0.01
  - name: Erythrocyte
    markers: [HBB, HBA1, HBA2]
    min_fraction: 0.05
review_triggers:
  max_cells_removed_per_sample_pct: 30
  max_cluster_removed_pct: 50
```

**Decision rules, applied in this order:**

1. **Hard limits:** a cell failing any hard limit is `exclude`. Reason codes: `LOW_GENES`, `LOW_UMIS`, `HIGH_MT`.
2. **Adaptive bounds:** per sample, median ± `n_mads` × MAD on the log scale for genes and UMIs, upper bound only for mito %. A cell failing these is `exclude`. Reason codes: `ADAPT_LOW_GENES`, `ADAPT_HIGH_GENES`, `ADAPT_LOW_UMIS`, `ADAPT_HIGH_UMIS`, `ADAPT_HIGH_MT`. Adaptive bounds never loosen a hard limit.
3. **Doublets:** a called doublet is `exclude` when `exclude_called` is set. Reason code: `DOUBLET`.
4. **Protected populations:** a cell whose `protected_<name>` ≥ `min_fraction`, and which is excluded *only* by adaptive rules (not by hard limits or doublet calls), becomes `review`. Reason code: `PROTECTED_POP:<name>`.
5. **Review triggers:**
   - If a sample would lose more than `max_cells_removed_per_sample_pct`, its adaptive exclusions become `review` (`SAMPLE_LOSS`).
   - If a `pre_cluster` would lose more than `max_cluster_removed_pct`, its adaptive exclusions become `review` (`CLUSTER_LOSS`).
6. **Everything else** is `keep`.

**`qc_decisions.tsv.gz` columns:** `cell_id`, `sample_id`, `decision` (`keep` / `exclude` / `review`), `reasons` (`;`-separated codes, empty for keep), `policy_hash`, `decided_by` (`rules` in step 2; `agent` / `human` later).

**How S2 treats `review` cells** before a gate exists (build step 3): `qc.review_action` in config, either `keep` (default; conservative, since nothing is lost) or `exclude`.

**`qc_summary.json`:** thresholds per sample; counts of keep, exclude and review per sample and per reason; protected-population losses; triggered review rules; policy hash.

### 3.5 S2 core processing

1. Subset to cells whose decision is `keep`, plus `review` cells if `review_action = keep`.
2. Normalise: `lognorm` (`NormalizeData`, scale factor 1e4) or `sct` (SCTransform v2, CPU only).
3. Variable genes: `FindVariableFeatures` with `vst` and `n_hvg` (default 2000).
4. Choose the device and memory plan (section 6).
5. **CPU path** (Seurat): `ScaleData` on HVGs, `RunPCA`, Harmony if `integration.method = harmony` and the R `harmony` package is installed (otherwise S2 fails with a clear message, or runs the GPU path if `device = auto` and the Python side has harmonypy), `FindNeighbors`, `FindClusters` with Leiden (`algorithm = 4`; uses leidenbase when available), `RunUMAP` (uwot).
6. **GPU path** (exchange): `write_exchange(slim)`, then `python -m cellwarden.gpu_core`, then `merge_exchange_results()`.
7. **Parameters**, both paths: `pcs` (`auto` or an integer; `auto` uses the elbow rule from DESIGN, minimum 10, maximum 50), `n_neighbors` (default 20), `resolution` (default 0.5), `seed`.
8. `S2/summary.json`: device and plan used and why, cells in and out, PCs, clusters, timings per step, peak memory.

## 4. `config.yaml` (sections used by S0–S2)

```yaml
project: my_study
run_dir: /scratch/$USER/cellwarden/my_study     # overridable with --run-dir
seed: 42
samples: samples.csv
qc_policy: qc_policy.yaml
object_format: rds                         # rds | qs2
resources:
  threads: 8
  max_memory_gb: 64
  gpu_mem_cap_gb: 20
  nice: 10
doublets:
  method: scrublet                         # scrublet | scdblfinder | none
qc:
  review_action: keep                      # keep | exclude (until the QC gate exists)
normalization: lognorm                     # lognorm | sct
n_hvg: 2000
device: auto                               # auto | cpu | gpu
integration:
  method: none                             # none | harmony
  batch_key: sample_id
pcs: auto
n_neighbors: 20
clustering:
  resolution: 0.5
```

Relative paths are resolved against the directory containing `config.yaml`. The runner validates types and allowed values and reports every problem at once, not just the first.

## 5. Device and memory plan

`cellwarden.resources.choose_plan(n_cells, n_hvg, device_setting, gpu_info)` returns `{device, plan, reason}`:

- `device: cpu` → CPU path.
- `device: gpu` → GPU path. Fails if no usable GPU is found (`cellwarden.gpu_core.gpu_available()`).
- `device: auto` → GPU path if a GPU is available and the estimated GPU memory (cells × n_hvg × 4 bytes × 3, for working copies) fits under the GPU memory cap; otherwise CPU path.
- `plan`: `in_memory` (fits), `managed_memory` (GPU, up to 3× the cap, using RAPIDS managed memory), or `cpu` (falls back to the CPU path). The reason is recorded in `S2/summary.json`.

## 6. Tests

- **Unit tests per module:** R testthat in `tests/R/`, pytest in `tests/python/`.
- **End-to-end:** `tests/integration/run_stages.sh` runs `cellwarden run` on the two-sample fixture (`tests/fixtures/pbmc_like`, split by its `sample` column into two inputs in two different formats) through S0–S2 twice, once with `device: cpu` and once with `device: gpu`. On this CPU-only workspace, the `gpu` run uses `gpu_core`'s CPU fallback via a test-only override `CELLWARDEN_ALLOW_CPU_AS_GPU=1`.
- **Checks:**
  - Both runs finish.
  - QC decisions are identical.
  - Cluster agreement between the two runs is ARI ≥ 0.7.
  - Clusters recover the planted cell types with purity ≥ 0.9.
  - Rerunning skips completed stages.
  - Changing `clustering.resolution` reruns only S2.
