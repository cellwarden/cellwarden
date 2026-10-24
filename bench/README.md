# Benchmark harness

Times the core Seurat workflow (normalise → HVG → scale → PCA → neighbours → Leiden → UMAP)
on several engines at several dataset sizes, checks each engine's results against Seurat,
and writes a report with scaling plots. It is Phase 0 of the scfast plan: the numbers it
produces decide which steps the Rust engine should replace and where the CPU/GPU cut-off sits.
The reasoning, the parity thresholds and the adoption rule are in
[`docs/RUST_ENGINE.md`](../docs/RUST_ENGINE.md) section 8 (and section 7 for parity),
so they are not repeated here.

Run everything from the repository root.

## Quick start

PBMC 3k, using the 10x folder that `examples/tutorials/seurat_pbmc3k.R` downloads to `~/seurat-workspace`:

```bash
bash bench/run_bench.sh --input ~/seurat-workspace/filtered_gene_bc_matrices/hg19 \
    --name pbmc3k --engines seurat,scanpy --reps 3 --threads 8
```

A synthetic dataset with known cell types, subsampled into a scaling curve:

```bash
python bench/synth.py --cells 100000 --genes 5000 --types 10 --out data/synth_100k.h5
bash bench/run_bench.sh --input data/synth_100k.h5 --labels data/synth_100k.h5.labels.tsv \
    --name synth100k --sizes 10000,50000,all --threads 8 \
    --qc "--max-mt 100 --max-genes 1e9"
```

`synth.py` writes the counts as a 10x `.h5` and the true type of every cell to
`<out>.labels.tsv`. Its counts are not shaped like PBMC, so the PBMC QC thresholds can drop
cells; the `--qc` override above switches the gene-count and mitochondrial filters off.
Synthetic data is fine for timing and for testing the harness; use real data for parity.

The report ends up in `bench/results/<name>/report.md`.

## Running in a container (100k and larger)

`docker/` builds an image with a current Seurat (5.1 or later, so Leiden runs through
`leidenbase` and does not build a dense cells x cells matrix), Scanpy and the scfast wheel,
then generates a synthetic dataset and runs the harness. It has not been built in CI yet, so
the first build may need small fixes.

```bash
CELLS=100000 THREADS=8 MEM=48g bash bench/docker/run.sh                # full pipeline
CELLS=100000 THREADS=8 MEM=48g SKIP_UMAP=1 bash bench/docker/run.sh    # without UMAP
```

Variables: `CELLS`, `GENES`, `TYPES`, `THREADS`, `REPS`, `ENGINES`
(default `seurat,scanpy,scfast-cpu`), `NAME`, `TIMEOUT`, `SKIP_UMAP`, and for the container
`MEM` and `CPUS`. Results land in `bench/results/<name>/` on the host (`report.md`,
`results.jsonl`, plots). Give the container real resources and nothing else to compete with:
timings and peak memory are only comparable when `THREADS` matches the CPUs it can use.

## Your own data, with every step's output

`pipeline_steps.py` runs the workflow once and keeps each step's tables and plots in its own
folder (`01_qc` ... `07_umap`), plus `summary.md`, `timings.json`, `cluster_marker_means.csv`
and `final.h5ad`. It takes a 10x folder, `.h5` or `.h5ad`; in a multiplexed or hashing `.h5`
only the Gene Expression features are used (Antibody Capture / hashtag features are dropped).
Its QC defaults suit modern GEM-X data (no upper gene cut-off, 20% mitochondrial); change them
with `--max-genes` and `--max-mt`.

```bash
python bench/pipeline_steps.py --input sample_filtered_feature_bc_matrix.h5 --out results/run --engine scfast-cpu
INPUT=~/Downloads/sample_filtered_feature_bc_matrix.h5 MEM=16g bash bench/docker/run.sh pipeline   # in the container
```

Use the `filtered_feature_bc_matrix.h5` (cells only). A `.cloupe` file is a Loupe Browser
file and cannot be read here.

## `run_bench.sh` options

| Option | Default | Meaning |
|---|---|---|
| `--input PATH` | required | 10x folder, 10x `.h5`, or `.h5ad` with raw counts |
| `--name NAME` | required | Run name; everything goes to `bench/results/<name>/` |
| `--sizes LIST` | `all` | Comma-separated cell counts after QC, or `all`. Sizes larger than the QC'd total are skipped |
| `--engines LIST` | `seurat,scanpy,rapids,scfast-cpu,scfast-gpu` | Engines to run, in this order at each size |
| `--reps N` | `3` | Measured repeats, plus one warm-up repeat |
| `--threads N` | not set | Sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMBA_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `RAYON_NUM_THREADS`, and passes `--threads` to the runners |
| `--labels TSV` | none | Barcode and `cell_type` columns; adds ARI against known types to the parity results |
| `--timeout S` | `7200` | Seconds allowed per engine per size |
| `--qc "ARGS"` | none | Extra arguments for `prep.py`, e.g. `--qc "--max-mt 100 --max-genes 1e9"` |
| `--skip-umap` | off | Leave out the UMAP step |

Set `PYTHON` or `RSCRIPT` to use other interpreters (defaults `python3` and `Rscript`).
The script uses `mapfile`, associative arrays and GNU `timeout`, so it needs bash 4 or newer;
on a Mac, install `bash` and `coreutils` from Homebrew.

It runs, in order: `prep.py` (shared QC and subsampling), each engine at each size from
smallest to largest, `parity.py`, then `report.py`. The individual scripts can also be run on
their own; each has `--help` and a usage example at the top.

## What every engine does

`prep.py` applies the QC from `examples/tutorials/seurat_pbmc3k.R` once, in Seurat's order: keep genes detected
in at least 3 cells, then cells with more than 200 and fewer than 2,500 genes and under 5%
mitochondrial counts (`MT-` prefix). Each threshold has its own flag (`--min-cells`,
`--min-genes`, `--max-genes`, `--max-mt`, `--mt-prefix`). It then writes one Cell Ranger
v3-style `matrix.h5` per size to `data/n<cells>/`, with `cells.tsv` alongside. Smaller sizes
are nested subsets of larger ones (prefixes of one seeded permutation), and every engine reads
the same file, so all engines start from identical cells, genes and counts.

Each runner then repeats these steps, timing each one:

| Step | Seurat (`run_seurat.R`) | Scanpy / rapids (`run_py.py`) |
|---|---|---|
| `read` | `Read10X_h5` + `CreateSeuratObject` | `read_10x_h5` |
| `to_gpu` | – | rapids only |
| `normalize` | `NormalizeData`, LogNormalize, scale factor 1e4 | `normalize_total(1e4)` + `log1p` |
| `hvg` | `FindVariableFeatures`, vst | `highly_variable_genes`, `seurat_v3` flavour on counts |
| `scale` | `ScaleData` on the HVGs | `scale(max_value=10)` on the HVG subset |
| `pca` | `RunPCA` | `pca` |
| `neighbors` | `FindNeighbors` (SNN) | `neighbors` |
| `cluster` | `FindClusters(algorithm = 4)` (Leiden) | `leiden` (igraph flavour for Scanpy) |
| `umap` | `RunUMAP` (uwot) | `umap` |
| `to_cpu` | – | rapids only |

Shared parameters (runner defaults, not exposed by `run_bench.sh`): 2,000 HVGs, 30 PCs computed
and the leading 10 used for neighbours and UMAP (`--dims`, as `examples/tutorials/seurat_pbmc3k.R` does with `dims = 1:10`;
the PCs beyond the signal are a flat noise bulk whose order is arbitrary, so neighbours computed on all
30 are not comparable between engines), k = 20 neighbours including the cell itself (Seurat's `k.param`), resolution 0.5, seed 0,
2 Leiden iterations, scaled values clipped at 10.

## Engines

| Engine | Device | Needs |
|---|---|---|
| `seurat` | CPU | R packages below. The baseline every other engine is compared with |
| `scanpy` | CPU | Python packages below |
| `rapids` | GPU | `rapids-singlecell` and `cupy` as well, and an NVIDIA GPU |
| `scfast-cpu` | CPU | The Rust engine. Records `unavailable` until it is built |
| `scfast-gpu` | GPU | As above, CUDA backend |

An engine that cannot start (missing package, no GPU, scfast not built) writes one record
with `status: "unavailable"` and exits cleanly, so the run carries on. An engine that fails
during a step, or runs past `--timeout`, is not run at any larger size. A timeout is recorded
as a single record with `step: "all"`, `rep: -1` and `error: "timeout after <S>s"`.

## Requirements

- **Python**: scanpy, anndata, scikit-misc (for the `seurat_v3` HVG flavour), h5py, numpy,
  pandas, scipy, scikit-learn, matplotlib, leidenalg and python-igraph. For `rapids`, also
  rapids-singlecell and cupy.
- **R**: Seurat, hdf5r, leidenbase, jsonlite and uwot.

Seurat 5.1 and newer runs Leiden through leidenbase. Seurat 5.0.x calls Python leidenalg
through reticulate instead, so set `RETICULATE_PYTHON` to a Python that has leidenalg
(older reticulate versions also need NumPy 1.x there). The backend actually used is recorded
as `leiden_backend` in `env/seurat_<dataset>.json`.

## Output

Everything lives in `bench/results/<name>/`:

| File | Contents |
|---|---|
| `data/` | `prep.json` (input, QC thresholds, cells and genes before and after, sizes written) and one `n<cells>/` folder per size with `matrix.h5` and `cells.tsv` |
| `results.jsonl` | One record per step per repeat, appended by every runner (below) |
| `parity.jsonl` | One record per engine per size, compared with Seurat (below) |
| `outputs/<engine>/<dataset>/` | Parity outputs from the first measured repeat: `pcs.f32` (cells × PCs, float32, row-major), `knn.i32` (cells × k−1, int32, 0-based, self excluded, −1 padding), `barcodes.txt`, `labels.txt` (cluster per cell), `meta.json` |
| `env/<engine>_<dataset>.json` | Node, platform, CPU model and count, load average, thread variables, `nvidia-smi` snapshot, commit, package versions and the runner's configuration |
| `report.md` | End-to-end table with speed-up, memory ratio, parity and verdict; per-step table; start-up times; failures |
| `summary_steps.csv` | Per engine, size and step: median, min and max wall time, repeats, peak RSS and GPU memory, RSS scope |
| `summary_totals.csv` | Per engine and size: compute, read and transfer time, total, peak memory, speed-up, memory ratio |
| `scaling_time.png`, `scaling_totals.png` | Time per step, and compute time and peak memory, against cell count (log–log). Only drawn with two or more sizes |

Each `results.jsonl` record carries `schema`, `engine`, `cells`, `genes` (after QC),
`dataset`, `threads`, `commit`, `device`, `node`, `rep` and `warmup`, plus, per step:
`step`, `kind` (`io`, `compute`, `transfer` or `startup`), `status` (`ok`, `error` or
`unavailable`), `error`, `wall_s`, `transfer_s`, `peak_rss_gb`, `rss_scope` and
`peak_gpu_gb`. Each engine also writes one start-up record per size (`step: "startup"`) with
`wall_s` for loading packages (and creating the CUDA context for rapids),
`process_to_ready_s` from process start, and `peak_rss_gb`.

Each `parity.jsonl` record has `engine`, `reference`, `dataset`, `cells`, `n_pcs_compared`
(10), `pc_similarity` and `pc_similarity_min` (principal angles between the top-10 PC
subspaces), `knn_overlap`, `ari`, `n_clusters_ref`, `n_clusters`, `pass` (per metric),
`pass_all` and `status`; with `--labels`, also `ari_truth` and `ari_truth_ref`.

## Reading the numbers

- Repeat 0 is a warm-up and is left out. Reported times are medians of the measured repeats;
  compute time in the report is the sum of the median `normalize` to `umap` steps, with read
  and transfers shown separately.
- GPU steps are synchronised before the clock stops, so GPU times include the work, not just
  the launch.
- On Linux, peak RSS is reset through `/proc/self/clear_refs` before each step and read from
  `VmHWM` afterwards, so it is the peak within that step (`rss_scope: "step"`). Elsewhere,
  macOS included, it is the peak since the process started (`rss_scope: "process"`); Seurat
  records no RSS at all on macOS.
- Peak RSS includes the interpreter's own baseline (R with Seurat loaded, or Python with
  Scanpy), which matters at small sizes.
- GPU memory is the size of CuPy's memory pool after the step: an upper bound on what the step
  actually used.
- Seurat's kNN for parity is recomputed after the timed steps (`FindNeighbors` with
  `return.neighbor = TRUE`), so it does not affect the timings.
- Scanpy and rapids run their own pipelines (different HVG, graph and Leiden implementations),
  so their parity figures are informative only. The adoption rule in the report's verdict
  column is meant for scfast, and only gives `adopt` or `keep optional` at 100k cells or more.

## Fair comparisons

- Always pass `--threads`; Seurat, BLAS, numba and rayon otherwise each pick their own default.
- Run on an idle node, with nothing else of yours running.
- On a shared login node, check `loadavg` and the `gpus` snapshot in `env/*.json`, and repeat
  the run if someone else was busy.
- Use datasets of 100k cells or more for adoption decisions. PBMC 3k is useful for checking
  parity and plumbing, but start-up overhead dominates at that size.

## Tests

The harness, like the rest of the project's checks for the Rust engine, is run by hand, locally
or on the node; nothing runs it in GitHub Actions (see `docs/RUST_ENGINE.md` section 7).
