# Tests

Two contracts are tested: the exchange format between Seurat (R) and AnnData (Python),
defined in [`docs/exchange-spec.md`](../docs/exchange-spec.md), and pipeline stages S0–S2,
defined in [`docs/stages-spec.md`](../docs/stages-spec.md).
Run everything from the repository root.

| Layer | Where | What it checks |
|---|---|---|
| Fixtures | `tests/fixtures/` | A small synthetic dataset with pinned checksums, and a QC-stress, two-sample input built from it |
| R unit tests | `tests/R/` | `R/exchange/exchange.R` and `R/stages/` on their own |
| Python unit tests | `tests/python/` | `python/cellwarden/` on its own |
| Round-trip | `tests/integration/run_roundtrip.sh` | R and Python reading each other's files, end to end |
| Stages S0–S2 | `tests/integration/run_stages.sh` | `cellwarden run` through S2 on the CPU and GPU paths, compared with each other and with the planted truth |
| Full pipeline | `tests/integration/run_full.sh` | `cellwarden run` through S5 with a mock LLM and offline Open Targets: gate files, S3/S4 outputs, `report.html` and `explorer.html` |

## Requirements

- R 4.3 or newer with Seurat 5, SeuratObject, hdf5r, Matrix, jsonlite, digest, yaml and testthat
  (leidenbase and uwot are used by S2 when present)
- Python 3.12 or newer with anndata, scanpy, h5py, scipy, pandas, pyyaml, pytest, igraph,
  leidenalg and harmonypy

No GPU is needed: `gpu_core` runs with `--device cpu` in the round-trip, and the stage test's
GPU run uses `gpu_core`'s CPU fallback (see below). The optional R packages harmony, presto,
scDblFinder, BPCells and qs2 are not needed by any test.

Inside the CellWarden image, clear the image's own settings first, since they change paths the tests
check (`CELLWARDEN_REPO_ROOT` is read when `cellwarden` is imported):

```bash
env -u CELLWARDEN_REPO_ROOT -u CELLWARDEN_IN_CONTAINER -u CELLWARDEN_CELLTYPIST_MODELS PYTHONPATH=$PWD/python python -m pytest tests/python
```

## Fixture

`tests/fixtures/pbmc_like/` holds a seeded synthetic dataset of 500 cells and 800 genes
in 10x MatrixMarket form (`matrix.mtx` genes × cells, `features.tsv`, `barcodes.tsv`),
cell annotations in `cells.tsv` (two samples, four cell types with their own marker genes),
and `expected.json`, which pins:

- `counts_checksum` (spec section 6), computed by an implementation inside
  `make_fixture.py` that does not use CellWarden
- `cells_sha256`, plus `genes_sha256_original` and `genes_sha256_seurat` (spec section 4),
  for the original names and for the names after Seurat turns `_` into `-`
- the order of the factor levels, the genes with underscores, the MT- genes and the marker genes

The gene names include underscores, dashes, MT- genes and look-alike names such as
`TBCE` / `TBCE.1` / `TBCE-AS1`. They stay unique after Seurat renames them.

```bash
python tests/fixtures/make_fixture.py           # regenerate (only if you change the generator)
python tests/fixtures/make_fixture.py --check   # regenerate in a temp dir and compare byte for byte
python tests/fixtures/make_fixture.py --verify  # recompute expected.json from the committed files
```

The committed files are what the tests use. If you change the generator, regenerate the files
and commit them together with it.

## Rust engine tests

The `scfast` tests skip cleanly when `scfastR` / `scfast` are not installed. To run them, build
and install the packages first (`R CMD INSTALL rust/scfast-r`, `maturin develop --release -m
rust/scfast-py/Cargo.toml`; see [`docs/INSTALL.md`](../docs/INSTALL.md)). With `scfastR` installed,
`engine: auto` selects the Rust path, so `run_full.sh` exercises it; `run_stages.sh` pins
`engine: seurat` because it compares the Seurat and exchange paths with each other. To use a
private R library, set `R_LIBS`.

## R unit tests

```bash
Rscript -e 'testthat::test_dir("tests/R")'
# or a single file
Rscript -e 'testthat::test_file("tests/R/test-exchange.R")'
```

## Python unit tests

```bash
PYTHONPATH=python python -m pytest -q tests/python
```

## Round-trip

```bash
tests/integration/run_roundtrip.sh              # temp workdir, removed if everything passes
tests/integration/run_roundtrip.sh /tmp/rt      # keep the .h5ad files and logs in /tmp/rt
KEEP_WORKDIR=1 tests/integration/run_roundtrip.sh
```

Set `RSCRIPT` or `PYTHON` to use other interpreters. Each step's log goes to `<workdir>/logs/`.

| Step | Script | What it does |
|---|---|---|
| S0 | `make_fixture.py --check` | The fixture is reproducible |
| S1 | `roundtrip.R write` | Loads the fixture into Seurat, runs NormalizeData and FindVariableFeatures (500 genes), checks `sc_counts_checksum` against `expected.json`, writes `r_full.h5ad` and `r_slim.h5ad`, and checks that `validate_exchange` returns no problems |
| S2 | `roundtrip.py check-r-files` | Reads both files with `anndata.read_h5ad` (any warning fails the step) and with `cellwarden.exchange.read_exchange`. Checks the encodings, `uns["exchange"]`, the counts checksum, that the slim file holds only the variable genes, that factor levels are kept, and that `var["gene_symbol_original"]` keeps the underscores |
| S3 | `python -m cellwarden.gpu_core ... --device cpu` | Turns the slim file into `results.h5ad` |
| S3b | `roundtrip.py check-results` | Checks the results file (`obsm`, `obsp`, cluster column, and that the clusters match the planted cell types). Builds the files for the refuse cases: one with an altered source checksum, and one missing a barcode |
| S4 | `roundtrip.R merge` | Merges the results into the original object. Checks the reductions, keys, graphs, clusters in `meta.data` and `Idents()`, and the cell order, then merges into a shuffled object to check matching by barcode. Refuse cases: merging twice without `overwrite`, a wrong checksum, a missing barcode (accepted with `allow_subset = TRUE`), and an unknown barcode |
| S5a | `roundtrip.py write-python` | Python builds a full file from the fixture (counts plus log1p `X`) with `write_exchange` |
| S5b | `roundtrip.R read-python` | `read_exchange_seurat` reads the Python file. Checks the counts checksum, factor levels, column types, feature metadata and variable features |

The script prints a PASS/FAIL table with timings at the end. It exits with a non-zero code if any
step fails, or is skipped because a step it depends on failed.

## Stages S0–S2

### Stage input fixture

`tests/fixtures/make_stage_inputs.py` turns `pbmc_like` into `tests/fixtures/stage_inputs/`, a
two-sample input in two formats with populations planted to stress the QC rules. It is seeded
and deterministic (about 0.5 MB).

| File | What it is |
|---|---|
| `donor_A/` | 10x `matrix.mtx.gz`, `features.tsv.gz`, `barcodes.tsv.gz` (format left blank in `samples.csv`, so S0 must auto-detect it) |
| `donor_B.h5ad` | Plain AnnData, not an exchange file: raw counts in `layers["counts"]`, log-normalised floats in `X`, genes in a different order and 4 genes missing |
| `samples.csv` | Sample sheet with paths relative to this folder, plus `donor`, `condition` and `batch` metadata columns. CellWarden resolves sample paths against the folder holding `config.yaml`, so this sheet suits a config placed here; the end-to-end test uses `tests/integration/stage_configs/samples.csv`, which lists the same files |
| `truth.tsv` | One row per cell: `cell_id` as the pipeline names it (`<sample_id>_<barcode>`), `sample`, planted `cell_type`, `injected` |
| `manifest.json` | Cells, genes and UMIs per sample, injected counts, and SHA-256 of each file |

Injected per sample (the `injected` column):

| `injected` | Per sample | How it is made | Expected QC outcome |
|---|---|---|---|
| `none` | 244 / 256 | The original `pbmc_like` cells (T, B, Mono, NK) | `keep` |
| `lowq` | 12 / 13 (5 %) | A real cell thinned to 18–30 % of its UMIs, then mitochondrial reads added up to 35–60 % | `exclude` (`HIGH_MT`) |
| `doublet` | 10 (4 %) | The sum of two real cells of different types from the same sample | `exclude` (`DOUBLET`, or an adaptive upper bound) |
| `platelet` | 12 | Small libraries (about 60 genes, 140 UMIs) with 17–50 % of UMIs in PPBP, PF4 and GNG11 | `review` with `PROTECTED_POP:Platelet`: they pass the hard limits but fall under the adaptive lower bounds |

Genes added to both samples: PPBP, PF4, GNG11, HBB, HBA1, HBA2 (ambient level only), and HBP1
and HBEGF, which are not haemoglobin genes and must not count towards `pct_hb`.

```bash
python tests/fixtures/make_stage_inputs.py           # regenerate (only if you change the generator)
python tests/fixtures/make_stage_inputs.py --check   # regenerate in a temp dir and compare
```

`--check` compares every file byte for byte, except `donor_B.h5ad`, whose content is compared
instead if its HDF5 bytes differ (different h5py versions can lay out the same data differently).

### End-to-end run

```bash
tests/integration/run_stages.sh                  # temp workdir, removed if everything passes
tests/integration/run_stages.sh /tmp/stages      # keep run directories and logs in /tmp/stages
KEEP_WORKDIR=1 tests/integration/run_stages.sh
```

It uses `tests/integration/stage_configs/cpu.yaml` and `gpu.yaml`, which differ only in `device`,
plus `stage_configs/samples.csv` and `stage_configs/qc_policy.yaml`. The script copies them into
`<workdir>/stage_configs/` (with a `samples.csv` whose relative paths point back at the fixture), so
the config with a changed resolution can sit next to them and share the same inputs. The same
configs also work by hand: `cellwarden run --config tests/integration/stage_configs/cpu.yaml --run-dir /tmp/x`. That policy matches `config/examples/qc_policy.yaml` except
for lower hard limits (40 genes, 80 UMIs), because the synthetic cells are shallow. The `gpu` run
sets the test-only override `CELLWARDEN_ALLOW_CPU_AS_GPU=1`, so on a machine without a GPU it still
takes the GPU path (slim exchange file, `python -m cellwarden.gpu_core`, merge back) but `gpu_core`
computes on the CPU. The checks are in `tests/integration/check_stages.py`; each step's log goes to
`<workdir>/logs/`, and the two run directories are `<workdir>/run_cpu` and `<workdir>/run_gpu`.

| Step | What it checks |
|---|---|
| T0 | `make_stage_inputs.py --check`: the inputs are reproducible; the three sample sheets resolve to the same files |
| T1 | `cellwarden doctor` runs |
| C1, G1 | `cellwarden run --until S2` succeeds with `device: cpu` and with `device: gpu` |
| C2, G2 | `state.json`: every stage `done`, with fingerprint and start/end times; every `logs/<STAGE>.log`; the config copies and `hashes.json` |
| C3 | `S0/ingest_report.json`: formats (donor_A auto-detected), cells, genes and UMIs per sample match `manifest.json`; the merged object has the union of genes |
| C4 | `S1/qc_metrics.tsv.gz`: every column from the spec; `n_genes`, `n_umis`, `pct_mito`, `pct_ribo`, `pct_hb`, `log10_genes_per_umi` and `protected_*` match a computation straight from the input files; `metrics_counts.h5ad` deleted after S1c |
| C5, G5 | `S1/qc_decisions.tsv.gz` against `truth.tsv`: at least 90 % of `lowq` excluded, at least 50 % of `doublet` excluded, at least 80 % of `platelet` in `review` with `PROTECTED_POP:Platelet`, at least 85 % of `none` kept. Prints the full decision × population table and the reason-code counts |
| X1 | QC decisions, reasons and policy hash are identical between the two runs |
| C6, G6 | `S2/summary.json` names the device, plan and reason; the S2 object (exported with a short Rscript) has counts/data layers, PCA, UMAP and a graph, holds exactly the keep + review cells, and its clusters recover the planted types with purity at least 0.9 on non-injected cells. The GPU run must record an exchange merge |
| X2 | ARI between the CPU-path and GPU-path clusters is at least 0.7 |
| R0 | `cellwarden status` runs and lists every stage |
| R1 | Running again with the same config skips every stage (start and end times and fingerprints unchanged) |
| R2 | `cellwarden resume` on a finished run changes nothing |
| R3 | With `clustering.resolution` changed, only S2 reruns |

The thresholds can be changed with environment variables: `MIN_LOWQ_EXCLUDED`,
`MIN_DOUBLET_EXCLUDED`, `MIN_PLATELET_REVIEW`, `MIN_NONE_KEPT`, `MIN_PURITY` and `MIN_ARI`.
The doublet threshold is deliberately low: scrublet has little to work with on about 280 cells
and 800 genes, so the step prints the rate it measured rather than pretending it is high.

A full run takes a few minutes on a laptop. Like the round-trip, it prints a PASS/FAIL table
with timings and exits non-zero if any step fails or is skipped.

## Full pipeline (S0–S5)

```bash
tests/integration/run_full.sh                    # temp workdir, removed if everything passes
tests/integration/run_full.sh /tmp/full          # keep the run directory and logs
```

Same fixture and `cpu.yaml` as the stage test, plus `llm.default.provider: mock`,
`opentargets.offline: true`, every gate `auto`, and no DGE contrasts (both fixture samples are
controls). It runs S0–S2 and then `cellwarden resume --until <stage>` for every later stage in the
runner's registry: G1, S3, G2, G3, S4, G4, S5. A gate that stops for a human (exit 3) is
approved with `cellwarden approve --by run_full` and the run resumed. A stage that is not registered
yet is reported as SKIP and does not fail the script.

| Step | What it checks |
|---|---|
| T0, T1 | inputs reproducible; `cellwarden doctor` with the mock LLM and offline Open Targets |
| F1–F4 | S0–S2 as in the stage test (state, QC decisions vs truth, S2 object and purity) |
| G1–G4 | each gate's `bundle.json`, `proposal.json`, `review.json`, `decision.json` (shapes of agents-spec section 2) and `report.html` |
| S3 | `markers.tsv.gz` columns, `cluster_summary.json`, `signatures.tsv.gz`, `evidence_bundle.json` without per-cell data, `annotation_consensus.tsv` |
| S4 | `dge_summary.json` (every listed DGE table exists) and `opentargets_summary.json` (offline) |
| S5 | `S5/umap.tsv.gz` has one row per S2 cell and a `cell_type_broad` column; `report.html` parses (balanced tags), has no external resources and every section; `explorer.html` embeds the JSON for the same cells |
| R0, R1 | `cellwarden status` lists S5; rerunning the same config skips every stage |

Python unit tests for the report (`tests/python/test_report.py`) and portability
(`tests/python/test_site.py`) run on synthetic run directories and a fake `sbatch`/`bsub` on
`PATH`; no R, scheduler or network is needed.

## Where the tests run

There is no hosted CI: we decided not to use GitHub Actions. Run the tests locally before
pushing, or on the HPC (on Iridis, inside the project's environment on the login node, keeping
to the login-node etiquette in [`docs/DESIGN.md`](../docs/DESIGN.md) section 2.1; the stage test
already caps itself at 2 threads and runs under `nice`). A full check is:

```bash
Rscript -e 'testthat::test_dir("tests/R")'
PYTHONPATH=python python -m pytest -q tests/python
tests/integration/run_roundtrip.sh
tests/integration/run_stages.sh
tests/integration/run_full.sh
```

## Benchmarks (development)

Only 2k and 5k cells during development; larger sizes are benchmarked on the HPC.

```bash
tests/benchmark/run_bench.sh          # S0-S2, CPU and GPU path, 2k and 5k cells
SIZES="2000" tests/benchmark/run_bench.sh
```

Results are written to `tests/benchmark/results/` (not committed). `make_bench_inputs.py` refuses sizes above 5,000.
