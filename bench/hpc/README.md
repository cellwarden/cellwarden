# Scale runs on the HPC (Phase 8)

Runs the pipeline's S0–S2 (`cellwarden run --until S2`) at 50k, 100k, 200k and 500k cells on
four engine variants and judges each against the Seurat baseline with the adoption rule in
[`docs/RUST_ENGINE.md`](../../docs/RUST_ENGINE.md) section 8.2. It is Phase 8 of
[`docs/scfast-integration.md`](../../docs/scfast-integration.md): the first place the
scfast CUDA backend meets a real GPU, since the development workspace has none. Nothing here
runs in CI; the unit tests only cover the input guard and the summariser.

| Variant | Config | What runs S2 |
|---|---|---|
| `seurat-cpu` | `engine: seurat`, `device: cpu` | Seurat on the CPU (baseline) |
| `scfast-cpu` | `engine: scfast`, `device: cpu` | scfastR in process (`cpu_scfast`) |
| `scfast-cuda` | `engine: scfast`, `device: gpu`, `engine_gpu: scfast` | scfastR in process on CUDA (`gpu_scfast`), UMAP on the CPU |
| `rapids` | `engine: seurat`, `device: gpu`, `engine_gpu: rapids` | rapids-singlecell through the exchange (`gpu_exchange`) |

All variants share one config apart from those keys: seed 42, 2,000 HVGs, 30 PCs, 20
neighbours, Leiden resolution 0.5, scrublet doublets, no integration, no LLM.

## Steps

1. **Inputs** (once, on the HPC; refused elsewhere without `CELLWARDEN_ALLOW_LARGE=1`):

   ```bash
   CELLWARDEN_ALLOW_LARGE=1 python bench/hpc/make_scale_inputs.py --out bench/hpc/data \
       --sizes 50000,100000,200000,500000 --genes 5000 --types 12
   ```

   Each size gets `bench/hpc/data/n<N>/` with `sample.h5` (10x `.h5`, from `bench/synth.py`),
   `samples.csv` and `truth.tsv` (planted type per cell, named `syn_<barcode>` as S0 names
   cells). Sizes up to 5,000 cells never need the variable, so the scripts can be tried
   locally: `--sizes 2000`.

2. **Runs**, interactively or as a job:

   ```bash
   bash bench/hpc/run_scale.sh --data bench/hpc/data --threads 8          # everything
   sbatch bench/hpc/slurm_scale.sh                                         # same, as a job
   sbatch --export=ALL,SIZES=50000,ENGINES=scfast-cpu,scfast-cuda bench/hpc/slurm_scale.sh
   ```

   `run_scale.sh` options: `--sizes`, `--engines`, `--threads` (also pins BLAS/OpenMP/numba/
   rayon), `--mem-gb` / `--gpu-cap-gb` (`resources.max_memory_gb` / `gpu_mem_cap_gb`),
   `--timeout` (seconds per run, default 4 h), `--out` (default `bench/results/scale`),
   `--seed`. `slurm_scale.sh` takes the same settings as environment variables
   (`DATA SIZES ENGINES THREADS OUT TIMEOUT MEM_GB GPU_CAP_GB`), plus `MODULES`,
   `CONDA_ENV` or `VENV` for the software stack; edit its `#SBATCH` lines for the site
   (partition `l4`, one GPU, 8 cores, 64 GB, 24 h).

   Each run lives in `<out>/n<N>/<variant>/`: `config.yaml`, `run/` (the CellWarden run
   directory with `S2/summary.json` and the stage logs), `time.log` (`/usr/bin/time -v`),
   `gpu_mem.log` (`nvidia-smi` samples once a second, GPU variants), `clusters.tsv.gz`
   (S2 clusters). One JSON line per run goes to `<out>/runs.jsonl`. A variant that fails or
   times out at one size is skipped at larger sizes.

3. **Summary**, run automatically at the end of `run_scale.sh` and repeatable on its own:

   ```bash
   python bench/hpc/summarise.py --runs bench/results/scale/runs.jsonl --out bench/results/scale
   ```

   Writes `summary.md`, `summary.csv` and `summary.json`: per size and variant, the S2 time
   (`S2/summary.json` `total_s`), the whole-run time, peak host RSS (largest process in the
   run, from `time.log`; the R stage's own peak as a fallback), peak GPU memory (largest
   `nvidia-smi` sample), ARI against the Seurat run at the same size and against the planted
   types, and the verdict.

## Verdict

Per non-baseline variant and size, following RUST_ENGINE.md 8.2:

- `fails parity` when ARI against the Seurat clusters is below 0.90;
- `parity ok; too small to judge` below 100k cells;
- `adopt` at 100k cells or more when S2 is at least 1.5× faster than Seurat **or** peak host
  RSS is at least 2× lower; otherwise `keep optional`;
- `failed` / `timeout` / `no Seurat baseline` / `no parity data` when a number is missing.

Read the table alongside the caveats in 8.2: the L4 node is shared (the job header prints
`uptime` and `nvidia-smi`; repeat a run that overlapped with another user), thread counts are
recorded with every run, scfast works in f32 and Seurat in f64, and the data is synthetic, so
use it for timing and the scaling curve and check parity on real data before adopting.

## What to expect to go wrong

- `engine_gpu: scfast needs scfastR built with CUDA`: scfastR was installed without the
  `cuda` feature or no device is visible; `python -m cellwarden doctor` shows both lines.
- `rapids` needs `rapids_singlecell` + `cupy` importable from the same Python; without them
  the variant fails at the first size and is skipped afterwards.
- Seurat at 500k cells may run out of memory in `ScaleData`/`RunPCA`; that is a result
  (recorded as `failed`), not a harness problem. Raise `--mem-gb` and the job's `--mem`
  together if the node has more.
