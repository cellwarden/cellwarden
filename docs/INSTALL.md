# Installing and running CellWarden

Three targets: the **Iridis X L4 login node**, any **Linux** machine, and **Windows**.
All use the same `config.yaml`; only the install and the `site` differ.

Which engine runs the CPU steps is set by `engine:` in `config.yaml`
([scfast-integration.md](scfast-integration.md)):

| `engine` | S1 doublets/pre-clusters, S2 PCA/neighbours/Leiden | exchange `.h5ad` files |
|---|---|---|
| `auto` (default) | Rust `scfast` if `scfastR` is installed and the plan is CPU, else Seurat | none on the scfast path |
| `scfast` | Rust `scfast`, error if not installed or if `device: gpu` | none |
| `seurat` | Seurat (CPU) or rapids-singlecell via exchange (GPU) | only on the GPU path |

On a GPU plan `engine_gpu: auto | scfast | rapids` picks between the Rust CUDA backend
(`gpu_scfast`, no exchange) and rapids-singlecell via the exchange; `auto` uses the Rust backend
only when `scfastR` was built with `--features cuda` and a device is visible. Rust is needed only
to build `scfast`; without it `engine: auto` falls back to Seurat.

The CUDA and cuGraph backend has run on an H100 (bench/results/FINAL.md) and on the Iridis X L4
(PBMC 3k through bench/run_py.py: PC similarity 1.0, kNN overlap 0.998 and ARI 0.91 against
Seurat). The R side (`scfastR` with `--features cuda`) has not been run through the pipeline on a
GPU yet, so `engine_gpu: auto` stays on rapids until `scfastR` reports a CUDA build.

## 0. Quick install (recommended)

**Everything, in one image.** R with Seurat 5 and the Bioconductor packages, CellWarden, the Rust
engine built with CUDA and cuGraph (`scfast`, `scfastR`), RAPIDS (cuGraph, cuVS, CuPy,
rapids-singlecell) and CellTypist's default model, on NVIDIA's RAPIDS base image. Nothing to
compile; it runs on a GPU or, slower, without one.

```bash
# HPC (Apptainer / Singularity): pull once, then point the site file at the .sif
apptainer pull cellwarden.sif docker://ghcr.io/cellwarden/cellwarden:latest      # Docker Hub copy: docker://ad2n15/cellwarden:latest
apptainer exec --nv cellwarden.sif cellwarden doctor
apptainer exec --nv cellwarden.sif cellwarden run --config config.yaml --detach

# Workstation or cloud VM (Docker with the NVIDIA container toolkit)
docker run --rm --gpus all -v "$PWD:$PWD" -w "$PWD" ghcr.io/cellwarden/cellwarden:latest run --config config.yaml
```

To keep `cellwarden` on the host and run only the stages in the image, set in the site file
(`cellwarden init-site` prints these lines when it finds Apptainer or Docker):

```yaml
container:
  runtime: apptainer
  image: /path/to/cellwarden.sif          # or docker://ghcr.io/cellwarden/cellwarden:<version>
  options: ["--nv"]
```

Each stage then runs in the image with the run directory, the config directory and the sample
folders bound; `cellwarden doctor` stops asking for R packages on the host and points at the image.
Check an image after pulling it with
`apptainer exec --nv cellwarden.sif bash /opt/cellwarden/containers/smoke_test.sh` (the whole pipeline on
the test fixture, about 10 minutes on a CPU).

**Python side only (no R, no compiler).** Prebuilt `scfast` wheels (Linux x86_64 with CUDA and
cuGraph loaded at run time; Linux and macOS ARM for the CPU) and `cellwarden`:

```bash
# the wheels are attached to each release: https://github.com/cellwarden/cellwarden/releases
pip install scfast-*.whl "$(ls cellwarden-*.whl)[all]" --extra-index-url https://pypi.nvidia.com
```

`[all]` adds Harmony, the report, the LLM clients and the CUDA 12 GPU stack; use plain `cellwarden` on
a machine without an NVIDIA GPU. The R stages (S0, S1a, S2, S4) still need R: use the image
through `container:` as above, or install R by hand (sections 1 and 2;
`Rscript containers/install_r_packages.R` installs every package the stages use).

The wheels and the image are built and published by `.github/workflows/release.yml` on every
`v*` tag (GHCR `ghcr.io/cellwarden/cellwarden`, Docker Hub `ad2n15/cellwarden`, wheels on the GitHub Release). The sections below build everything from source.

## 1. Iridis X, L4 login node

Not yet run on the real node; the steps below are what the code expects. Check the points marked
**confirm** with the Iridis team (hpc@soton.ac.uk) first, especially whether multi-hour GPU work on
the shared login node is allowed. If it is not, set `execution: split` (see section 5 of the user guide).

```bash
ssh <user>@iridisx.soton.ac.uk            # confirm the login host name
nvidia-smi                                # the L4 should be listed; note free memory
module avail 2>&1 | grep -i -E "^R/|python|apptainer|rust|cuda"   # confirm module names
```

**One-time setup** (put everything on scratch, not home):

```bash
cd /scratch/$USER && git clone https://github.com/cellwarden/cellwarden.git singlecell && cd singlecell
module load R python                      # confirm names/versions: R >= 4.3, Python >= 3.12
python3 -m venv /scratch/$USER/venv-cellwarden && source /scratch/$USER/venv-cellwarden/bin/activate

# Rust engine first (cellwarden depends on scfast): needs cargo >= 1.80. On Iridis keep the
# toolchain on scratch (export RUSTUP_HOME=/scratch/$USER/rustup CARGO_HOME=/scratch/$USER/cargo).
curl --proto '=https' -sSf https://sh.rustup.rs | sh -s -- -y && source "${CARGO_HOME:-$HOME/.cargo}/env"
pip install "maturin>=1.9.4,<2"
maturin develop --release --features cuda,cugraph -m rust/scfast-py/Cargo.toml   # scfast for Python
pip install -e "python[harmony,llm,report,dev]"
pip install -e "python[gpu]" --extra-index-url https://pypi.nvidia.com            # CUDA 12; large download

# R packages (user library on scratch), then scfastR
mkdir -p /scratch/$USER/Rlib && export R_LIBS_USER=/scratch/$USER/Rlib
Rscript containers/install_r_packages.R
R CMD INSTALL rust/scfast-r
```

On Iridis X, without a published image, build it on the login node (compute nodes have no
internet), keeping Apptainer's temporary directory on the local disk (GPFS breaks fakeroot builds):

```bash
APPTAINER_TMPDIR=/tmp/$USER-apptainer apptainer build --fakeroot cellwarden.sif containers/cellwarden-build.def
```

Iridis has no `fusermount`, so run the image with `--unsquash` (it unpacks to `APPTAINER_TMPDIR`
first, ~30 s, in RAM on compute nodes, so jobs need ~96 GB): `apptainer exec --unsquash --nv
cellwarden.sif cellwarden doctor`, and put `--unsquash` in `container.options` of the site file.

Better on Iridis: unpack the image once into a sandbox directory (~130k files, 12 GB) and run that,
with no `--unsquash` and no extra memory. Unpack on the local disk and copy with `rsync -a`
(extracting straight onto GPFS drops the execute bits):

```bash
apptainer build --sandbox /tmp/$USER-sbx cellwarden.sif
rsync -a /tmp/$USER-sbx/ /path/to/shared/cellwarden_sandbox/ && rm -rf /tmp/$USER-sbx
apptainer exec --nv /path/to/shared/cellwarden_sandbox cellwarden doctor
```

After a code change, update the sandbox in place instead of rebuilding the image (login node,
~5 min with the Rust rebuild, ~20 s with `--code-only`); rebuild only when the base image or the
dependency lists change:

```bash
containers/update_sandbox.sh /path/to/shared/cellwarden_sandbox
```

On Iridis the image (section 0) replaces all of the above: `apptainer pull` it once into a
shared folder and set `container.image` in the site file (`conf/sites/iridisx.yaml`
already passes `--nv`).

**Configure and check:**

```bash
cp -r config/examples /scratch/$USER/my_study && cd /scratch/$USER/my_study
# edit config.yaml: site: iridisx, run_dir: /scratch/$USER/cellwarden/my_study, engine: auto,
# device: auto, plus samples.csv and qc_policy.yaml
export ANTHROPIC_API_KEY=...              # optional; without it the gates use rules only
cellwarden doctor --config config.yaml        # packages, scfastR/scfast, GPU, LLM key, budget, Open Targets
```

`doctor` prints the engine `auto` would choose. For datasets under about 100k cells it will pick
the Rust CPU engine; above that, with the L4 free, S2 uses rapids-singlecell through the exchange.
(The exact cut-over is set by the device plan; `S2/summary.json` records which path ran.)

**Run, detached so a dropped SSH session does not kill it:**

```bash
cellwarden run --config config.yaml --detach
cellwarden status --run-dir /scratch/$USER/cellwarden/my_study
cellwarden approve --run-dir /scratch/$USER/cellwarden/my_study --stage G1 --by $USER   # when a gate waits (exit code 3)
cellwarden resume  --run-dir /scratch/$USER/cellwarden/my_study
```

Results: `S5/report.html` and `S5/explorer.html` in the run directory. Copy them to your laptop with
`scp`. Be considerate on the shared node: keep `resources.threads` at 8 or fewer, `nice: 10`, and
`gpu_mem_cap_gb: 20` as in the site file.

### 1b. Enabling the Rust GPU backend on the L4 (acceptance tests first)

```bash
# needs the CUDA 12 toolkit (nvrtc, cublas) and, for GPU Leiden, a RAPIDS env with libcugraph_c
module load cuda/12.4                                   # confirm the name; or activate the RAPIDS conda env
cd /scratch/$USER/singlecell/rust

# 1. Rust parity tests on the device (PCA subspace > 0.99, kNN overlap >= 90%, Leiden ARI >= 0.9)
cargo test -p scfast-core --release --features cuda --test cuda -- --ignored --nocapture
export SCFAST_CUGRAPH_LIB=$CONDA_PREFIX/lib/libcugraph_c.so   # if not on the loader path
export SCFAST_CUGRAPH_API=legacy                              # or 24.08 / 24.10 to match `conda list cugraph`
SCFAST_CUGRAPH_TEST=1 cargo test --features cugraph -- --ignored cugraph --nocapture

# 2. Rebuild the R package and the wheel with the features on
SCFASTR_CARGO_FEATURES=cuda,cugraph R CMD INSTALL rust/scfast-r
Rscript -e 'str(scfastR::scfast_info())'               # cuda_built TRUE, cuda_available TRUE, device_name "NVIDIA L4"
cd rust/scfast-py && maturin build --release --features cuda,cugraph && pip install target/wheels/scfast-*.whl

# 3. Pipeline-level check, then the scale runs (bench/hpc/README.md)
cellwarden doctor --config config.yaml                       # "scfastR cuda built: ok", "CUDA device visible: ok"
sbatch --export=ALL,SIZES=50000,ENGINES=seurat-cpu,scfast-cpu,scfast-cuda,rapids bench/hpc/slurm_scale.sh
```

If a parity test fails, report the numbers; `engine_gpu: rapids` keeps the pipeline on the proven
path meanwhile. The most likely failure is the cuGraph graph-constructor layout: try the other
`SCFAST_CUGRAPH_API` values before anything else.

## 2. Linux (workstation, server, cloud VM)

Same as above without the `module` lines. On Ubuntu 22.04/24.04:

```bash
sudo apt-get install -y build-essential gfortran git tmux cmake pkg-config \
  libhdf5-dev libcurl4-openssl-dev libssl-dev libxml2-dev libglpk-dev libgsl-dev \
  python3-venv python3-dev r-base r-base-dev
git clone https://github.com/cellwarden/cellwarden.git singlecell && cd singlecell
python3 -m venv .venv && source .venv/bin/activate
# Rust, scfast, the Python package, R packages and scfastR: as in section 1
# (or skip all of it and use the image, section 0)
cellwarden init-site --out conf/sites/mybox.yaml     # detects scheduler, GPU, container runtime
```

Set `site: workstation` (or your generated file) and run exactly as in section 1.
With Docker and an NVIDIA GPU: `docker build -t cellwarden -f containers/Dockerfile .` then
`docker run --rm --gpus all -v $PWD:/work cellwarden cellwarden doctor`. Without a GPU, drop `--gpus all`
and use `device: cpu`.

## 3. Windows

Run it **inside WSL2**, which gives you a real Ubuntu, so the Linux steps apply unchanged.
Native Windows is not supported: the runner relies on `nice`, `tmux`/`nohup` and POSIX process
handling, and rapids-singlecell is Linux-only. Native installs have not been tested.

```powershell
# PowerShell as Administrator, once; then reboot
wsl --install -d Ubuntu-24.04
```

Open the Ubuntu terminal and follow **section 2**. Notes:

- Keep the project and run directory **inside the Linux filesystem** (`~/singlecell`), not under
  `/mnt/c/...`; I/O there is much slower.
- GPU: install the current NVIDIA Windows driver (it exposes the GPU to WSL2); do not install a Linux
  driver in WSL. Check with `nvidia-smi`. Without a GPU use `device: cpu`.
- Memory: WSL2 takes half the RAM by default. Raise it in `%UserProfile%\.wslconfig`
  (`[wsl2]` / `memory=32GB`) and run `wsl --shutdown`.
- Alternative: Docker Desktop with the WSL2 backend, then the `docker run` line in section 2.
- View results by opening `S5\report.html` from Explorer at `\\wsl$\Ubuntu\home\<you>\...`.

## 4. Check that it worked

```bash
PYTHONPATH=python python3 -m pytest -q tests/python     # unit tests, offline
bash tests/integration/run_full.sh                      # whole pipeline S0-S5 on the synthetic fixture
```

On the scfast path `S2/summary.json` shows `"path": "cpu_scfast"` and no `.h5ad` files appear under
the run directory.
