# Getting started with CellWarden

CellWarden takes single-cell RNA-seq count files and runs the whole analysis:

- QC, doublet removal, normalisation, batch correction and clustering;
- cell-type annotation;
- differential expression, with drug and disease evidence from Open Targets;
- a self-contained HTML report.

At four points ("review gates") LLM agents check the decisions; you approve them, or let the
agents decide.

This page covers the everyday path: install, check, run, find the results. The details are in the
[User guide](USER_GUIDE.md) and the [Pipeline reference](PIPELINE.md).

**Contents**
1. [What you need](#1-what-you-need)
2. [Install](#2-install)
3. [Connect an AI backend (optional)](#3-connect-an-ai-backend-optional)
4. [Quick test (5 minutes)](#4-quick-test-5-minutes)
5. [Run your own data](#5-run-your-own-data)
6. [Guided setup (`-i`)](#6-guided-setup--i)
7. [Review gates and LLM agents](#7-review-gates-and-llm-agents)
8. [Differential expression](#8-differential-expression)
9. [Big datasets: compute nodes and GPUs](#9-big-datasets-compute-nodes-and-gpus)
10. [Results](#10-results)
11. [Everyday commands](#11-everyday-commands)
12. [Troubleshooting](#12-troubleshooting)
13. [FAQ](#13-faq)

---

## 1. What you need

| | Requirement |
|---|---|
| **Computer** | Linux x86_64: an HPC login or compute node, a workstation, or a cloud VM. macOS / Windows through Docker Desktop (section 2A) or WSL2 ([INSTALL.md](INSTALL.md)). |
| **Container runtime** | Apptainer ≥ 1.3 (HPC) or Docker (workstation, cloud, macOS / Windows via Docker Desktop). Everything else (R 4.6, Seurat 5, Bioconductor, Python 3.12, the Rust engine, CellTypist, RAPIDS) is inside the CellWarden image. |
| **Disk** | The image is 6.6 GB as a `.sif` file, or 12 GB / ~133k files unpacked as a sandbox. Each run needs a few hundred MB to a few GB, depending on the number of cells. |
| **Memory** | PBMC 3k (2,700 cells) peaks at about 1.4 GB. Memory grows with the number of cells; for 100k+ cells start with 64 GB, and check the peak memory shown per stage. |
| **CPU / GPU** | 8 cores is a good default. A GPU (e.g. NVIDIA L4) is optional; the CPU path is complete. |
| **Internet** | Needed for the LLM review agents and the Open Targets evidence. Not needed for the stages in between (most of the computing). On many clusters, compute nodes have **no** internet: see [section 9](#9-big-datasets-compute-nodes-and-gpus). |
| **LLM key (optional)** | One of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `NVIDIA_API_KEY`. Without a key the gates use rules only and stop for you each time. |
| **Data** | 10x `.h5` or `matrix.mtx` folders, `.h5ad`, Seurat `.rds` / `.qs2`, `.loom`, or csv/tsv count tables. **Human** data. Raw 10x matrices are reduced to cells automatically. |

---

## 2. Install

The image is published on GitHub's registry as **`ghcr.io/cellwarden/cellwarden`**, with a copy on
Docker Hub as `ad2n15/cellwarden`. Use a version tag (`0.3.0`) for reproducible work, or `latest`.
[Releases](https://github.com/cellwarden/cellwarden/releases) lists the versions.

### Option A: Docker (workstation, cloud VM, macOS / Windows)

You need Docker. For a GPU you also need the NVIDIA Container Toolkit (Linux).

```bash
docker pull ghcr.io/cellwarden/cellwarden:0.3.0
```

**Make a short command.** The image starts `cellwarden` in `/work`, so mount your project folder
there:

```bash
alias cellwarden-docker='docker run --rm -it --user "$(id -u):$(id -g)" -e HOME=/home/cellwarden -v "$HOME/.cellwarden:/home/cellwarden/.cellwarden" -v "$PWD:/work" ghcr.io/cellwarden/cellwarden:0.3.0'
mkdir -p ~/.cellwarden
cellwarden-docker doctor
```

| Part | Why |
|---|---|
| `-it` | Keeps the terminal, for the progress display, `-i` and the DE questions. |
| `--user` | Results belong to you, not the image's user. |
| `~/.cellwarden` mount | The LLM budget ledger and the time-left history persist between runs. |
| `-v "$PWD:/work"` | Data paths must be inside the current folder, given as relative paths. For data elsewhere, add `-v /data:/data:ro` and use `/data/...` paths. |

**Run.** From the folder that holds the data:

```bash
cd ~/cellwarden-test
cellwarden-docker run filtered_gene_bc_matrices/hg19 --name pbmc3k --out pbmc3k_full
```

Add these options to the alias, before the image name:

| Option | Add |
|---|---|
| API keys | `--env-file .env` |
| GPU | `--gpus all` |
| More cores / memory | `--cpus 16 --memory 64g` |
| Keep running after you close the terminal | Instead of `--detach`, use `-d` in place of `-it`, then `docker logs -f <id>` |

**macOS / Windows:** use Docker Desktop. The image is linux/amd64: on Apple silicon add
`--platform linux/amd64` (emulated, so slower, and no GPU). On Windows, run the commands inside
WSL2 ([INSTALL.md](INSTALL.md)).

### Option B: Apptainer / Singularity (HPC)

```bash
apptainer pull cellwarden.sif docker://ghcr.io/cellwarden/cellwarden:0.3.0      # Docker Hub copy: docker://ad2n15/cellwarden:0.3.0
apptainer exec --cleanenv cellwarden.sif cellwarden doctor
```

- **Shared clusters:** use the `ghcr.io` image. Docker Hub limits anonymous pulls per IP address,
  and a cluster's users share one (the error is `toomanyrequests`).
- **Building the image:** this takes a few minutes and lots of small files. Set
  `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR` to a disk with space, not home.
- **Sites without `fusermount` (Iridis):** a `.sif` only runs with `--unsquash`, which unpacks it
  into RAM on every start. Unpack it once into a sandbox folder, then use that folder in place of
  `cellwarden.sif`:
  ```bash
  apptainer build --sandbox /tmp/$USER-sbx docker://ghcr.io/cellwarden/cellwarden:0.3.0
  rsync -a /tmp/$USER-sbx/ /path/to/shared/cellwarden_sandbox/ && rm -rf /tmp/$USER-sbx
  ```
  The `rsync -a` step matters on GPFS: unpacking straight onto it loses the execute permissions.

To save typing, define a shell function, for example in `~/.bashrc` (use your `.sif` or sandbox
path):

```bash
cellwarden() { apptainer exec --cleanenv --env-file "${CELLWARDEN_ENV:-/dev/null}" -B /path/to/data,/scratch /path/to/cellwarden.sif cellwarden "$@"; }
```

`cellwarden run ...` then works like a local command. `CELLWARDEN_ENV=.env` passes your API keys
(section 3). `-B` binds the filesystems your data lives on.

### Option C: a shared copy on your cluster

If your group has unpacked the image into a shared folder, there is nothing to install: use
that folder in place of `cellwarden.sif`. For example, the Iridis X setup is described in
[docs/sites/iridis.md](sites/iridis.md).

### Option D: build from source

This is for development, or for sites that build every image themselves. It takes about 40–60
minutes, needs internet, and about 15 GB of disk:

```bash
git clone https://github.com/cellwarden/cellwarden.git && cd singlecell-container
docker build -t cellwarden -f containers/Dockerfile .                                            # Docker
APPTAINER_TMPDIR=/tmp/$USER-apptainer apptainer build --fakeroot cellwarden.sif containers/cellwarden-build.def   # Apptainer
```

After a code change, update a sandbox in place in about 20 s, instead of rebuilding. Without
`--code-only` it also rebuilds the Rust engine, in about 5 minutes:

```bash
containers/update_sandbox.sh /path/to/cellwarden_sandbox --code-only
```

### Check the install

```bash
cellwarden doctor
```

With Docker this is `cellwarden-docker doctor`. It lists the Python and R packages, the GPU, the LLM
provider (with a test call), the budget and Open Targets. Every **required** line must say `ok`.

---

## 3. Connect an AI backend (optional)

The review agents at the four gates (section 7) call a large language model. Connect one by
setting **one** API key, or by pointing CellWarden at a model server you run yourself. Without a
backend everything still runs: the gates decide by rules and stop for your approval.

| Provider | Key variable | Get a key | Default model |
|---|---|---|---|
| Claude (Anthropic) | `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) → API keys | `claude-haiku-4-5` |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | `gpt-4.1-mini` |
| NVIDIA NIM | `NVIDIA_API_KEY` | [build.nvidia.com](https://build.nvidia.com) → *Get API Key* (free development credits) | `nvidia/nemotron-3-ultra-550b-a55b` |

**Step 1: put the key in a `.env` file**, in your project folder:

```bash
cat > .env <<'EOF'
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxx
EOF
chmod 600 .env
```

- **Format:** one `NAME=value` per line, with no quotes, no spaces around `=` and no `export`.
  Lines starting with `#` are comments.
- **Keep it private:** `chmod 600` makes it readable only by you. Never commit it (add `.env` to
  `.gitignore`).
- **Where the key goes:** CellWarden never writes the key anywhere. The run records only which
  variable it used.

**Step 2: pass the file to cellwarden.** Containers don't see your shell's variables unless told:

| How you run CellWarden | Add |
|---|---|
| Docker (section 2A) | `--env-file .env` in the `docker run` command or the `cellwarden-docker` alias |
| Apptainer `.sif` or sandbox (2B, 2C) | `--env-file .env` after `apptainer exec --cleanenv` |
| The `cellwarden` shell function (2B) | `CELLWARDEN_ENV=.env cellwarden run ...` |
| A native install (no container) | `set -a; source .env; set +a` before `cellwarden run` |

**Step 3: check it.** `cellwarden doctor`, run the same way, makes a small test call:

```
llm:
  [  ok] provider               nvidia, model nvidia/nemotron-3-ultra-550b-a55b (nvidia (auto: $NVIDIA_API_KEY is set))
  [  ok] api key                $NVIDIA_API_KEY is set (provider nvidia)
  [  ok] endpoint               https://integrate.api.nvidia.com/v1: HTTP 404 (host reachable)
  [  ok] tool-use test          ok: stop_reason=tool_use, tool_calls=1
  [  ok] budget                 GBP 3.57 of 30.00 spent across runs (/home/you/.cellwarden/ledger.json)
```

`provider ... none: no API key set` means the key didn't reach CellWarden; see step 2.

**Choosing a provider and model:**
- **Several keys:** CellWarden uses Claude first, then OpenAI, then NVIDIA.
- **Pick one per run:** `--llm nvidia` (or `anthropic`, `openai`, `none`).
- **Another model:** `--llm-model <name>`. It must support tool (function) calling.
- **In a config file:** `llm: {default: {provider: nvidia, model: ...}}`.
- **Per gate:** for example, use a different model for the reviewer:
  `llm: {roles: {reviewer: {model: ...}}}`.

**A model server you run yourself** (vLLM, Ollama, a local NIM). No data leaves your site, and
no key is needed unless the server asks for one:

```yaml
llm:
  default:
    provider: openai_compatible
    base_url: http://localhost:8000/v1     # Ollama: http://localhost:11434/v1
    model: <the model name the server serves>
    # api_key_env: MY_SERVER_KEY           # only if the server needs a key (put it in .env)
```

**Cost and network:**
- **Budget:** every call is priced and logged, with a budget of £5 per run and £30 in total
  (section 7).
- **Internet:** the agents need it. On clusters without internet on the compute nodes, run the gate steps on the login node (section 9).

---

## 4. Quick test (5 minutes)

This runs the classic 10x PBMC 3k dataset. Use the copy in the repository, or download it:

```bash
mkdir -p ~/cellwarden-test && cd ~/cellwarden-test
curl -sL https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz | tar -xz
```

**Without an LLM.** This stops after clustering and markers, and needs no key or internet:

```bash
cellwarden run filtered_gene_bc_matrices/hg19 --name pbmc3k --llm none --no-gates --until evidence --out pbmc3k_test
```

**Full pipeline with the agents.** Run this on a node with internet, after connecting an AI backend
(section 3):

```bash
cellwarden run filtered_gene_bc_matrices/hg19 --name pbmc3k --out pbmc3k_full
```

In a terminal you see one line per finished stage. The current stage shows with its latest step,
a progress bar, the elapsed time and an estimate of the time left:

```
✓ Ingest (S0) 9.6s · peak 604.01 MB
✓ QC metrics (S1a) 23.1s · peak 1.01 GB
⠼ Normalise, PCA, batch correction, clustering  S2 · 12.3s
  ↳ harmony
  ━━━━━━━━━━━━━━━━━━━╸━━━━━━━━━━━━━━━━━━━━  5/12 stages · 50s elapsed · about 1 min left
```

Then open `pbmc3k_full/S5/report.html` in a browser. It should show about 2,500 cells after QC,
around 9 clusters, and T, B, NK, monocyte, dendritic and platelet populations, each with marker
genes and Open Targets evidence.

To check the image itself, run the built-in smoke test. It takes about 10 minutes, with a mock
LLM and offline:

```bash
apptainer exec --cleanenv cellwarden.sif bash /opt/cellwarden/containers/smoke_test.sh        # on a cluster, inside an srun / sbatch job
```

---

## 5. Run your own data

**Give the files directly.** The format is detected, and each file or folder is one sample, named
after it:

```bash
cellwarden run data/donorA.h5 data/donorB.h5 data/donorC.h5ad --meta condition=control --out results/my_study
```

**With sample information, use a sample sheet.** Any extra column (donor, condition, sex, age,
batch…) becomes cell metadata, which you can use for batch correction and comparisons:

```csv
sample_id,path,condition,donor,sex
ctrl1,data/ctrl1.h5,control,D1,F
ctrl2,data/ctrl2.h5,control,D2,M
stim1,data/stim1.h5,stimulated,D1,F
stim2,data/stim2.h5,stimulated,D2,M
```

```bash
cellwarden run samples.csv --integrate donor --out results/my_study      # Harmony batch correction on donor
```

**Full control: a config file.** Copy [`config/examples/`](../config/examples) next to your data,
edit `config.yaml` (stages, gates, DE), `samples.csv` and `qc_policy.yaml` (QC limits), then:

```bash
cellwarden doctor --config config.yaml
cellwarden run --config config.yaml
```

Every stage is checkpointed. Running the same command again skips finished, unchanged stages.
Changing a setting reruns only the stages that use it, and the ones after them.

---

## 6. Guided setup (`-i`)

Add `-i` and CellWarden asks a few questions first. Enter keeps the suggested value:

```bash
cellwarden run data/*.h5 -i --out results/my_study
```

1. **Mode:** *Automatic* uses the defaults, with adaptive QC and the agents deciding. *Guided*
   asks the questions below.
2. **QC hard limits:** minimum genes (200) and UMIs (400) per cell, maximum mitochondrial % (25).
   `off` disables a limit.
3. **Adaptive QC:** the number of MADs (3), or `off`.
4. **Doublets:** Scrublet, scDblFinder or none.
5. **Clustering:** PCs (`auto`), Leiden resolution (0.5), variable genes (2000).
6. **DE:** the method, the test, and either markers per cell type, two groups, or **choose after
   annotation** (section 8).
7. **Batch correction** with Harmony (only with more than one sample).
8. **LLM**, from the keys it finds.
9. **Review gates:** agents decide, or stop at every gate.

It shows a summary, then asks before starting. The answers are saved, and it prints a command
that repeats the run without questions. `-i` needs a terminal, and is used **without** `--config`.

---

## 7. Review gates and LLM agents

| Gate | Name on the command line | What is reviewed | Default |
|---|---|---|---|
| G1 | `qc-review` | QC decisions per sample | agents decide |
| G2 | `context-review` | tissue and disease context | stops for you |
| G3 | `annotation-review` | cell-type labels of the clusters | agents decide |
| G4 | `interpretation-review` | interpretation of the DE genes | stops for you |

The agents need an AI backend: an API key, or a local model server ([section 3](#3-connect-an-ai-backend-optional)).
Without one, every gate decides by rules and stops for you.

**Letting the agents decide at every gate.** Use `-i` and choose "Agents decide", or use a config
with:

```yaml
gates: {qc-review: auto, context-review: auto, annotation-review: auto, interpretation-review: auto}
```

Even in `auto`, a gate stops when the reviewer agent disagrees, or when the LLM call fails.

**When a gate stops.** Open its report, then approve and continue:

```bash
cellwarden approve --run-dir results/my_study --stage context-review --by "Your Name"
cellwarden resume  --run-dir results/my_study
```

The report is at `results/my_study/gates/G2/report.html`. To change the proposal, edit a copy of
`gates/G2/proposal.json` and pass `--from-file edited.json`.

**Cost.** Every call is priced and logged. The default budget is £5 per run and £30 in total
(`llm.budget`). A call that would cross it is not made, and that gate falls back to rules.

---

## 8. Differential expression

The default is Seurat **FindMarkers** with the **MAST** test. The two common setups:

| You have | You get, with no settings |
|---|---|
| one sample, or no comparison set | the **markers of each cell type** against the other cells |
| comparisons in `dge.contrasts` | each comparison, within each cell type |

**Choose the groups after annotation (recommended).** Cell types and clusters only have names
after clustering and the annotation review. With `-i`, choose *"choose after clustering and
annotation"*, or set `dge: {interactive: true}`. The run then pauses before the DE stage and asks:

```
Clustering and the cell-type annotation review are done. Choose the differential expression comparison(s).
  cell types: T cell (1361 cells), monocyte (581 cells), B cell (282 cells), NK cell (111 cells), ...
  clusters: 0 (663 cells), 1 (440 cells), 2 (417 cells), ...
  [1] markers of each cell type against the other cells
  [2] two cell types or clusters
  [3] two groups of samples (condition, donor, sex)
```

You pick groups by number or name. Group 2 can be `rest` (all other cells), and you can add more
comparisons. The choice is saved, so `resume` won't ask again.

**Comparisons in the config.** Any metadata column works for either side:

```yaml
dge:
  test: MAST                    # or wilcox (fast), LR, t, bimod, negbinom, poisson, DESeq2
  contrasts:
    - {name: stim_vs_ctrl, column: condition, numerator: stimulated, denominator: control}     # per cell type
    - {name: F_vs_M,       column: sex,       numerator: F,  denominator: M, covariates: [donor]}
    - {name: old_vs_rest,  column: age_group, numerator: [60-70, 70-80], denominator: rest, within: none}
    - {name: CD8_vs_CD4,   column: cell_type_fine, numerator: CD8 T cell, denominator: CD4 T cell, within: none}
```

- `within: none` compares all cells together. The default compares cell type by cell type.
- Covariates are used by MAST, LR, negbinom and poisson.

**Pseudobulk.** Use `method: pseudobulk` for DESeq2 on samples. You need two or more samples per
side, and the column must have one value per sample.

**Speed.** MAST takes about 1–2 minutes per cell type on PBMC 3k, and cell types run in
parallel. For hundreds of thousands of cells, use `test: wilcox` or set `max_cells_per_side: 2000`.

Each DE result lists the top 50 up and 50 down genes. Each gene comes with its Open Targets
record: associated diseases, drugs in the clinic (phase, indication, mechanism) and tractability.

---

## 9. Big datasets: compute nodes and GPUs

On many clusters (Iridis X, for example: [docs/sites/iridis.md](sites/iridis.md)) the compute nodes have no internet, but the agents and Open Targets need it. Split the
run into three steps that share the run folder; each step continues where the last one stopped:

```bash
# 1. login node: setup questions, ingest, QC and the QC review (light; needs the terminal and internet)
cellwarden run data/*.h5 -i --until qc-review --out /scratch/$USER/my_study

# 2. compute node: normalisation, batch correction, clustering, UMAP, markers, CellTypist (heavy, offline)
export CELLWARDEN_IMAGE=/path/to/cellwarden.sif CELLWARDEN_BIND=/path/to/data,/scratch   # the image or sandbox, and folders to mount
sbatch --export=ALL,RUN=/scratch/$USER/my_study conf/jobs/slurm_compute_offline.sbatch

# 3. login node, after the job: context and annotation reviews, DE + Open Targets, interpretation, report
cellwarden resume --run-dir /scratch/$USER/my_study
```

- **Resources:** edit the `#SBATCH` lines of
  [`conf/jobs/slurm_compute_offline.sbatch`](../conf/jobs/slurm_compute_offline.sbatch). The
  defaults are 8 cores, 64 GB and 4 hours on `amd`; the job's threads follow `--cpus-per-task`.
- **Following the job:** `squeue -u $USER`, `cellwarden status --run-dir ...`, or
  `<run>/dashboard.html`.
- **GPU:** in the job script, use `--partition=l4 --gres=gpu:1` and add `--nv` after
  `apptainer exec`. In a direct run, add `--nv` too.
- **Long runs on the login node:** `cellwarden run ... --detach` runs inside tmux and survives a
  dropped SSH connection.

---

## 10. Results

Everything is in the run folder (`--out`):

| Where | What |
|---|---|
| `S5/report.html` | **Start here**: one self-contained page with QC, clusters, markers, annotation, DE, Open Targets, gate decisions, LLM cost and software versions |
| `S5/explorer.html` | Interactive UMAP: colour by cell type, cluster, sample or gene |
| `dashboard.html` | Run progress: stages, timings, memory, gates (updated after every stage) |
| `S4/dge/<comparison>__<group>.tsv.gz` | Full DE tables: `gene, baseMean, log2fc, pvalue, padj, direction, pct_1, pct_2` |
| `S4/opentargets_summary.json` | Open Targets records of the top DE genes |
| `S3/markers.tsv.gz`, `S3/annotation_consensus.tsv` | Cluster markers and cell-type calls |
| `S2/object.rds` | The final Seurat object (`readRDS()` in R) |
| `gates/G1..G4/` | Each gate's evidence, proposal, review, decision and `report.html` |
| `logs/<stage>.log` | Full log of each stage |
| `config/` | The exact settings used (`config.resolved.yaml`), for reproducibility |

---

## 11. Everyday commands

| Task | Command |
|---|---|
| Check the install and keys | `cellwarden doctor` |
| Run | `cellwarden run <files or samples.csv> --out DIR` |
| Run with questions | `cellwarden run <files> -i --out DIR` |
| Stop after a stage | `--until clustering` (stage names below, or IDs like `S2`) |
| Continue / rerun after a change | `cellwarden resume --run-dir DIR` |
| Progress and timings | `cellwarden status --run-dir DIR` |
| Approve a gate | `cellwarden approve --run-dir DIR --stage context-review --by NAME` |
| Run in the background (tmux) | add `--detach` |
| One line per stage (for logs) | add `--progress plain` |

**Stage names:** `ingest`, `qc-metrics`, `doublets`, `qc-filter`, `qc-review`, `clustering`,
`evidence`, `context-review`, `annotation-review`, `dge`, `interpretation-review`, `report`.

---

## 12. Troubleshooting

| You see | Cause and fix |
|---|---|
| `fusermount: No such file or directory` when starting a `.sif` | The site has no FUSE. Use the sandbox folder (section 2), or add `--unsquash`. `--unsquash` unpacks into RAM, so jobs need ~96 GB. |
| `bash: executable file not found` in a sandbox | It was unpacked straight onto GPFS and lost its execute permissions. Unpack on `/tmp` and copy with `rsync -a` (section 2B). |
| `doctor` says `no API key set`, or every gate runs rules-only | The key didn't reach the container: add `--env-file .env` (section 3, step 2). Check the file has `NAME=value` lines without quotes or `export`. |
| A gate stops with `fallback: true` | The LLM call failed. See `logs/G<n>.log`: e.g. `HTTP 410` means the model was retired (`--llm-model <current model>`), `401` means a wrong key, and no internet happens on compute nodes. |
| A gate stops with `reviewer: escalate` | The reviewer agent disagreed, or found a problem. Read `gates/G<n>/report.html`, then approve or edit. |
| `the worker made no valid proposal: ... must be null or match EFO_/MONDO_` | The agent gave an invalid ID; the policy check refused it. Approve the gate by hand, or rerun it. |
| `-i needs a terminal` | `-i` was used in a batch job or a pipe. Run it on the login node, then submit the heavy part (section 9). |
| The run waits before DE (`waits for the comparisons`) | You chose "after annotation" and the run has no terminal. `cellwarden resume` in a terminal asks. |
| A stage `FAILED` | The last 20 lines of the log are printed. The full log is `logs/<stage>.log`. Fix the cause, then `cellwarden resume`. |
| `disk quota exceeded` (home or scratch) | Many clusters limit the **number of files** as well as space. Keep runs, images and caches on a project folder, not in home. |
| DE is slow | `dge.test: wilcox`, or `dge.max_cells_per_side: 2000` (section 8). |

---

## 13. FAQ

**What do the LLM agents see?** Aggregated evidence only: QC summaries, cluster statistics,
marker lists and DE summaries, never the expression matrix. Free-text sample metadata is passed
as data, never as instructions. Every call is logged in `llm/agent_log.jsonl`.

**Can I run without an LLM?** Yes, with `--llm none`. The rules make every decision, and each
gate stops for your approval. Use `--no-gates` to skip the gates entirely.

**Is it reproducible?** The exact settings, package versions, random seed and every decision are
recorded in the run folder. Rerunning with the same config skips everything unchanged.

**Mouse data?** Not supported: the annotation references and Open Targets are human.

**Where is the full reference?** [User guide](USER_GUIDE.md) (configuration, sites, schedulers,
containers), [Pipeline reference](PIPELINE.md) (every stage and setting), [Install](INSTALL.md)
(native installs, building, GPUs), and [Architecture](ARCHITECTURE.md).
