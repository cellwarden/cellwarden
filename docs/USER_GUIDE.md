# CellWarden user guide

For a shorter, task-oriented introduction see [GETTING_STARTED.md](GETTING_STARTED.md).

How to run the pipeline on your own machine or on an HPC, read its report, and move it to a
new site. The design is in [`DESIGN.md`](DESIGN.md); the stage-by-stage contracts are in
[`stages-spec.md`](stages-spec.md) (S0–S2) and [`agents-spec.md`](agents-spec.md) (gates,
S3–S5, portability). The `cellwarden` command is the only thing you run.

**The principle behind every decision the pipeline makes:** rules decide, agents advise,
reviewers check, humans approve, everything is logged. No agent changes data directly; it makes
a proposal that code validates against policy, and you can always see (and overrule) the result.

## 1. Quick start

```bash
# 1. install (see README.md), then check the node you are on
cellwarden doctor --config config.yaml

# 2. run; --detach keeps it alive inside tmux when your SSH session drops
cellwarden run --config config.yaml --detach

# 3. follow it: text, or open <run_dir>/dashboard.html (rewritten after every stage; refreshes itself)
cellwarden status --run-dir /scratch/$USER/cellwarden/my_study
cellwarden dashboard --run-dir /scratch/$USER/cellwarden/my_study --watch 30   # optional: keep regenerating

# 4. if a gate stops the run for your approval (exit code 3), it prints what to do:
cellwarden approve --run-dir /scratch/$USER/cellwarden/my_study --stage context-review [--from-file edited.json] [--by you]
cellwarden resume  --run-dir /scratch/$USER/cellwarden/my_study

# 5. open the result
<run-dir>/S5/report.html      # everything, one self-contained page
<run-dir>/S5/explorer.html    # interactive UMAP
```

Stages run in the order below. Every stage is checkpointed in `state.json` with a fingerprint of
its inputs and settings, so `cellwarden run` again with the same config skips what is done, and a
config change reruns only the stages that use the changed settings and those after them.

The terminal, `cellwarden status` and the dashboard show each stage's name; on the command line
(`--until`, `cellwarden approve --stage`) the name, the title or the short ID all work, in any case
(`--until clustering`, `--until S2`). Folders and logs keep the short IDs (`S2/`, `logs/S2.log`).

| ID | Name | What it does |
|---|---|---|
| S0 | `ingest` | Reads every sample (10x h5, 10x mtx folder, h5ad, Seurat rds/qs2, loom, csv/tsv), harmonises genes, merges |
| S1a | `qc-metrics` | Per-cell QC metrics |
| S1b | `doublets` | Doublet scores and calls, pre-clustering |
| S1c | `qc-filter` | Rules-based QC decisions (`keep` / `exclude` / `review`) |
| G1 | `qc-review` | QC agent reviews the rules' decisions (gate) |
| S2 | `clustering` | Filter, normalise, variable genes, PCA, optional Harmony, neighbours, Leiden, UMAP; CPU (Seurat) or GPU (rapids-singlecell) |
| S3 | `evidence` | Markers, signature scores, CellTypist (if installed), consensus annotation in code |
| G2 | `context-review` | Context agent: tissue and disease candidates (gate) |
| G3 | `annotation-review` | Annotation agent (gate) |
| S4 | `dge` | Differential expression (FindMarkers with MAST by default, or pseudobulk DESeq2) per comparison and cell type; markers of each cell type when no comparison is set; Open Targets evidence for the DE genes |
| G4 | `interpretation-review` | Interpretation agent writes labelled hypotheses, the reviewer checks every cited ID (gate) |
| S5 | `report` | `report.html` and `explorer.html` |

### Progress

In a terminal, `cellwarden run` and `cellwarden resume` show a live display: finished stages with their
time and memory, the current stage and its latest step (from the stage's log), a bar, the elapsed
time and an estimate of the time left. The estimate appears once ingest has counted the cells;
it starts from built-in timings and adapts to your machine (recent runs are kept in
`~/.cellwarden/stage_times.json`). In logs, pipes, `--detach` and scheduler jobs the output is one
line per stage instead. `--progress live|plain` forces either.

### Guided setup: `cellwarden run -i`

`cellwarden run <data> -i` asks a few questions before the run; Enter keeps each suggested value:

1. mode: **automatic** (defaults, adaptive QC, the review agents decide at each gate) or **guided**;
2. QC hard limits: minimum genes (200) and UMIs (400) per cell, maximum mitochondrial % (25);
   `off` disables a limit;
3. adaptive QC: the number of MADs (3), or `off` for hard limits only;
4. doublets: Scrublet, scDblFinder or none, and whether called doublets are removed;
5. clustering: PCs (`auto`), Leiden resolution (0.5), highly variable genes (2000);
6. batch correction with Harmony (only with more than one sample);
7. differential expression: FindMarkers (MAST, wilcox, LR, ...) or pseudobulk, and either the
   markers of each cell type, two groups of any column (condition, donor, sex, ...), or **choose
   after clustering and annotation** from the actual cell types and clusters (the run asks before S4);
8. the LLM for the review agents, from the API keys that are set;
9. the review gates: agents decide, stop at every gate, or the defaults.

It then shows a summary and asks to start. The answers are saved as
`<run-dir>/config/wizard.yaml` (plus `wizard_qc_policy.yaml`), and the command that repeats the
run without questions is printed. `-i` needs a terminal and is used without `--config`.

## 2. Configuration

Three files, usually next to your data (relative paths resolve against the folder holding
`config.yaml`):

| File | What goes in it |
|---|---|
| `config.yaml` | Project, run directory, `site`, `execution`, resource caps, stage settings, `llm`, `gates`, `dge`, `opentargets`, `report` |
| `samples.csv` | `sample_id`, `path`, optional `format`; any other column (donor, condition, batch, ...) becomes cell metadata and can be used by `dge` |
| `qc_policy.yaml` | Hard limits, adaptive MAD bounds, doublet handling, protected populations, review triggers |

Start from [`config/examples/`](../config/examples). Settings are validated before anything
runs and every problem is reported at once. The sections added by build steps 3–7:

```yaml
engine: auto                     # auto | scfast | seurat   (section 5b): which engine runs the CPU steps
site: iridisx                    # a name in conf/sites/<name>.yaml, or a path; supplies defaults for
                                 # execution, resources, container, scheduler, scratch (config.yaml wins per key)
execution: direct                # direct | slurm | pbs | lsf | sge | split   (section 5)

gates:                           # who gets the last word at each gate (section 4)
  G1: auto                       # auto: agent proposal approved by the reviewer -> run continues
  G2: review                     # review: run stops for a human even when the reviewer approves
  G3: auto                       # escalate: always a human
  G4: review

llm:                             # optional: without it the first key set is used (Claude, OpenAI, NVIDIA), else rules-only
  default:
    provider: auto               # auto | anthropic | openai | nvidia | openai_compatible (vLLM, Ollama, local NIM) | mock (tests)
    # model: claude-haiku-4-5    # default per provider: claude-haiku-4-5, gpt-4.1-mini, nvidia/nemotron-3-ultra-550b-a55b
    # api_key_env: ANTHROPIC_API_KEY   # the NAME of the variable holding the key; the key itself is never written anywhere
    # base_url: http://localhost:11434/v1     # openai_compatible only (anthropic/openai/nvidia know theirs)
  roles: {}                      # per-role overrides: qc_worker, context_worker, annotation_worker, interpretation_worker, reviewer
  budget:
    per_run_limit: 5             # GBP; a call that would cross a limit is not made
    total_limit: 30              # GBP across all runs (ledger in ~/.cellwarden/ledger.json)
    usd_to_gbp: 0.78
  replay: false                  # true: reuse logged outputs for identical prompts (no calls, no cost)

dge:
  method: findmarkers            # findmarkers (Seurat, compares cells) | pseudobulk (compares samples)
  test: MAST                     # FindMarkers test: MAST, wilcox, wilcox_limma, LR, t, bimod, negbinom, poisson, DESeq2
  pseudobulk_method: deseq2      # method pseudobulk: deseq2 | limma_voom
  markers: auto                  # auto: with no contrasts, markers of each cell type vs the other cells; true: always too
  interactive: false             # true: stop before S4, once the cell types are known, and ask for the comparisons
  group_by: cell_type_broad      # the groups tested separately (cell_type_fine, cluster, ...)
  contrasts:                     # each side is one or more values of any cell metadata column
    - {name: disease_vs_control, column: condition, numerator: disease, denominator: control, covariates: [batch]}
    - {name: female_vs_male, column: sex, numerator: F, denominator: M}
    - {name: old_vs_young, column: age_group, numerator: [60-70, 70-80], denominator: rest, within: none}
    - {name: B_vs_T, column: cell_type_broad, numerator: B cell, denominator: T cell, within: none}
  min_pct: 0.1                   # FindMarkers: genes expressed in at least this fraction of one side
  logfc_threshold: 0.25          # FindMarkers: minimum |log2 FC| to test a gene
  min_cells_per_side: 10         # FindMarkers: fewer cells on a side skips the group
  max_cells_per_side: null       # FindMarkers: downsample each side to this many cells (speed)
  sample_col: sample_id          # pseudobulk: the sample column (one pseudobulk per group x sample)
  min_cells_per_sample: 20       # pseudobulk: smaller pseudobulk samples are dropped
  min_replicates: 2              # pseudobulk: samples needed on each side
  padj: 0.05
  lfc: 0.5

opentargets:
  enabled: true
  offline: false                 # true: read the cache only, never the network
  max_genes_per_direction: 50
  cache_dir: refs/opentargets    # shared across runs; entries never expire

report:
  explorer_max_cells: 100000     # explorer.html subsamples above this
  expr_genes: 200                # variable genes exported for colour-by-gene
  expr_max_cells: 20000          # cells with expression values in the explorer
  top_markers: 10                # markers per cluster shown in report.html
```

**Claude, OpenAI or NVIDIA.** Export one key and the agents use it:
`ANTHROPIC_API_KEY` (Claude Haiku 4.5), `OPENAI_API_KEY` (gpt-4.1-mini) or `NVIDIA_API_KEY`
(NVIDIA NIM, `nvidia/nemotron-3-ultra-550b-a55b`). With several keys, Claude is preferred, then OpenAI,
then NVIDIA. Pick one per run with `cellwarden run ... --llm nvidia --llm-model <model>`, or in the config:

```yaml
llm:
  default: {provider: nvidia}                 # model, endpoint and key variable filled in
  roles:
    reviewer: {provider: anthropic}           # e.g. a different provider for the reviewer
```

A local NIM or vLLM server: `{provider: openai_compatible, base_url: http://localhost:8000/v1, model: ...}`.
The NVIDIA preset asks for JSON with `json_object` (not every NIM model supports `json_schema`); use
a NIM model that supports tool calls. `cellwarden doctor` shows which provider and model were chosen and
makes a tool-use test call.

### Differential expression (S4)

By default S4 runs Seurat `FindMarkers` with the MAST test. Each comparison in `dge.contrasts`
names a cell metadata `column` and the values on each side:
- **Columns:** any `samples.csv` column (condition, donor, sex, age group, sample, batch, ...) becomes
  cell metadata, so it works, as do `cell_type_broad`, `cell_type_fine` and `seurat_clusters`.
- **Sides:** `numerator` and `denominator` take one value or a list; `denominator: rest` means every
  other value.
- **Groups:** the comparison runs within each cell type (`group_by`) unless `within: none`, which
  uses all cells together.
- **Covariates:** these are passed to MAST, LR, negbinom and poisson as `latent.vars`.

With no comparisons set, S4 finds the markers of each cell type against the other cells, so a
single-sample study still gets DE genes and Open Targets evidence. Groups are tested in parallel
(`resources.threads`). `test` picks another FindMarkers test (`wilcox` is much faster than MAST on
large data).

**Choosing after annotation.** Cell types and clusters only have names once clustering (S2) and
the annotation review (G3) are done. With `dge.interactive: true`, the run stops just before S4,
which is what `cellwarden run -i` sets when you choose "after clustering and annotation" or type a
cell-level column. It then lists the cell types, fine cell types and clusters with their cell
counts, and the sample-sheet groups. You pick markers per cell type, two cell types or clusters
(by number, or `rest`), or two groups of samples (within each cell type or across all cells), and
can add more comparisons. The choice is written into the run's config and
`config/dge_choice.yaml`, so `resume` keeps it; delete that file to be asked again. Without a
terminal (scheduler jobs, `--detach` with nohup) the run waits before S4, and `cellwarden resume` in
a terminal asks.

`method: pseudobulk` (DESeq2, or `pseudobulk_method: limma_voom`) sums counts per cell type and
sample and compares samples instead. Its column must have one value per sample, and each side
needs `min_replicates` samples. Configs that set `method: deseq2` or `limma_voom` keep running
pseudobulk.

### Engine: Rust, Seurat or GPU

`engine` picks the code that does doublets, pre-clustering, PCA, neighbours and Leiden:

| `engine` | What runs | Exchange `.h5ad` files |
|---|---|---|
| `auto` (default) | Rust `scfast` when `scfastR` is installed and the plan is CPU; on a GPU plan, `engine_gpu` decides | none on the Rust paths |
| `scfast` | Rust only. Errors if `scfastR` is missing, or with `device: gpu` unless `engine_gpu: scfast` | none |
| `seurat` | Seurat on CPU; rapids-singlecell through the exchange when the plan is GPU | GPU path only |

On a **GPU plan** a second key, `engine_gpu: auto | scfast | rapids` (default `auto`), chooses
between the Rust CUDA backend (`gpu_scfast`: PCA, kNN and Leiden on the GPU in process, UMAP on
the CPU, no exchange files) and rapids-singlecell through the exchange (`gpu_exchange`). `auto`
takes the Rust backend only when `scfastR` was built with the `cuda` feature **and** a device is
visible (`cellwarden doctor` prints both), otherwise rapids. The CUDA backend has not yet been run
on real hardware — see [`INSTALL.md`](INSTALL.md) for the acceptance tests to run on the L4
before trusting `gpu_scfast` on a real dataset.

On the Rust path S1a scores doublets itself (a Scrublet-style detector) and S2 runs entirely in R,
so Python and Scanpy are not involved and nothing is written for the exchange. Normalisation, HVG
selection and the scaling statistics are one fused pass (`PreprocessFast`; HVGs agree with Seurat's
vst at ≥ 99.8%). `S2/summary.json` records `path` (`cpu_scfast`, `gpu_scfast`, `cpu_seurat`,
`gpu_exchange`, `exchange_cpu`), `engine`, `scfast_version`, `scfast_device` and `preprocessing`. Doublet calls differ slightly from Scrublet's (the detector is a reimplementation),
and Harmony without the R `harmony` package falls back to the exchange route. `cellwarden doctor` prints
the engine `auto` would choose. Install steps, including Rust, are in [`INSTALL.md`](INSTALL.md).

## 3. The run directory

```
<run-dir>/
  config/            copies of your files + config.resolved.yaml (what the stages actually read) + hashes.json
  state.json         per stage: status, start/end, wall time, peak memory, exit code, input fingerprint
  logs/<STAGE>.log
  S0/  object.rds  ingest_report.json
  S1/  qc_metrics.tsv.gz  qc_decisions.tsv.gz  qc_summary.json
  S2/  object.rds  summary.json
  S3/  markers.tsv.gz  cluster_summary.json  signatures.tsv.gz  celltypist.tsv.gz  evidence_bundle.json
       annotation_consensus.tsv  annotation.tsv (after G3)
  S4/  pseudobulk/  dge/<contrast>__<group>.tsv.gz  dge_summary.json  opentargets_summary.json
  S5/  report.html  explorer.html  umap.tsv.gz  expr_subset.tsv.gz  export.json
  gates/G1..G4/      bundle.json  proposal.json  review.json  decision.json  report.html
  llm/               ledger.json (cost)  agent_log.jsonl (every call, replayable)
  submit/            <scheduler>.sh and last_job.json when a job was submitted (section 5)
```

Put the run directory on scratch; results you keep belong on project storage.

### Watching a run: the dashboard

`<run_dir>/dashboard.html` is one self-contained page (no network, no JavaScript) that `cellwarden run`
rewrites after every stage change. It shows the pipeline strip (S0 … S5 with the four gates)
coloured by status, a wall-clock timeline with the waiting time at each gate hatched, and a card
per stage and gate: wall time, peak memory, cells in and out, QC keep/review/exclude, engine and
path, clusters, the S2 step timings, contrasts, the gate mode, who decided, fallback, reviewer
verdict and a link to that gate's `report.html`. Open it over SMB/`scp`, or in a browser on the
login node; while the run is live it reloads itself every 30 s. `cellwarden dashboard --run-dir D`
regenerates it by hand, and `--watch N` keeps doing so until the run finishes. The tail of the log
is shown inline for a running, waiting or failed stage. Set `CELLWARDEN_NO_DASHBOARD=1` to disable it.

## 4. Gates and approvals

At each gate the pipeline builds `bundle.json` (aggregated tables only, never per-cell data or
free-text metadata), the rules engine makes a baseline decision, the worker agent proposes,
code validates the proposal against policy, and a separate reviewer agent (which never sees the
worker's rationale) returns `approve`, `request_changes` or `escalate`. What happens next
depends on `gates.<G>`:

| Mode | Outcome |
|---|---|
| `auto` | reviewer approves -> `decision.json` written, run continues; otherwise the run stops for you |
| `review` | the run always stops for you with the proposal and review laid out |
| `escalate` | same, and the reviewer is told to be strict |

When a run stops at a gate it exits with code 3, writes `gates/<G>/report.html`, and prints:

```
cellwarden approve --run-dir D --stage context-review [--from-file edited.json] [--by NAME]
cellwarden resume  --run-dir D
```

Open the gate's `report.html`, optionally copy `decision.json`, edit it and pass it with
`--from-file`; the edited decision is validated against the same policy as the agent's. The
decision records `decided_by: human:<name>`. Without an `llm` section, or when the budget guard
refuses a call, the rules baseline becomes the decision (`fallback: true`) and the gate goes to
`review` mode, so nothing is ever applied silently.

Every LLM call is logged to `llm/agent_log.jsonl` with hashes of the prompt and input, the
tokens used and the cost in GBP; `llm/ledger.json` holds the totals. `cellwarden status` shows the
spend so far, and `report.html` has a per-gate, per-role cost table.

## 5. Execution modes and sites

`cellwarden` drives the stages itself. Where they run is a configuration switch, not a different
pipeline:

| `execution` | Where the stages run | Use when |
|---|---|---|
| `direct` | on the current node (inside `tmux` with `--detach`), with the resource caps and `nice` from `resources` | the node with internet is a login node or workstation (Iridis X default) |
| `slurm`, `pbs`, `lsf`, `sge` | one scheduler job on a compute node that has internet | a site's compute nodes have internet |
| `split` | heavy stages (S0–S4) as an offline scheduler job; gates, Open Targets and the report on the node you submit from | a site does not allow compute on its internet node |

`cellwarden run --execution slurm` (or `execution: slurm` in the config or site file) prepares the
run directory, writes `<run-dir>/submit/slurm.sh` from `conf/templates/slurm.sh.j2` with the
`scheduler` settings (partition, account, time, cpus, mem_gb, gpus, extra directives), and
calls `sbatch`. The job runs `cellwarden resume --run-dir D --execution direct`, so the job and your
login shell share one `state.json`: `cellwarden status` works from either, and `cellwarden resume` after
the job (or after a gate) continues from wherever it stopped. The API key is never written into
the script; the scheduler propagates your environment (`--export=ALL`, `-V`). Set
`container.runtime` and `container.image` to run the job inside the Apptainer/Docker image.

`split` submits the stages up to S4 with `--no-gates` (gate stages are recorded as skipped and
S2 uses the rules' QC baseline), then prints the `cellwarden resume` command to run on the
submitting node once the job has finished: it runs the skipped gates in place, Open Targets and
S5. With `execution: split` and a gate mode of `review`, expect the run to stop at each gate
for your approval as usual.

### Site files

A site is one small YAML file, `conf/sites/<name>.yaml`, that supplies defaults for
`execution`, `resources`, `container`, `scheduler` and `scratch`. Three are shipped:

| Site | For |
|---|---|
| `iridisx` | the Iridis X L4 login node: `direct`, 8 threads, 64 GB, 20 GB GPU cap, `nice 10`, Apptainer with `--nv`; `scheduler.partition: gpu` for `slurm`/`split` |
| `workstation` | your own machine or VM: no scheduler, no container, higher caps, `nice 0` |
| `docker` | inside the CellWarden Docker image with `--gpus all` |

Point `config.yaml` at one with `site: iridisx` (or a path). Anything set in `config.yaml`
overrides the site file key by key. Moving to a new HPC means writing a new site file, not
touching pipeline code.

### `cellwarden init-site`

On a new machine, let the pipeline draft the site file:

```bash
cellwarden init-site                                    # print the draft
cellwarden init-site --write conf/sites/mycluster.yaml --name mycluster
cellwarden init-site --json                             # the raw detection
cellwarden init-site --no-internet-check                # skip the HTTPS probes (offline node)
```

It detects the scheduler (`sbatch` / `qsub` / `bsub`, telling PBS and SGE apart), the GPU and
its memory (`nvidia-smi`), the container runtime (`apptainer` / `singularity` / `docker`), CPU
count and RAM, scratch candidates, and whether HTTPS reaches the LLM endpoint and Open Targets.
It then drafts conservative values: half the node when a scheduler is present (a shared node),
`--nv` when Apptainer and a GPU are both there, `split` when there is a scheduler but the LLM
endpoint is not reachable from this node. Review the draft, fill in `scheduler.partition` and
`account`, and point `config.yaml` at it.

### `cellwarden doctor`

```bash
cellwarden doctor --config config.yaml
```

Preflight checks, grouped: R and Python packages (required and optional), GPU, write access to
the run directory and scratch, the site file (valid, known keys), the execution mode (submit
command on PATH), the container runtime and image, the LLM API key (present or not; the key is
never printed), the LLM endpoint, a 1-token tool-use test (skipped with a note when no key is
set), the budget remaining across runs, and Open Targets (reachable, or `offline`). Required
checks failing give exit code 1; the LLM and Open Targets checks are informational, since the
pipeline degrades to rules-only gates and cached evidence without them. `CELLWARDEN_DOCTOR_OFFLINE=1`
skips the network probes (the tests set it).

### Containers

`containers/cellwarden.def` (Apptainer) and `containers/Dockerfile` build the same image (CUDA 12
base, R 4.4 with Seurat 5, BPCells, presto, DESeq2, limma, harmony, leidenbase; Python 3.12 with
CellWarden, rapids-singlecell, CellTypist, jinja2, matplotlib):

```bash
apptainer build cellwarden.sif containers/cellwarden.def
docker build -t cellwarden:latest -f containers/Dockerfile .
```

Set `container.runtime: apptainer` and `container.image: /path/to/cellwarden.sif` (with
`options: ["--nv"]` for the GPU) in the site file; scheduler jobs then run inside it.

## 6. The report and the explorer

`S5/report.html` is a single self-contained file (inline CSS and JavaScript, plots as inline
SVG or base64 PNG, no external resources), so it can be emailed or kept with the results. Its
sections, in order:

| Section | Contents |
|---|---|
| Run summary | project, run id, samples, execution mode and site, the per-sample ingest table (format, cells, genes, UMIs, warnings), and the stage table from `state.json` (status, start, wall time, peak memory, exit code) |
| Resources | caps in force, host, container, the device and memory plan S2 chose and why, peak memory over all stages |
| Quality control | per-sample table (kept / review / excluded, thresholds actually applied), decisions and reason-code charts, protected populations, triggered review rules, per-sample histograms of genes, UMIs and % mito with the thresholds drawn |
| Clusters and annotation | UMAP coloured by cluster, by broad and fine cell type, by sample and by condition; cells per cluster; per-cluster medians, labels with CL ids, confidence and status, top markers |
| Markers | top N markers per cluster (log2FC, % in, % out, adjusted p) |
| Differential expression | per contrast and group: method used, samples per side, tested / up / down, top genes, why a group was skipped; a bar chart per contrast |
| Open Targets evidence | per contrast, group and direction: each DE gene with Ensembl id, known drugs (phase, disease), top associated diseases, tractability; offline badge when the cache alone was used |
| Gate decisions | one card per gate: mode, who decided, reviewer verdict and findings, policy check, the agent's rationale (advisory), changes versus the proposal, the decision JSON, link to the gate's own report; "waiting for a human" when a decision is pending |
| LLM cost | provider, number of calls (and replays), total GBP against the per-run and total limits, a per-gate / per-role token and cost table from `llm/ledger.json` |
| Software | CellWarden, Python and R package versions recorded by the stages |

A section whose inputs are missing (a stage that has not run, or was skipped) says so instead of
failing, so the report can be built at any point: `python -m cellwarden.stages.s5_report --run-dir D
--skip-export` rebuilds it from what is there without rerunning the R export.

`S5/explorer.html` draws the UMAP on a canvas from data embedded in the page: colour by cluster,
cell type, sample, condition or any other sample-sheet column, by QC metrics, or by the
expression of any of the exported variable genes (type the name). Hover shows the cell, the
legend toggles groups, wheel zooms, drag pans, and a button saves the view as PNG. Above
`report.explorer_max_cells` (100k) cells a seeded subsample is embedded, keeping the cells that
have expression values first.

Each gate's `gates/<G>/report.html` is built from the same templates and ends with the
`cellwarden approve` command for that gate.

## 7. Tests

```bash
PYTHONPATH=python python -m pytest -q tests/python     # unit tests, all offline
tests/integration/run_stages.sh                        # S0-S2 end to end, CPU and GPU path
tests/integration/run_full.sh                          # S0-S5 with a mock LLM and offline Open Targets
```

`run_full.sh` prints a PASS / FAIL / SKIP table; a stage that is not registered in the runner yet
is SKIP, so the script passes on a partial tree. Details in [`tests/README.md`](../tests/README.md).
