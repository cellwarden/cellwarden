# CellWarden

**A single-cell RNA-seq pipeline whose decisions are guarded: rules decide, LLM agents advise,
a reviewer agent checks, and you approve. Everything is logged.**

[![Release](https://img.shields.io/github/v/release/cellwarden/cellwarden)](https://github.com/cellwarden/cellwarden/releases)
[![Docker pulls](https://img.shields.io/docker/pulls/ad2n15/cellwarden)](https://hub.docker.com/r/ad2n15/cellwarden)
[![Licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cellwarden/cellwarden/blob/main/examples/colab/cellwarden_quickstart.ipynb)

Give it count files (10x, h5ad, Seurat, loom, csv). CellWarden then runs a complete Seurat-based
analysis:
- QC with doublet removal, normalisation, batch correction (Harmony), clustering, UMAP;
- cell-type annotation against 72 human cell types;
- differential expression (FindMarkers / MAST or pseudobulk DESeq2);
- drug and disease evidence for the DE genes from Open Targets;
- one self-contained HTML report.

At four **review gates** (QC, tissue context, annotation, DE interpretation), LLM agents propose
decisions within policy limits. A separate reviewer agent checks them, and the run continues or
stops for you.

**See it on real data:** [PBMC 3k report](https://cellwarden.github.io/cellwarden/examples/pbmc3k/report.html) ·
[interactive UMAP](https://cellwarden.github.io/cellwarden/examples/pbmc3k/explorer.html) ·
[run dashboard](https://cellwarden.github.io/cellwarden/examples/pbmc3k/dashboard.html)

```
ingest → QC metrics · doublets · QC rules → [QC review] → normalise · PCA · Harmony · clustering · UMAP
→ markers + cell-type evidence → [context review] → [annotation review]
→ differential expression + Open Targets → [interpretation review] → report + explorer
```

## Install

One image has everything: R 4.6 with Seurat 5 and Bioconductor, the Rust engine **scfast**,
RAPIDS for NVIDIA GPUs, and CellTypist. There's nothing to compile, and a GPU is optional.

```bash
docker pull ghcr.io/cellwarden/cellwarden:latest                        # workstation, cloud, macOS / Windows (Docker Desktop)
apptainer pull cellwarden.sif docker://ghcr.io/cellwarden/cellwarden:latest   # HPC (Docker Hub copy: docker://ad2n15/cellwarden:latest)
```

No install at all: [open the Colab notebook](https://colab.research.google.com/github/cellwarden/cellwarden/blob/main/examples/colab/cellwarden_quickstart.ipynb).

## Run it

```bash
cellwarden doctor                                   # checks the install and your LLM key
cellwarden run data/*.h5 -i --out results/study     # -i: a few questions first, Enter keeps the defaults
```

In a terminal you see each stage finish and a live progress bar with the time left. With `-i`,
CellWarden can pause before differential expression and let you pick the groups from the real
cell types. When a gate wants your decision:

```bash
cellwarden approve --run-dir results/study --stage context-review --by "Your Name"
cellwarden resume  --run-dir results/study
```

Then open `results/study/S5/report.html`.

**Step by step**, covering Docker and Apptainer commands, connecting an LLM (Claude, OpenAI,
NVIDIA or your own server), HPC clusters and troubleshooting:
**[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**.

## What makes it different

| | |
|---|---|
| **Guarded decisions** | QC thresholds, tissue context, cell-type labels and DE interpretations go through gates. Rules give the baseline; an agent may only change it within `qc_policy.yaml` and the gate's policy, a reviewer agent checks, and you approve when it matters. Every proposal, review and decision is a JSON file with a report. |
| **Fast** | The Rust engine scfast does preprocessing, PCA, neighbours, Leiden and Harmony in process: Harmony on 200k cells takes 0.4 s (R harmony: 24 s). The GPU path uses RAPIDS. |
| **Any data, any machine** | 10x h5 / mtx, h5ad, Seurat rds / qs2, loom and csv are detected automatically. Runs on a laptop, workstation, cloud VM or HPC (Slurm, PBS, LSF, SGE), with checkpoints and `resume`. |
| **Your LLM, capped cost** | Claude, OpenAI, NVIDIA NIM or a local OpenAI-compatible server (vLLM, Ollama). A budget refuses calls beyond £5 per run / £30 in total. Agents see aggregated evidence, never the count matrix ([SECURITY.md](SECURITY.md)). |
| **Readable results** | One self-contained report (QC, clusters, markers, annotation, DE, Open Targets drugs and diseases, gate decisions, LLM cost, software versions), an interactive UMAP explorer, and a live run dashboard. |

## Documentation

| | |
|---|---|
| [Getting started](docs/GETTING_STARTED.md) | Requirements, install, quick test, how-tos, troubleshooting, FAQ |
| [User guide](docs/USER_GUIDE.md) | Configuration, gates, sites and schedulers, containers |
| [Pipeline reference](docs/PIPELINE.md) | Every stage and setting |
| [Install details](docs/INSTALL.md) | Native installs, building the image, GPUs |
| [Architecture](docs/ARCHITECTURE.md) | How the pieces fit together |
| [Rust engine](docs/RUST_ENGINE.md) | scfast, with Python and R bindings |
| [Benchmarks](bench/results/FINAL.md) | Speed and parity with Seurat |
| [Example site: Iridis X](docs/sites/iridis.md) | Running on a Slurm cluster without internet on compute nodes |

## Status and limitations

Version 0.3.0 is the first public release. Please [tell us how it went](https://github.com/cellwarden/cellwarden/issues/new?template=tested.yml).
- **Human data only:** the annotation references and Open Targets are human.
- **Image size:** the image is large (6.8 GB) because it carries R, Bioconductor and RAPIDS.
- **Tested:** the CPU path end to end (PBMC 3k, synthetic multi-sample data); 570 Python tests,
  1,219 R expectations, and a full pipeline test with a mock LLM.
- **Partly tested:** the GPU path, with the scfast CUDA backend benchmarked on L4 and H100
  ([benchmarks](bench/results/FINAL.md)).
- **LLM agents:** they make mistakes. That is why their proposals pass policy checks and a reviewer,
  and why the gates for context and DE interpretation stop for you by default.

## Citing and credits

CellWarden was started by **Mukesh Agrawal, Ahmed Dawoud and David Williams** at the
*Reinvent the Wheel* AI hackathon (October 2026). If you use it, please cite it ([CITATION.cff](CITATION.cff));
see [AUTHORS](AUTHORS) and [NOTICE](NOTICE).

It builds on Seurat, harmony, presto, BPCells, DESeq2, limma, MAST, scDblFinder, CellTypist,
scanpy, RAPIDS and Open Targets. Please cite them too when you use their results.

Contributions are welcome: [CONTRIBUTING.md](CONTRIBUTING.md). Licence: [Apache-2.0](LICENSE).

The [tutorial scripts](examples/tutorials/) that the project grew from compare plain Seurat and
Scanpy workflows on PBMC 3k.
