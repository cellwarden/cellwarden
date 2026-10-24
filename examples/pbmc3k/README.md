# Example: 10x PBMC 3k

The output of one CellWarden run on the classic 10x Genomics PBMC 3k dataset (2,700 peripheral
blood mononuclear cells, one sample), with every review gate on `auto` and NVIDIA NIM agents.

| File | What it is |
|---|---|
| [report.html](https://cellwarden.github.io/cellwarden/examples/pbmc3k/report.html) | The full report: QC, clusters, markers, cell types, differential expression, Open Targets evidence, gate decisions, LLM cost, software |
| [explorer.html](https://cellwarden.github.io/cellwarden/examples/pbmc3k/explorer.html) | Interactive UMAP, coloured by cell type, cluster or any gene |
| [dashboard.html](https://cellwarden.github.io/cellwarden/examples/pbmc3k/dashboard.html) | Stage timings, memory and gate decisions |
| [gates/G1 to G4](https://cellwarden.github.io/cellwarden/examples/pbmc3k/gates/G3/report.html) | Each review gate: evidence, the agent's proposal, the reviewer's verdict, the decision |

GitHub shows `.html` files as source; the links above open them on GitHub Pages.

## What happened in this run
- **QC:** the rules alone would keep 2,292 cells and remove 400.
  - The QC agent judged the per-sample adaptive bounds too aggressive: a mitochondrial cut-off
    of 4.4%, and lower gene and UMI bounds that removed 176 cells.
  - It proposed per-sample limits within the QC policy: 200–2,500 genes, 400–6,000 UMIs, at
    most 6% mitochondrial reads.
  - The reviewer agent approved: 2,408 cells kept, 270 removed, and 22 flagged for review and
    kept, so 2,430 cells went on to clustering. The QC review report shows the whole exchange.
- **Clustering:** 9 clusters (scfast engine, 10 PCs, Leiden resolution 0.5).
- **Annotation:** T cell, monocyte, B cell, NK cell, dendritic cell and platelet populations,
  reviewed by the annotation agent.
- **Context:** the context agent called the tissue peripheral blood (UBERON_0001972), and the
  most likely state a healthy donor (confidence 0.85).
- **Differential expression:** FindMarkers with MAST, the markers of each cell type against the
  other cells, for 6 cell types.
- **Open Targets:** records for all 353 top DE genes (2,899 drug–indication rows), e.g.
  anti-CD20 antibodies (glofitamab, obinutuzumab and others) for MS4A1 in B cells.
- **Interpretation:** the interpretation agent wrote hypotheses citing Open Targets evidence
  IDs, and the reviewer approved them.
- **Time and cost:** 8.6 minutes end to end on a CPU-only HPC login node. The LLM calls were
  priced at about £0.73 by the budget ledger; NVIDIA NIM's hosted models run on free
  development credits.

## Reproduce it

```bash
examples/pbmc3k/get_data.sh
export NVIDIA_API_KEY=...            # or ANTHROPIC_API_KEY / OPENAI_API_KEY; see docs/GETTING_STARTED.md section 3
cellwarden run filtered_gene_bc_matrices/hg19 --config examples/pbmc3k/config.yaml --name pbmc3k --out pbmc3k
```

LLM agents are not deterministic, so a rerun gives the same numbers but differently worded agent
text. Without a key, run with `--config examples/pbmc3k/config_no_llm.yaml --llm none --no-gates`.

Data: 10x Genomics, "3k PBMCs from a Healthy Donor", licensed CC BY 4.0. Local paths and the
host name were removed from the copied reports.
