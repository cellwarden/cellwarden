# Published timings for the 10x 1.3M mouse brain dataset

External reference points for our GPU and CPU numbers on the same dataset
(10x "1.3 Million Brain Cells from E18 Mice", `1M_neurons_filtered_gene_bc_matrices_h5.h5`).
None of these sources runs exactly our workflow, so compare orders of magnitude and step
shares, not seconds. The differences are listed under each source.

Our workflow (`bench/run_py.py`, `bench/run_seurat.R`): all 1,306,127 cells before QC; QC
200 < genes < 6000, mitochondrial (`mt-`) < 20 %, genes in ≥ 3 cells; 2,000 HVGs (`seurat_v3`);
30 PCs; k = 20 neighbours on 10 PCs; Leiden at resolution 0.5, 2 iterations; UMAP. No
regress-out, no t-SNE, no diffusion map.

## 1. Dicks et al. 2026, rapids-singlecell paper (primary source)

S. Dicks et al., "GPU-accelerated single-cell analysis at scale with rapids-singlecell",
arXiv:2603.02402 (March 2026). <https://arxiv.org/abs/2603.02402>

- Workflow: normalisation, HVG selection, PCA, neighbours, UMAP, t-SNE, Leiden on "1 million
  cells from the 10x Genomics mouse brain dataset". Timing excludes I/O. Mean of 3 runs.
- CPU (Scanpy) on an AMD Threadripper Pro 7975WX (32 cores, 64 threads, 512 GB RAM):
  **over 52 minutes**.
- GPU (rapids-singlecell, pooled RMM allocator): **26 s on one NVIDIA DGX B200**
  (">120×"); **~25 s on an RTX PRO 6000 Blackwell WS** (120×); H200 at 110×.
- Per-step speed-ups quoted in the text: ~70× preprocessing, ~350× UMAP, ~100× Leiden.

## 2. Per-step timings from the paper's reproducibility notebooks

<https://github.com/theislab/rapids_singlecell-reproducibility>, commit `48d2b1f0fa`
(2026-03-13): `benchmarks/speed/notebooks/demo_cpu-seuratv3-brain-1M.ipynb` and
`demo_gpu-seuratv3-brain-1M.ipynb`. Single runs as saved in the notebooks; the hardware is
not printed in the notebooks (the paper's Methods give the machines above). GPU env:
RAPIDS 25.10a, Python 3.13 (from a printed path).

Data: first 1,000,000 cells of the 1.3M file, QC 500 < genes < 5000, `mt-` < 20 %, genes in
≥ 3 cells → 982,490 cells × 22,539 genes; 5,000 HVGs (`seurat_v3`), regress-out is a no-op
(commented), 100 PCs, 15 neighbours on 50 PCs, UMAP `min_dist=0.3`, Leiden resolution 1.0.

| Step | CPU (Scanpy) wall | GPU (rapids-singlecell) wall |
|---|---|---|
| read (zarr) | 2.7 s | 2.5 s + 7.6 s to GPU |
| QC metrics + filter + filter_genes | 36.1 s | 0.4 s |
| normalize_total + log1p | 2.2 s | 0.06 s |
| HVG (seurat_v3, 5,000) | 15.0 s | 0.22 s |
| subset to HVGs | 14.3 s | 0.71 s |
| scale | 0.44 s | 0.016 s |
| PCA (100 comps) | 2 min 27 s | 0.97 s |
| neighbors (k = 15, 50 PCs) | 2 min 20 s | 3.1 s (`nn_descent`) |
| UMAP | 7 min 27 s | 0.98 s |
| Leiden | 32.2 s (igraph) / 1 h 12 min (leidenalg) | 1.25 s (cuGraph) |
| Louvain | 3 min 45 s (igraph) | 1.55 s |
| t-SNE | 25 min 41 s | 17.3 s |
| diffmap | 50.9 s | 2.4 s |
| **Total printed ("Total Processing time")** | **2,602 s (43.4 min)** | **28.1 s** |
| Preprocessing only (QC → PCA) | 217 s | 3.1 s |

The comparable core (normalize → HVG → scale → PCA → neighbours → UMAP → Leiden igraph, no
t-SNE, Louvain or diffmap) sums to about 13.3 min (798 s) on CPU and about 7.3 s on GPU.

## 3. rapids-singlecell documentation notebook

<https://github.com/scverse/rapids_singlecell-notebooks/blob/main/brain_1M.ipynb>
(rendered in the docs as "1 Million Brain Cells"; last touched 2026-07-10, outputs from a
`rapids-23.12` kernel, GPU not stated). First 1,000,000 cells, same QC as section 2,
5,000 HVGs, regress-out of total counts and % MT, 100 PCs, 15 neighbours on 50 PCs.

- Load + to GPU: 72.8 s (h5ad). Preprocessing (QC → PCA, incl. regress-out 17.4 s): 61.4 s.
- neighbors 18.8 s (brute force), UMAP 2.5 s, Leiden 6.7 s, t-SNE 31.8 s.
- "Total Processing time": 217.9 s (includes plots, logreg marker genes and diffmap).

This is an older RAPIDS release; section 2 supersedes it for the current speed.

## 4. NVIDIA blog, 2021 (historical, RAPIDS scripts before rapids-singlecell)

"Analyzing the RNA-Sequence of 1.3M Mouse Brain Cells with RAPIDS on NVIDIA GPUs",
2021-09-08.
<https://developer.nvidia.com/blog/analyzing-the-rna-sequence-of-1-3m-mouse-brain-cells-with-rapids-on-nvidia-gpus>

- All 1.3M cells, full workflow: **11 minutes on one A100 40 GB**; "almost twice" on one
  V100; preprocessing alone "over 16 minutes" on one V100 and "less than 2 minutes" on 8 V100s.
- CPU: the linked earlier post reports "over three hours" for the 1M-cell version on a CPU
  instance. No per-step table in the page text; QC thresholds not stated.

## 5. Billato et al. 2025, independent benchmark

I. Billato et al., "Benchmarking large-scale single-cell RNA-seq analysis", bioRxiv
doi:10.1101/2025.10.28.681564 (2025-10-29), not peer reviewed.
<https://pmc.ncbi.nlm.nih.gov/articles/PMC12636554/>

- Dataset: the same 10x E18 brain, 1,308,421 cells (TENxBrainData Bioconductor package).
- Full pipelines on 16 Intel Xeon Gold 6130 cores + 2 × Tesla P100 16 GB:
  **rapids_singlecell ≈ 20 min, OSCA ≈ 2.5 h, Seurat ≈ 3 h**. Scanpy per-step times are in
  their Supplementary Table S17 (not read here).
- PCA alone: rapids_singlecell exact SVD of 1.3M cells in 7.5 s (A100 40 GB); Scanpy
  randomized SVD ≈ 1.6 min (dense) / 2 min (sparse) on 2 × AMD EPYC 7301.

## How to use these numbers in the talk

- Seurat at 1.3M: ~3 h on 16 cores is the only published full-pipeline figure (Billato).
- Scanpy at ~1M: ~13 min core workflow, ~43–52 min with t-SNE etc., on 32 cores (Dicks).
- rapids-singlecell at ~1M: ~7 s core workflow, ~25–28 s with t-SNE, on B200 / RTX PRO 6000
  class GPUs (Dicks). Our GPU runs on L4 or H100 should be compared with care: a different
  GPU, HVG count, PC count and k.
