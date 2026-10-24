# Benchmark report: `h100_final_pbmc68k`

Node `modal` · unknown · 32 logical CPUs · GPU: NVIDIA H100 80GB HBM3 · commit `unknown`

## End to end

Compute = normalise → UMAP (median of measured repeats, summed). Read and CPU⇄GPU transfers are shown separately. Speed-up and memory ratio are against Seurat at the same size; higher is better.

| Cells | Engine | Compute s | Read s | Transfer s | Peak RSS GB | Peak GPU GB | Speed-up | Memory ratio | PC sim | kNN overlap | ARI | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 68,260 | seurat | 121.37 | 4.64 | 0.00 | – | – | 1.00× | –× | – | – | – | baseline |
| 68,260 | scfast-gpu | 0.57 | 0.08 | 0.04 | 9.70 | 2.26 | 214.40× | –× | 1.000 | 0.994 | 0.719 | fails parity |

Parity thresholds: PC similarity ≥ 0.99, kNN overlap ≥ 0.9, ARI ≥ 0.9. Adoption: parity passes and, at ≥ 100,000 cells, ≥ 1.5× faster or ≥ 2.0× less peak memory. Scanpy and rapids run their own pipelines (different HVG, graph and Leiden implementations), so their parity is informative only; the rule is meant for scfast.

## Agreement with known cell types

| Cells | Engine | ARI vs truth | Reference ARI vs truth |
|---:|---|---:|---:|
| 68,260 | scfast-gpu | 0.251 | 0.283 |

## Per step

Median wall time in seconds (min–max over measured repeats); peak RSS within the step when the platform supports per-step peaks.

| Cells | Engine | Step | Median s | Min–max s | Repeats | Peak RSS GB | RSS scope |
|---:|---|---|---:|---:|---:|---:|---|
| 68,260 | seurat | read | 4.638 | 4.491–5.312 | 3 | – | process |
| 68,260 | seurat | normalize | 0.500 | 0.478–0.522 | 3 | – | process |
| 68,260 | seurat | hvg | 1.811 | 1.733–2.066 | 3 | – | process |
| 68,260 | seurat | scale | 1.424 | 1.415–1.985 | 3 | – | process |
| 68,260 | seurat | pca | 21.006 | 20.976–22.002 | 3 | – | process |
| 68,260 | seurat | neighbors | 20.125 | 19.876–21.607 | 3 | – | process |
| 68,260 | seurat | cluster | 7.043 | 6.728–8.100 | 3 | – | process |
| 68,260 | seurat | umap | 69.457 | 64.727–69.982 | 3 | – | process |
| 68,260 | scfast-gpu | read | 0.076 | 0.076–0.077 | 2 | 9.70 | process |
| 68,260 | scfast-gpu | to_gpu | 0.034 | 0.033–0.035 | 2 | 9.70 | process |
| 68,260 | scfast-gpu | normalize | 0.009 | 0.008–0.010 | 2 | 9.70 | process |
| 68,260 | scfast-gpu | hvg | 0.000 | 0.000–0.000 | 2 | 9.70 | process |
| 68,260 | scfast-gpu | scale | 0.020 | 0.019–0.020 | 2 | 9.70 | process |
| 68,260 | scfast-gpu | pca | 0.071 | 0.067–0.075 | 2 | 9.70 | process |
| 68,260 | scfast-gpu | neighbors | 0.095 | 0.095–0.096 | 2 | 9.70 | process |
| 68,260 | scfast-gpu | cluster | 0.212 | 0.212–0.212 | 2 | 9.70 | process |
| 68,260 | scfast-gpu | umap | 0.159 | 0.158–0.159 | 2 | 9.70 | process |
| 68,260 | scfast-gpu | to_cpu | 0.010 | 0.008–0.013 | 2 | 9.70 | process |

## Start-up (cold)

| Engine | Cells | Start-up s |
|---|---:|---:|
| scfast-gpu | 68,260 | 9.01 |
| seurat | 68,260 | 5.65 |
