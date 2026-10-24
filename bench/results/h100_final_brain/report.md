# Benchmark report: `h100_final_brain`

Node `modal` · unknown · 32 logical CPUs · GPU: NVIDIA H100 80GB HBM3 · commit `unknown`

## End to end

Compute = normalise → UMAP (median of measured repeats, summed). Read and CPU⇄GPU transfers are shown separately. Speed-up and memory ratio are against Seurat at the same size; higher is better.

| Cells | Engine | Compute s | Read s | Transfer s | Peak RSS GB | Peak GPU GB | Speed-up | Memory ratio | PC sim | kNN overlap | ARI | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100,000 | seurat | 194.60 | 21.01 | 0.00 | – | – | 1.00× | –× | – | – | – | baseline |
| 100,000 | scfast-gpu | 1.19 | 0.28 | 0.33 | 12.52 | 2.97 | 164.12× | –× | 1.000 | 0.996 | 0.548 | fails parity |
| 500,000 | seurat | 966.82 | 88.44 | 0.00 | – | – | 1.00× | –× | – | – | – | baseline |
| 500,000 | scfast-gpu | 3.22 | 1.29 | 1.19 | 25.66 | 10.20 | 300.06× | –× | 1.000 | 0.994 | 0.476 | fails parity |
| 1,290,549 | scfast-gpu | 7.02 | 13.34 | 4.65 | 102.62 | 43.26 | –× | –× | – | – | – | no parity data |

Parity thresholds: PC similarity ≥ 0.99, kNN overlap ≥ 0.9, ARI ≥ 0.9. Adoption: parity passes and, at ≥ 100,000 cells, ≥ 1.5× faster or ≥ 2.0× less peak memory. Scanpy and rapids run their own pipelines (different HVG, graph and Leiden implementations), so their parity is informative only; the rule is meant for scfast.

## Per step

Median wall time in seconds (min–max over measured repeats); peak RSS within the step when the platform supports per-step peaks.

| Cells | Engine | Step | Median s | Min–max s | Repeats | Peak RSS GB | RSS scope |
|---:|---|---|---:|---:|---:|---:|---|
| 100,000 | seurat | read | 21.008 | 20.756–21.769 | 3 | – | process |
| 100,000 | seurat | normalize | 2.691 | 2.689–2.714 | 3 | – | process |
| 100,000 | seurat | hvg | 7.965 | 7.940–8.178 | 3 | – | process |
| 100,000 | seurat | scale | 2.563 | 2.408–2.720 | 3 | – | process |
| 100,000 | seurat | pca | 30.545 | 30.392–30.558 | 3 | – | process |
| 100,000 | seurat | neighbors | 30.505 | 30.393–30.551 | 3 | – | process |
| 100,000 | seurat | cluster | 13.846 | 12.770–15.195 | 3 | – | process |
| 100,000 | seurat | umap | 106.482 | 105.588–108.299 | 3 | – | process |
| 100,000 | scfast-gpu | read | 0.282 | 0.276–0.288 | 2 | 12.52 | process |
| 100,000 | scfast-gpu | to_gpu | 0.287 | 0.188–0.385 | 2 | 12.52 | process |
| 100,000 | scfast-gpu | normalize | 0.018 | 0.018–0.018 | 2 | 12.52 | process |
| 100,000 | scfast-gpu | hvg | 0.000 | 0.000–0.000 | 2 | 12.52 | process |
| 100,000 | scfast-gpu | scale | 0.027 | 0.027–0.028 | 2 | 12.52 | process |
| 100,000 | scfast-gpu | pca | 0.118 | 0.107–0.129 | 2 | 12.52 | process |
| 100,000 | scfast-gpu | neighbors | 0.125 | 0.125–0.126 | 2 | 12.52 | process |
| 100,000 | scfast-gpu | cluster | 0.351 | 0.350–0.352 | 2 | 12.52 | process |
| 100,000 | scfast-gpu | umap | 0.546 | 0.545–0.546 | 2 | 12.52 | process |
| 100,000 | scfast-gpu | to_cpu | 0.045 | 0.045–0.045 | 2 | 12.52 | process |
| 500,000 | seurat | read | 88.443 | 88.443–88.443 | 1 | – | process |
| 500,000 | seurat | normalize | 10.623 | 10.623–10.623 | 1 | – | process |
| 500,000 | seurat | hvg | 26.818 | 26.818–26.818 | 1 | – | process |
| 500,000 | seurat | scale | 11.259 | 11.259–11.259 | 1 | – | process |
| 500,000 | seurat | pca | 125.205 | 125.205–125.205 | 1 | – | process |
| 500,000 | seurat | neighbors | 152.600 | 152.600–152.600 | 1 | – | process |
| 500,000 | seurat | cluster | 61.315 | 61.315–61.315 | 1 | – | process |
| 500,000 | seurat | umap | 578.999 | 578.999–578.999 | 1 | – | process |
| 500,000 | scfast-gpu | read | 1.292 | 1.291–1.293 | 2 | 25.66 | process |
| 500,000 | scfast-gpu | to_gpu | 0.970 | 0.969–0.971 | 2 | 25.66 | process |
| 500,000 | scfast-gpu | normalize | 0.063 | 0.062–0.064 | 2 | 25.66 | process |
| 500,000 | scfast-gpu | hvg | 0.000 | 0.000–0.000 | 2 | 25.66 | process |
| 500,000 | scfast-gpu | scale | 0.152 | 0.149–0.154 | 2 | 25.66 | process |
| 500,000 | scfast-gpu | pca | 0.403 | 0.387–0.419 | 2 | 25.66 | process |
| 500,000 | scfast-gpu | neighbors | 0.703 | 0.697–0.708 | 2 | 25.66 | process |
| 500,000 | scfast-gpu | cluster | 0.863 | 0.861–0.864 | 2 | 25.66 | process |
| 500,000 | scfast-gpu | umap | 1.039 | 1.038–1.040 | 2 | 25.66 | process |
| 500,000 | scfast-gpu | to_cpu | 0.224 | 0.219–0.228 | 2 | 25.66 | process |
| 1,290,549 | scfast-gpu | read | 13.340 | 12.167–14.513 | 2 | 102.62 | process |
| 1,290,549 | scfast-gpu | to_gpu | 4.049 | 3.690–4.408 | 2 | 102.62 | process |
| 1,290,549 | scfast-gpu | normalize | 0.147 | 0.146–0.148 | 2 | 102.62 | process |
| 1,290,549 | scfast-gpu | hvg | 0.000 | 0.000–0.000 | 2 | 102.62 | process |
| 1,290,549 | scfast-gpu | scale | 0.486 | 0.425–0.547 | 2 | 102.62 | process |
| 1,290,549 | scfast-gpu | pca | 0.983 | 0.921–1.046 | 2 | 102.62 | process |
| 1,290,549 | scfast-gpu | neighbors | 2.815 | 2.807–2.823 | 2 | 102.62 | process |
| 1,290,549 | scfast-gpu | cluster | 2.193 | 2.165–2.220 | 2 | 102.62 | process |
| 1,290,549 | scfast-gpu | umap | 0.395 | 0.394–0.395 | 2 | 102.62 | process |
| 1,290,549 | scfast-gpu | to_cpu | 0.600 | 0.592–0.608 | 2 | 102.62 | process |

## Start-up (cold)

| Engine | Cells | Start-up s |
|---|---:|---:|
| scfast-gpu | 100,000 | 6.25 |
| scfast-gpu | 500,000 | 4.22 |
| scfast-gpu | 1,290,549 | 4.00 |
| seurat | 500,000 | 4.39 |
| seurat | 100,000 | 6.00 |

## Scaling

![scaling_time.png](scaling_time.png)
![scaling_totals.png](scaling_totals.png)

Plotted values are in `summary_steps.csv` and `summary_totals.csv`.
