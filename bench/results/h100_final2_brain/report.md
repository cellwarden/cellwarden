# Benchmark report: `h100_final2_brain`

Node `modal` · unknown · 32 logical CPUs · GPU: NVIDIA H100 80GB HBM3 · commit `unknown`

## End to end

Compute = normalise → UMAP (median of measured repeats, summed). Read and CPU⇄GPU transfers are shown separately. Speed-up and memory ratio are against Seurat at the same size; higher is better.

| Cells | Engine | Compute s | Read s | Transfer s | Peak RSS GB | Peak GPU GB | Speed-up | Memory ratio | PC sim | kNN overlap | ARI | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 500,000 | seurat | 966.82 | 88.44 | 0.00 | – | – | 1.00× | –× | – | – | – | baseline |
| 500,000 | scfast-gpu | 3.20 | 1.18 | 1.28 | 25.71 | 10.20 | 302.18× | –× | 1.000 | 0.994 | 0.476 | fails parity |
| 1,290,549 | scfast-gpu | 6.78 | 2.96 | 3.58 | 64.56 | 24.10 | –× | –× | – | – | – | no parity data |

Parity thresholds: PC similarity ≥ 0.99, kNN overlap ≥ 0.9, ARI ≥ 0.9. Adoption: parity passes and, at ≥ 100,000 cells, ≥ 1.5× faster or ≥ 2.0× less peak memory. Scanpy and rapids run their own pipelines (different HVG, graph and Leiden implementations), so their parity is informative only; the rule is meant for scfast.

## Per step

Median wall time in seconds (min–max over measured repeats); peak RSS within the step when the platform supports per-step peaks.

| Cells | Engine | Step | Median s | Min–max s | Repeats | Peak RSS GB | RSS scope |
|---:|---|---|---:|---:|---:|---:|---|
| 500,000 | seurat | read | 88.443 | 88.443–88.443 | 1 | – | process |
| 500,000 | seurat | normalize | 10.623 | 10.623–10.623 | 1 | – | process |
| 500,000 | seurat | hvg | 26.818 | 26.818–26.818 | 1 | – | process |
| 500,000 | seurat | scale | 11.259 | 11.259–11.259 | 1 | – | process |
| 500,000 | seurat | pca | 125.205 | 125.205–125.205 | 1 | – | process |
| 500,000 | seurat | neighbors | 152.600 | 152.600–152.600 | 1 | – | process |
| 500,000 | seurat | cluster | 61.315 | 61.315–61.315 | 1 | – | process |
| 500,000 | seurat | umap | 578.999 | 578.999–578.999 | 1 | – | process |
| 500,000 | scfast-gpu | read | 1.183 | 1.176–1.190 | 2 | 25.71 | process |
| 500,000 | scfast-gpu | to_gpu | 1.099 | 0.736–1.462 | 2 | 25.71 | process |
| 500,000 | scfast-gpu | normalize | 0.097 | 0.062–0.131 | 2 | 25.71 | process |
| 500,000 | scfast-gpu | hvg | 0.000 | 0.000–0.000 | 2 | 25.71 | process |
| 500,000 | scfast-gpu | scale | 0.160 | 0.144–0.176 | 2 | 25.71 | process |
| 500,000 | scfast-gpu | pca | 0.361 | 0.359–0.363 | 2 | 25.71 | process |
| 500,000 | scfast-gpu | neighbors | 0.702 | 0.696–0.708 | 2 | 25.71 | process |
| 500,000 | scfast-gpu | cluster | 0.848 | 0.847–0.848 | 2 | 25.71 | process |
| 500,000 | scfast-gpu | umap | 1.032 | 1.032–1.032 | 2 | 25.71 | process |
| 500,000 | scfast-gpu | to_cpu | 0.185 | 0.176–0.195 | 2 | 25.71 | process |
| 1,290,549 | scfast-gpu | read | 2.961 | 2.960–2.962 | 2 | 64.56 | process |
| 1,290,549 | scfast-gpu | to_gpu | 3.096 | 1.768–4.424 | 2 | 64.56 | process |
| 1,290,549 | scfast-gpu | normalize | 0.154 | 0.149–0.158 | 2 | 64.56 | process |
| 1,290,549 | scfast-gpu | hvg | 0.000 | 0.000–0.000 | 2 | 64.56 | process |
| 1,290,549 | scfast-gpu | scale | 0.440 | 0.422–0.459 | 2 | 64.56 | process |
| 1,290,549 | scfast-gpu | pca | 0.894 | 0.890–0.899 | 2 | 64.56 | process |
| 1,290,549 | scfast-gpu | neighbors | 2.742 | 2.737–2.748 | 2 | 64.56 | process |
| 1,290,549 | scfast-gpu | cluster | 2.160 | 2.158–2.161 | 2 | 64.56 | process |
| 1,290,549 | scfast-gpu | umap | 0.391 | 0.391–0.391 | 2 | 64.56 | process |
| 1,290,549 | scfast-gpu | to_cpu | 0.485 | 0.470–0.500 | 2 | 64.56 | process |

## Start-up (cold)

| Engine | Cells | Start-up s |
|---|---:|---:|
| scfast-gpu | 500,000 | 5.66 |
| scfast-gpu | 1,290,549 | 3.85 |
| seurat | 500,000 | 4.39 |

## Scaling

![scaling_time.png](scaling_time.png)
![scaling_totals.png](scaling_totals.png)

Plotted values are in `summary_steps.csv` and `summary_totals.csv`.
