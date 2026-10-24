# Benchmark report: `h100_cuvs`

Node `modal` · unknown (GenuineIntel family 6 model 207) · 32 logical CPUs · GPU: NVIDIA H100 80GB HBM3 · commit `unknown`

## End to end

Compute = normalise → UMAP (median of measured repeats, summed). Read and CPU⇄GPU transfers are shown separately. Speed-up and memory ratio are against Seurat at the same size; higher is better.

| Cells | Engine | Compute s | Read s | Transfer s | Peak RSS GB | Peak GPU GB | Speed-up | Memory ratio | PC sim | kNN overlap | ARI | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1,290,549 | scfast-gpu | 5.35 | 3.73 | 3.08 | 47.67 | 24.23 | –× | –× | – | – | – | no parity data |

Parity thresholds: PC similarity ≥ 0.99, kNN overlap ≥ 0.9, ARI ≥ 0.9. Adoption: parity passes and, at ≥ 100,000 cells, ≥ 1.5× faster or ≥ 2.0× less peak memory. Scanpy and rapids run their own pipelines (different HVG, graph and Leiden implementations), so their parity is informative only; the rule is meant for scfast.

## Per step

Median wall time in seconds (min–max over measured repeats); peak RSS within the step when the platform supports per-step peaks.

| Cells | Engine | Step | Median s | Min–max s | Repeats | Peak RSS GB | RSS scope |
|---:|---|---|---:|---:|---:|---:|---|
| 1,290,549 | scfast-gpu | read | 3.726 | 3.722–3.730 | 2 | 47.67 | process |
| 1,290,549 | scfast-gpu | to_gpu | 2.574 | 2.574–2.575 | 2 | 47.67 | process |
| 1,290,549 | scfast-gpu | normalize | 0.151 | 0.150–0.151 | 2 | 47.67 | process |
| 1,290,549 | scfast-gpu | hvg | 0.000 | 0.000–0.000 | 2 | 47.67 | process |
| 1,290,549 | scfast-gpu | scale | 0.458 | 0.457–0.460 | 2 | 47.67 | process |
| 1,290,549 | scfast-gpu | pca | 0.947 | 0.910–0.984 | 2 | 47.67 | process |
| 1,290,549 | scfast-gpu | neighbors | 1.176 | 1.173–1.179 | 2 | 47.67 | process |
| 1,290,549 | scfast-gpu | cluster | 2.207 | 2.206–2.208 | 2 | 47.67 | process |
| 1,290,549 | scfast-gpu | umap | 0.409 | 0.394–0.424 | 2 | 47.67 | process |
| 1,290,549 | scfast-gpu | to_cpu | 0.509 | 0.492–0.526 | 2 | 47.67 | process |

## Start-up (cold)

| Engine | Cells | Start-up s |
|---|---:|---:|
| scfast-gpu | 1,290,549 | 6.24 |
