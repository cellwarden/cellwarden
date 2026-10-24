# Benchmark report: `pbmc68k_cpu16`

Node `modal` · unknown · 32 logical CPUs · GPU: none · commit `unknown`

Seurat 5.6.0 Leiden backend: leidenbase.

## End to end

Compute = normalise → UMAP (median of measured repeats, summed). Read and CPU⇄GPU transfers are shown separately. Speed-up and memory ratio are against Seurat at the same size; higher is better.

| Cells | Engine | Compute s | Read s | Transfer s | Peak RSS GB | Peak GPU GB | Speed-up | Memory ratio | PC sim | kNN overlap | ARI | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 68,260 | seurat | 121.37 | 4.64 | 0.00 | – | – | 1.00× | –× | – | – | – | baseline |
| 68,260 | scanpy | 57.18 | 2.00 | 0.00 | 6.41 | – | 2.12× | –× | 1.000 | 0.974 | 0.559 | fails parity |
| 68,260 | scfast-cpu | 1.80 | 0.11 | 0.00 | 1.09 | – | 67.39× | –× | 1.000 | 0.945 | 0.705 | fails parity |

Parity thresholds: PC similarity ≥ 0.99, kNN overlap ≥ 0.9, ARI ≥ 0.9. Adoption: parity passes and, at ≥ 100,000 cells, ≥ 1.5× faster or ≥ 2.0× less peak memory. Scanpy and rapids run their own pipelines (different HVG, graph and Leiden implementations), so their parity is informative only; the rule is meant for scfast.

## Agreement with known cell types

| Cells | Engine | ARI vs truth | Reference ARI vs truth |
|---:|---|---:|---:|
| 68,260 | scanpy | 0.258 | 0.283 |
| 68,260 | scfast-cpu | 0.267 | 0.283 |

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
| 68,260 | scanpy | read | 2.002 | 1.989–2.185 | 3 | 6.41 | process |
| 68,260 | scanpy | normalize | 0.513 | 0.500–0.538 | 3 | 6.41 | process |
| 68,260 | scanpy | hvg | 0.817 | 0.772–0.842 | 3 | 6.41 | process |
| 68,260 | scanpy | scale | 0.966 | 0.874–0.973 | 3 | 6.41 | process |
| 68,260 | scanpy | pca | 12.088 | 9.996–12.309 | 3 | 6.41 | process |
| 68,260 | scanpy | neighbors | 5.392 | 5.140–5.428 | 3 | 6.41 | process |
| 68,260 | scanpy | cluster | 2.949 | 2.820–3.621 | 3 | 6.41 | process |
| 68,260 | scanpy | umap | 34.455 | 34.336–34.984 | 3 | 6.41 | process |
| 68,260 | scfast-cpu | read | 0.112 | 0.085–0.134 | 3 | 1.03 | process |
| 68,260 | scfast-cpu | normalize | 0.079 | 0.073–0.086 | 3 | 1.09 | process |
| 68,260 | scfast-cpu | hvg | 0.000 | 0.000–0.000 | 3 | 1.09 | process |
| 68,260 | scfast-cpu | scale | 0.002 | 0.002–0.002 | 3 | 1.09 | process |
| 68,260 | scfast-cpu | pca | 0.301 | 0.297–0.311 | 3 | 1.09 | process |
| 68,260 | scfast-cpu | neighbors | 0.167 | 0.167–0.179 | 3 | 1.09 | process |
| 68,260 | scfast-cpu | cluster | 0.718 | 0.679–0.759 | 3 | 1.09 | process |
| 68,260 | scfast-cpu | umap | 0.534 | 0.522–0.542 | 3 | 1.09 | process |

## Start-up (cold)

| Engine | Cells | Start-up s |
|---|---:|---:|
| scfast-cpu | 68,260 | 0.00 |
| seurat | 68,260 | 5.65 |
| scanpy | 68,260 | 5.38 |
