# Benchmark report: `laptop_brain100k`

Node `Mukeshs-MacBook-Pro.local` · Apple M4 Pro · 12 logical CPUs · GPU: none · commits `820f4a6`, `e60ffc8`

**Warning:** records in this run come from more than one commit, so engines may not have run the same harness code. Rerun on a single commit for publishable numbers.

Seurat 5.6.0 Leiden backend: leidenbase.

## End to end

Compute = normalise → UMAP (median of measured repeats, summed). Read and CPU⇄GPU transfers are shown separately. Speed-up and memory ratio are against Seurat at the same size; higher is better.

| Cells | Engine | Compute s | Read s | Transfer s | Peak RSS GB | Peak GPU GB | Speed-up | Memory ratio | PC sim | kNN overlap | ARI | Verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100,000 | seurat | 98.79 | 10.49 | 0.00 | – | – | 1.00× | –× | – | – | – | baseline |
| 100,000 | scanpy | 47.96 | 5.87 | 0.00 | 10.60 | – | 2.06× | –× | – | – | – | no parity data |
| 100,000 | scfast-cpu | 1.55 | 0.24 | 0.00 | 2.94 | – | 63.77× | –× | – | – | – | no parity data |

Parity thresholds: PC similarity ≥ 0.99, kNN overlap ≥ 0.9, ARI ≥ 0.9. Adoption: parity passes and, at ≥ 100,000 cells, ≥ 1.5× faster or ≥ 2.0× less peak memory. Scanpy and rapids run their own pipelines (different HVG, graph and Leiden implementations), so their parity is informative only; the rule is meant for scfast.

## Per step

Median wall time in seconds (min–max over measured repeats); peak RSS within the step when the platform supports per-step peaks.

| Cells | Engine | Step | Median s | Min–max s | Repeats | Peak RSS GB | RSS scope |
|---:|---|---|---:|---:|---:|---:|---|
| 100,000 | seurat | read | 10.493 | 10.493–10.493 | 1 | – | process |
| 100,000 | seurat | normalize | 0.731 | 0.731–0.731 | 1 | – | process |
| 100,000 | seurat | hvg | 4.391 | 4.391–4.391 | 1 | – | process |
| 100,000 | seurat | scale | 1.084 | 1.084–1.084 | 1 | – | process |
| 100,000 | seurat | pca | 11.143 | 11.143–11.143 | 1 | – | process |
| 100,000 | seurat | neighbors | 15.747 | 15.747–15.747 | 1 | – | process |
| 100,000 | seurat | cluster | 6.736 | 6.736–6.736 | 1 | – | process |
| 100,000 | seurat | umap | 58.959 | 58.959–58.959 | 1 | – | process |
| 100,000 | scanpy | read | 5.871 | 5.871–5.871 | 1 | 6.93 | process |
| 100,000 | scanpy | normalize | 0.833 | 0.833–0.833 | 1 | 7.35 | process |
| 100,000 | scanpy | hvg | 2.004 | 2.004–2.004 | 1 | 7.35 | process |
| 100,000 | scanpy | scale | 1.417 | 1.417–1.417 | 1 | 7.35 | process |
| 100,000 | scanpy | pca | 3.279 | 3.279–3.279 | 1 | 10.60 | process |
| 100,000 | scanpy | neighbors | 3.748 | 3.748–3.748 | 1 | 10.60 | process |
| 100,000 | scanpy | cluster | 2.312 | 2.312–2.312 | 1 | 10.60 | process |
| 100,000 | scanpy | umap | 34.364 | 34.364–34.364 | 1 | 10.60 | process |
| 100,000 | scfast-cpu | read | 0.244 | 0.234–0.254 | 3 | 2.94 | process |
| 100,000 | scfast-cpu | normalize | 0.160 | 0.153–0.196 | 3 | 2.94 | process |
| 100,000 | scfast-cpu | hvg | 0.000 | 0.000–0.000 | 3 | 2.94 | process |
| 100,000 | scfast-cpu | scale | 0.015 | 0.015–0.016 | 3 | 2.94 | process |
| 100,000 | scfast-cpu | pca | 0.356 | 0.354–0.357 | 3 | 2.94 | process |
| 100,000 | scfast-cpu | neighbors | 0.144 | 0.142–0.145 | 3 | 2.94 | process |
| 100,000 | scfast-cpu | cluster | 0.376 | 0.373–0.390 | 3 | 2.94 | process |
| 100,000 | scfast-cpu | umap | 0.497 | 0.495–0.500 | 3 | 2.94 | process |

## Start-up (cold)

| Engine | Cells | Start-up s |
|---|---:|---:|
| scfast-cpu | 100,000 | 0.00 |
| scanpy | 100,000 | 1.41 |
| seurat | 100,000 | 2.23 |
