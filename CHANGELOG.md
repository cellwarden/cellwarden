# Changelog

## 0.3.0 (2026-10-25): first public release as CellWarden
- **Renamed scpipe to CellWarden.** The command is `cellwarden`; `scpipe` stays as an alias for
  this release. The image is `cellwarden/cellwarden`. `SCPIPE_*` environment variables are
  still read.
- **Differential expression:** Seurat FindMarkers with MAST by default (any FindMarkers test),
  or pseudobulk DESeq2 / limma-voom.
  - Comparisons can use any column.
  - Markers of each cell type are found when no comparison is set.
  - The comparisons can be chosen after annotation.
- **Command line:**
  - stage names on the command line;
  - a live progress display with time left;
  - a setup wizard (`cellwarden run -i`).
- **Rust Harmony** batch correction, 40x faster than R harmony on 200k cells.
- **Default NVIDIA model:** Nemotron 3 Ultra, with thinking off. Per-role `max_tokens` now
  takes effect.
- **Open Targets** updated for its October 2026 API.
- **Images:** Docker Hub and GHCR images, scfast wheels, and the getting-started guide.

## 0.2.0 (2026-10-24)
First tagged release, as scpipe.
