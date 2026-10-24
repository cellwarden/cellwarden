# R packages for every cellwarden stage (R/stages/*.R), installed by containers/Dockerfile.
# Also usable on a native install: Rscript containers/install_r_packages.R
# (binaries come from the Posit Package Manager when the repos option points there).
# Only missing packages are installed. CELLWARDEN_R_LIB=<dir> installs them into that library instead
# of the default one (containers/update_sandbox.sh uses it to add packages to an unpacked image).

cran <- c(
  # S0-S2 and the exchange format
  "Seurat", "SeuratObject", "Matrix", "hdf5r", "jsonlite", "digest", "yaml", "optparse",
  "data.table", "R.utils", "future", "irlba", "igraph", "RhpcBLASctl", "qs2",
  # S2 core: integration, clustering, embedding
  "harmony", "leidenbase", "uwot",
  # tooling
  "remotes", "BiocManager", "testthat"
)
# MAST: the default test of S4's FindMarkers (dge.test)
bioc <- c("DESeq2", "limma", "edgeR", "SummarizedExperiment", "scDblFinder", "BiocParallel", "glmGamPoi", "MAST")
github <- c(BPCells = "bnprks/BPCells/r", presto = "immunogenomics/presto")

lib <- Sys.getenv("CELLWARDEN_R_LIB", "")
if (nzchar(lib)) {
  dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(lib, .libPaths()))
}
target <- .libPaths()[1]
have <- function() rownames(installed.packages())

todo <- setdiff(cran, have())
if (length(todo)) install.packages(todo, lib = target)
todo <- setdiff(bioc, have())
if (length(todo)) BiocManager::install(todo, update = FALSE, ask = FALSE, lib = target)
todo <- github[setdiff(names(github), have())]
if (length(todo)) remotes::install_github(unname(todo), upgrade = "never", lib = target)

need <- c("Seurat", "SeuratObject", "hdf5r", "harmony", "leidenbase", "BPCells", "presto",
          "DESeq2", "limma", "qs2", "MAST")
missing <- need[!vapply(need, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("R packages failed to install: ", paste(missing, collapse = ", "))
