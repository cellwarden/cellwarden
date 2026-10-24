#!/usr/bin/env bash
# Download the 10x Genomics PBMC 3k example (2,700 cells; CC BY 4.0, 10x Genomics) into
# ./filtered_gene_bc_matrices/hg19, the folder the examples and the Colab notebook use.
set -euo pipefail
url=https://cf.10xgenomics.com/samples/cell/pbmc3k/pbmc3k_filtered_gene_bc_matrices.tar.gz
[ -d filtered_gene_bc_matrices/hg19 ] && { echo "filtered_gene_bc_matrices/hg19 already here"; exit 0; }
curl -fsSL "$url" | tar -xz
echo "PBMC 3k ready: filtered_gene_bc_matrices/hg19"
