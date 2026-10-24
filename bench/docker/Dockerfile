# One image for the whole project: build, test and benchmark scfast (CPU).
# Build from the repository root:   docker build -f bench/docker/Dockerfile -t scfast .
# Use through bench/docker/run.sh. Not yet built in CI: the first build may need small fixes.
FROM rocker/r-ver:4.4.2
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv python3-dev build-essential pkg-config git curl ca-certificates \
      libhdf5-dev libglpk-dev libxml2-dev libcurl4-openssl-dev libssl-dev libgsl-dev libfontconfig1-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain (scfast-core, scfast-py, and scfastR compiles its Rust code on install)
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:/opt/venv/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.85.0

# R: Posit binaries via rocker/r-ver. Seurat >= 5.1 runs Leiden through leidenbase (no dense matrix).
RUN Rscript -e 'install.packages(c("Seurat","SeuratObject","hdf5r","leidenbase","jsonlite","testthat","digest","Matrix","sessioninfo"))' \
 && Rscript -e 'stopifnot(packageVersion("Seurat") >= "5.1.0"); cat("Seurat", as.character(packageVersion("Seurat")), "\n")'

RUN python3 -m venv /opt/venv \
 && pip install --no-cache-dir numpy scipy pandas h5py anndata scanpy scikit-misc scikit-learn matplotlib \
      leidenalg igraph harmonypy pyyaml pytest "maturin>=1.9.4,<2"

WORKDIR /work
COPY . /work
# Pre-build so `test` and `bench` start fast; both rebuild if the sources change.
RUN maturin build --release -m rust/scfast-py/Cargo.toml -o /tmp/dist \
 && pip install --no-cache-dir /tmp/dist/scfast-*.whl \
 && R CMD INSTALL --no-multiarch rust/scfast-r
ENV PYTHON=/opt/venv/bin/python RSCRIPT=Rscript PYTHONPATH=/work/python
ENTRYPOINT ["bash", "/work/docker/entrypoint.sh"]
CMD ["test"]
