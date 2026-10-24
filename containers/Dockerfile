# cellwarden single image (docs/DESIGN.md 7.5): everything every stage needs, prebuilt.
#
#   NVIDIA RAPIDS base (CUDA 12, Python 3.12, CuPy, cuGraph, cuVS, cuML in /opt/conda)
#   + R with Seurat 5, BPCells, presto, DESeq2, limma, leidenbase, hdf5r, harmony
#   + cellwarden, rapids-singlecell, CellTypist (with its default model) and harmonypy
#   + the Rust engine built with CUDA and cuGraph: scfast (Python) and scfastR (R).
# No Rust toolchain or compiler is needed at run time: the engine is built in the "build" stage.
#
#   docker build -t cellwarden -f containers/Dockerfile .
#   docker run --rm --gpus all -v $PWD:/work cellwarden doctor
#   docker pull ghcr.io/cellwarden/cellwarden:latest                                    # published by .github/workflows/release.yml
#   apptainer pull cellwarden.sif docker://ghcr.io/cellwarden/cellwarden:latest             # Docker Hub copy: docker://ad2n15/cellwarden:latest
#   apptainer exec --nv cellwarden.sif cellwarden run --config config.yaml
#
# The image also runs on machines without a GPU: every GPU step falls back to the CPU.
ARG RAPIDS_IMAGE=nvcr.io/nvidia/rapidsai/base:26.10-cuda12-py3.12

# ---------------------------------------------------------------- base: R and system libraries
FROM ${RAPIDS_IMAGE} AS base
USER root
ARG DEBIAN_FRONTEND=noninteractive
# R and its packages use the system toolchain and libraries, never conda's.
ENV SYS_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# System libraries for R, hdf5r, BPCells, igraph, uwot and curl; tmux for `cellwarden run --detach`.
RUN PATH=$SYS_PATH apt-get update && PATH=$SYS_PATH apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg locales tmux git build-essential gfortran cmake pkg-config \
      libhdf5-dev libcurl4-openssl-dev libssl-dev libxml2-dev libglpk-dev libgsl-dev \
      libpng-dev libjpeg-dev libtiff-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
      libfreetype6-dev libcairo2-dev libxt-dev libgit2-dev libzstd-dev liblzma-dev libbz2-dev \
    && sed -i 's/^# *en_GB.UTF-8/en_GB.UTF-8/' /etc/locale.gen && locale-gen \
    && rm -rf /var/lib/apt/lists/*
ENV LANG=en_GB.UTF-8 LC_ALL=en_GB.UTF-8 TZ=Etc/UTC

# Current R from CRAN for Ubuntu 24.04 (its r-recommended packages need the latest R, so no pin).
RUN curl -fsSL https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
      > /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
    && echo "deb https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/" > /etc/apt/sources.list.d/cran.list \
    && PATH=$SYS_PATH apt-get update && PATH=$SYS_PATH apt-get install -y --no-install-recommends r-base r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# R packages: binary builds from the Posit Package Manager keep this fast.
COPY containers/install_r_packages.R /tmp/install_r_packages.R
RUN echo 'options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/noble/latest"), Ncpus = 4)' \
      > /usr/lib/R/etc/Rprofile.site \
    && echo 'options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, R.version$arch, R.version$os)))' \
      >> /usr/lib/R/etc/Rprofile.site \
    && PATH=$SYS_PATH Rscript /tmp/install_r_packages.R && rm /tmp/install_r_packages.R

# ---------------------------------------------------------------- build: the Rust engine
FROM base AS build
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable \
    && /opt/cargo/bin/cargo --version
COPY rust /src/rust
WORKDIR /src/rust
# CUDA and cuGraph are dlopen'ed at run time, so neither is needed to build.
RUN /opt/conda/bin/pip install --no-cache-dir "maturin>=1.9.4,<2" \
    && PATH=/opt/cargo/bin:$PATH /opt/conda/bin/maturin build --release --locked \
         -m scfast-py/Cargo.toml --features cuda,cugraph -o /out
RUN mkdir -p /out && cd /out \
    && PATH=/opt/cargo/bin:$SYS_PATH SCFASTR_CARGO_FEATURES=cuda,cugraph SCFASTR_TARGET_DIR=/tmp/scfastr-target \
         R CMD INSTALL --build /src/rust/scfast-r

# ---------------------------------------------------------------- final image
FROM base
COPY --from=build /out /tmp/scfast
COPY python /opt/cellwarden/python
COPY R /opt/cellwarden/R
COPY conf /opt/cellwarden/conf
COPY config /opt/cellwarden/config
COPY docs /opt/cellwarden/docs
COPY tests /opt/cellwarden/tests
COPY containers/smoke_test.sh /opt/cellwarden/containers/smoke_test.sh

# scfastR (binary package from the build stage).
RUN PATH=$SYS_PATH R CMD INSTALL /tmp/scfast/scfastR_*.tar.gz \
    && PATH=$SYS_PATH Rscript -e 'stopifnot(requireNamespace("scfastR", quietly = TRUE))'

# Python into the RAPIDS conda environment. rapids-singlecell without its [rapids] extra: the
# RAPIDS libraries are already in the image as conda packages; only its pure-Python deps come in.
RUN pip install --no-cache-dir /tmp/scfast/scfast-*.whl "/opt/cellwarden/python[harmony,report,llm,test]" \
         "rapids-singlecell-cu12==0.18.1" \
    && python -c "import cellwarden, scfast, rapids_singlecell, cupy, harmonypy, celltypist; i = scfast.device_info(); assert i['cuda_built'], i" \
    && rm -rf /tmp/scfast

# CellTypist's default model, so annotation works offline (s3_celltypist reads CELLWARDEN_CELLTYPIST_MODELS).
ENV CELLWARDEN_CELLTYPIST_MODELS=/opt/cellwarden/models/celltypist
RUN python -m cellwarden.stages.s3_celltypist --models-dir "$CELLWARDEN_CELLTYPIST_MODELS" --download Immune_All_Low.pkl \
    && chmod -R a+rX /opt/cellwarden

ENV CELLWARDEN_REPO_ROOT=/opt/cellwarden CELLWARDEN_IN_CONTAINER=1 \
    OMP_NUM_THREADS=8 \
    MPLCONFIGDIR=/tmp/mpl NUMBA_CACHE_DIR=/tmp/numba CUPY_CACHE_DIR=/tmp/cupy
# org.opencontainers.image.source links the GHCR package to the (public) repository
LABEL org.opencontainers.image.title="cellwarden" \
      org.opencontainers.image.description="Agentic single-cell RNA-seq pipeline: Seurat 5, Rust engine (scfast), RAPIDS, CellTypist, LLM review gates" \
      org.opencontainers.image.source="https://github.com/cellwarden/cellwarden" \
      org.opencontainers.image.licenses="Apache-2.0"
WORKDIR /work
USER rapids
ENTRYPOINT ["cellwarden"]
CMD ["--help"]
