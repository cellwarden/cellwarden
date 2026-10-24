#!/bin/bash
# Heavy, offline part of an cellwarden run on a compute node: normalisation, PCA, batch correction,
# clustering, UMAP (S2) and markers / CellTypist evidence (S3). Compute nodes have no internet,
# so the review gates (LLM) and Open Targets run on the login node before and after this job.
#
#   sbatch --export=ALL,RUN=/path/to/run_dir conf/jobs/slurm_compute_offline.sbatch
# Edit the partition, cores, memory and time below for your data and site (docs/GETTING_STARTED.md, 8).
# CELLWARDEN_IMAGE: your cellwarden .sif or unpacked sandbox (required); CELLWARDEN_BIND: folders to mount,
# comma-separated (data, run and scratch folders; default $HOME).
#SBATCH --job-name=cellwarden_compute
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${RUN:?set RUN to the run directory: sbatch --export=ALL,RUN=/path/to/run conf/jobs/slurm_compute_offline.sbatch}"
module load apptainer/1.4.0
apptainer exec --cleanenv --env RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \
  -B "${CELLWARDEN_BIND:-$HOME}" "${CELLWARDEN_IMAGE:?set CELLWARDEN_IMAGE to your cellwarden .sif or sandbox folder}" \
  cellwarden resume --run-dir "$RUN" --until evidence --progress plain
