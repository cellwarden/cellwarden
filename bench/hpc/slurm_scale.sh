#!/usr/bin/env bash
# Slurm job for the scale runs on the Iridis X L4 node (docs/scfast-integration.md, Phase 8).
#
#   sbatch bench/hpc/slurm_scale.sh                          # all sizes, all engine variants
#   sbatch --export=ALL,SIZES=50000,ENGINES=scfast-cpu,scfast-cuda bench/hpc/slurm_scale.sh
#
# Environment overrides (sbatch --export): DATA (inputs, default bench/hpc/data), SIZES,
# ENGINES, THREADS (default: the job's CPUs), OUT (default bench/results/scale), TIMEOUT
# (seconds per run), MODULES (space-separated `module load` names), CONDA_ENV / VENV (one
# of them, activated before the runs). The inputs must already exist (make_scale_inputs.py,
# run with CELLWARDEN_ALLOW_LARGE=1 on the HPC); generating 500k cells takes a few minutes and
# is better done once on a login node or in its own job.
#
# Resource lines are for one L4 (24 GB) with 8 cores and 64 GB host memory; adjust the
# partition/account to the site. Everything runs from the repository root.
#SBATCH --job-name=cellwarden-scale
#SBATCH --partition=l4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=bench/results/scale/slurm-%j.out
#SBATCH --error=bench/results/scale/slurm-%j.err
set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then cd "$SLURM_SUBMIT_DIR"; fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
mkdir -p bench/results/scale

for m in ${MODULES:-}; do module load "$m"; done
if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV"
elif [[ -n "${VENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "cpus=$THREADS mem=${SLURM_MEM_PER_NODE:-?}MB gpus=${CUDA_VISIBLE_DEVICES:-none}"
command -v nvidia-smi >/dev/null && nvidia-smi
uptime

bash "$HERE/run_scale.sh" \
  --data "${DATA:-$REPO/bench/hpc/data}" \
  --sizes "${SIZES:-50000,100000,200000,500000}" \
  --engines "${ENGINES:-seurat-cpu,scfast-cpu,scfast-cuda,rapids}" \
  --threads "$THREADS" \
  --mem-gb "${MEM_GB:-60}" \
  --gpu-cap-gb "${GPU_CAP_GB:-20}" \
  --timeout "${TIMEOUT:-14400}" \
  --out "${OUT:-$REPO/bench/results/scale}"

echo "done: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
