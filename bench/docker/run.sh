#!/usr/bin/env bash
# Build the image and run a mode in it. Results from `bench` appear in bench/results/ on the host.
#   bash bench/docker/run.sh test                       # whole test suite
#   STAGES="rust pytest" bash bench/docker/run.sh test  # only some stages
#   CELLS=100000 THREADS=8 MEM=48g bash bench/docker/run.sh bench
#   INPUT=~/Downloads/<file>.h5 bash bench/docker/run.sh pipeline   # every step's outputs for your data
#   bash bench/docker/run.sh all                        # tests, then the benchmark
#   bash bench/docker/run.sh shell
# MEM and CPUS cap the container (Seurat needs tens of GB at 100k cells).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-test}"
docker build -f bench/docker/Dockerfile -t scfast .
mkdir -p bench/results
tty=(); [[ -t 0 ]] && tty=(-it)
mount=()
if [[ -n "${INPUT:-}" ]]; then
  [[ -e "$INPUT" ]] || { echo "INPUT not found: $INPUT" >&2; exit 2; }
  INPUT="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"
  mount=(-v "$(dirname "$INPUT"):/input:ro"); export INPUT_FILE="/input/$(basename "$INPUT")"
fi
docker run --rm "${tty[@]}" "${mount[@]}" ${MEM:+--memory "$MEM"} ${CPUS:+--cpus "$CPUS"} \
  -e CELLS -e GENES -e TYPES -e THREADS -e REPS -e ENGINES -e NAME -e TIMEOUT -e SKIP_UMAP -e STAGES -e SKIP -e INPUT_FILE -e ENGINE -e MAX_GENES -e MAX_MT -e UMAP_EPOCHS -e UMAP_INIT \
  -v "$PWD/bench/results:/work/bench/results" scfast "$mode"
