#!/usr/bin/env bash
# Development benchmark: S0-S2 at 2k and 5k cells, CPU path and GPU path.
# Larger sizes are deliberately refused by make_bench_inputs.py; benchmark
# them on the HPC before the first real run.
#
#   tests/benchmark/run_bench.sh            # 2k and 5k
#   SIZES="2000" tests/benchmark/run_bench.sh
#
# On a machine without a GPU the "gpu" row uses gpu_core's CPU fallback
# (CELLWARDEN_ALLOW_CPU_AS_GPU=1), so it measures the exchange-format overhead,
# not GPU speed. Results go to tests/benchmark/results/bench_<date>.tsv.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=python
PY=${PYTHON:-python3}
SIZES=${SIZES:-"2000 5000"}
OUT=tests/benchmark/results; mkdir -p "$OUT"
TSV="$OUT/bench_$(date -u +%Y%m%dT%H%M%SZ).tsv"
echo -e "cells\tdevice\tstage\tseconds\tpeak_rss_mb\tpath" > "$TSV"

$PY tests/benchmark/make_bench_inputs.py --sizes $SIZES
for n in $SIZES; do
  k=$((n / 1000)); d=tests/benchmark/inputs/bench_${k}k
  for dev in cpu gpu; do
    rm -rf "$d/run_$dev"
    if [[ $dev == gpu ]] && ! command -v nvidia-smi >/dev/null 2>&1; then export CELLWARDEN_ALLOW_CPU_AS_GPU=1; fi
    nice -n 10 $PY -m cellwarden run --config "$d/config_$dev.yaml" --until S2 > "$d/run_$dev.log" 2>&1 \
      || { echo "run failed: $d/run_$dev.log"; tail -20 "$d/run_$dev.log"; exit 1; }
    $PY - "$d/run_$dev" "$n" "$dev" "$TSV" <<'EOF'
import json, sys
run, n, dev, tsv = sys.argv[1:]
st = json.load(open(f"{run}/state.json"))
path = ""
try: path = json.load(open(f"{run}/S2/summary.json")).get("path", "")
except Exception: pass
with open(tsv, "a") as fh:
    for stage, s in st["stages"].items():
        secs = s.get("wall_s", "")
        rss = round(s.get("peak_rss_bytes", 0) / 1e6, 1) if s.get("peak_rss_bytes") else ""
        fh.write(f"{n}\t{dev}\t{stage}\t{secs}\t{rss}\t{path if stage=='S2' else ''}\n")
EOF
  done
done
echo; column -t -s $'\t' "$TSV"
