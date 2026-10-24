#!/usr/bin/env bash
# Scale runs on the HPC (docs/scfast-integration.md, Phase 8): every engine variant at every
# size, each as a real pipeline run `cellwarden run --until S2` (S0 ingest, S1 QC, S2 core).
#
#   bash bench/hpc/run_scale.sh --data bench/hpc/data --sizes 50000,100000,200000,500000 \
#       --engines seurat-cpu,scfast-cpu,scfast-cuda,rapids --threads 8 [--out bench/results/scale]
#
# Engine variants (config keys they set; everything else is the config/examples default):
#   seurat-cpu   engine: seurat  device: cpu                        Seurat on the CPU (baseline)
#   scfast-cpu   engine: scfast  device: cpu                        scfastR in process, CPU
#   scfast-cuda  engine: scfast  device: gpu  engine_gpu: scfast    scfastR in process, CUDA
#   rapids       engine: seurat  device: gpu  engine_gpu: rapids    rapids-singlecell via the exchange
#
# Per run, under <out>/n<N>/<engine>/: config.yaml + samples.csv, run/ (the cellwarden run dir),
# time.log (/usr/bin/time -v), gpu_mem.log (nvidia-smi samples, GPU variants), clusters.tsv.gz
# (S2 clusters, for ARI). One JSON line per run is appended to <out>/runs.jsonl; summarise.py
# turns that file into the table and the adoption verdicts. A variant that fails or times out at
# one size is not run at larger sizes. Inputs come from make_scale_inputs.py. Not run in CI.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DATA="" SIZES="50000,100000,200000,500000" ENGINES="seurat-cpu,scfast-cpu,scfast-cuda,rapids"
THREADS=8 MEM_GB=64 GPU_CAP_GB=20 TIMEOUT=14400 OUT="$REPO/bench/results/scale" SEED=42
PYTHON="${PYTHON:-python3}" RSCRIPT="${RSCRIPT:-Rscript}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA="$2"; shift 2 ;;
    --sizes) SIZES="$2"; shift 2 ;;
    --engines) ENGINES="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --mem-gb) MEM_GB="$2"; shift 2 ;;          # resources.max_memory_gb
    --gpu-cap-gb) GPU_CAP_GB="$2"; shift 2 ;;  # resources.gpu_mem_cap_gb (L4: 24 GB card, keep headroom)
    --timeout) TIMEOUT="$2"; shift 2 ;;        # seconds per run
    --out) OUT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$DATA" ]] || { echo "--data is required (folder written by make_scale_inputs.py)" >&2; exit 2; }
DATA="$(cd "$DATA" && pwd)"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"

export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMBA_NUM_THREADS VECLIB_MAXIMUM_THREADS RAYON_NUM_THREADS; do
  export "$v=$THREADS"
done
NODE="$(hostname -s 2>/dev/null || hostname)"
COMMIT="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
TIME_BIN="$(command -v /usr/bin/time || true)"
HAVE_SMI=0; command -v nvidia-smi >/dev/null 2>&1 && HAVE_SMI=1

echo "scale runs: data=$DATA out=$OUT sizes=$SIZES engines=$ENGINES threads=$THREADS node=$NODE commit=$COMMIT"
echo "environment:"; "$PYTHON" -m cellwarden doctor 2>&1 | sed 's/^/  /' || true
[[ $HAVE_SMI -eq 1 ]] && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  gpu: /'
echo

engine_config() {   # engine_config ENGINE -> the config lines that differ per variant
  case "$1" in
    seurat-cpu)  printf 'engine: seurat\ndevice: cpu\nengine_gpu: rapids\n' ;;
    scfast-cpu)  printf 'engine: scfast\ndevice: cpu\nengine_gpu: rapids\n' ;;
    scfast-cuda) printf 'engine: scfast\ndevice: gpu\nengine_gpu: scfast\n' ;;
    rapids)      printf 'engine: seurat\ndevice: gpu\nengine_gpu: rapids\n' ;;
    *) echo "unknown engine variant: $1 (seurat-cpu, scfast-cpu, scfast-cuda, rapids)" >&2; return 2 ;;
  esac
}

write_config() {   # write_config DIR N ENGINE
  local dir="$1" n="$2" eng="$3"
  printf 'sample_id,path,format,donor,condition,batch\nsyn,%s,10x_h5,D1,control,B1\n' "$DATA/n$n/sample.h5" > "$dir/samples.csv"
  {
    echo "# scale run: $eng at $n cells (bench/hpc/run_scale.sh)"
    echo "project: scale_${eng}_n${n}"
    echo "run_dir: $dir/run"
    echo "seed: $SEED"
    echo "samples: samples.csv"
    echo "object_format: rds"
    echo "resources:"
    echo "  threads: $THREADS"
    echo "  max_memory_gb: $MEM_GB"
    echo "  gpu_mem_cap_gb: $GPU_CAP_GB"
    echo "  nice: 0"
    echo "doublets:"
    echo "  method: scrublet"
    echo "qc:"
    echo "  review_action: keep"
    echo "normalization: lognorm"
    echo "n_hvg: 2000"
    engine_config "$eng"
    echo "integration:"
    echo "  method: none"
    echo "  batch_key: sample_id"
    echo "pcs: 30"
    echo "n_neighbors: 20"
    echo "clustering:"
    echo "  resolution: 0.5"
    echo "gates:"
    echo "  G1: auto"
  } > "$dir/config.yaml"
}

export_clusters() {   # export_clusters RUN_DIR OUT_TSV_GZ
  "$RSCRIPT" --vanilla -e '
    a <- commandArgs(TRUE); suppressPackageStartupMessages(library(Seurat))
    f <- file.path(a[1], "S2", "object.rds"); if (!file.exists(f)) f <- file.path(a[1], "S2", "object.qs2")
    obj <- if (grepl("qs2$", f)) qs2::qs_read(f) else readRDS(f)
    con <- gzfile(a[2], "w"); write.table(data.frame(cell_id = colnames(obj), cluster = as.character(obj$seurat_clusters)),
                                          con, sep = "\t", quote = FALSE, row.names = FALSE); close(con)' "$1" "$2"
}

IFS=',' read -r -a SIZE_LIST <<< "$SIZES"
IFS=',' read -r -a ENGINE_LIST <<< "$ENGINES"
for eng in "${ENGINE_LIST[@]}"; do engine_config "$eng" >/dev/null; done
declare -A STOPPED=()

for n in "${SIZE_LIST[@]}"; do
  [[ -f "$DATA/n$n/sample.h5" ]] || { echo "!! no input for n$n under $DATA (run make_scale_inputs.py)"; continue; }
  for eng in "${ENGINE_LIST[@]}"; do
    if [[ -n "${STOPPED[$eng]:-}" ]]; then echo "-- skip $eng n$n (failed at a smaller size)"; continue; fi
    dir="$OUT/n$n/$eng"
    rm -rf "$dir"; mkdir -p "$dir"
    write_config "$dir" "$n" "$eng"
    echo "== $eng n$n -> $dir"
    smi_pid=""
    if [[ $HAVE_SMI -eq 1 && ( "$eng" == "scfast-cuda" || "$eng" == "rapids" ) ]]; then
      nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -l 1 > "$dir/gpu_mem.log" 2>/dev/null &
      smi_pid=$!
    fi
    started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    t0=$(date +%s.%N)
    set +e
    if [[ -n "$TIME_BIN" ]]; then
      timeout "$TIMEOUT" "$TIME_BIN" -v -o "$dir/time.log" \
        "$PYTHON" -m cellwarden run --config "$dir/config.yaml" --run-dir "$dir/run" --until S2 > "$dir/stdout.log" 2>&1
    else
      timeout "$TIMEOUT" "$PYTHON" -m cellwarden run --config "$dir/config.yaml" --run-dir "$dir/run" --until S2 > "$dir/stdout.log" 2>&1
    fi
    rc=$?
    set -e
    t1=$(date +%s.%N)
    [[ -n "$smi_pid" ]] && { kill "$smi_pid" 2>/dev/null || true; wait "$smi_pid" 2>/dev/null || true; }
    status="ok"
    if [[ $rc -eq 124 ]]; then status="timeout"; elif [[ $rc -ne 0 ]]; then status="failed"; fi
    if [[ "$status" == "ok" ]]; then
      export_clusters "$dir/run" "$dir/clusters.tsv.gz" || { echo "   (cluster export failed)"; }
    else
      STOPPED[$eng]=1
      echo "   $status (exit $rc); last lines of the stage logs:"
      for f in "$dir"/run/logs/*.log; do [[ -e "$f" ]] && { echo "   --- $f"; tail -n 15 "$f" | sed 's/^/   /'; }; done
    fi
    wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')"
    "$PYTHON" - "$OUT/runs.jsonl" <<PY
import json, sys
rec = {"size": int("$n"), "engine": "$eng", "status": "$status", "exit": int("$rc"), "wall_s": float("$wall"),
       "threads": int("$THREADS"), "node": "$NODE", "commit": "$COMMIT", "started_utc": "$started",
       "dir": "$dir", "run_dir": "$dir/run", "clusters": "$dir/clusters.tsv.gz", "truth": "$DATA/n$n/truth.tsv",
       "time_log": "$dir/time.log", "gpu_mem_log": "$dir/gpu_mem.log"}
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(rec) + "\n")
PY
    echo "   $status in ${wall}s"
  done
done

echo
echo "== summarise -> $OUT/summary.md"
"$PYTHON" "$HERE/summarise.py" --runs "$OUT/runs.jsonl" --out "$OUT"
