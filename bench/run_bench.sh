#!/usr/bin/env bash
# Run the full benchmark: shared QC + subsampling, every engine at every size,
# parity against Seurat, and the report (docs/RUST_ENGINE.md section 8).
#
#   bash bench/run_bench.sh --input pbmc3k/filtered_gene_bc_matrices/hg19 \
#       --name pbmc3k --sizes all --engines seurat,scanpy,rapids --reps 3 --threads 8
#
# Results go to bench/results/<name>/ (report.md, results.jsonl, parity.jsonl, plots).
# An engine that fails or times out at one size is not run at larger sizes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="" NAME="" SIZES="all" ENGINES="seurat,scanpy,scanpy-gpu,scfast-cpu,scfast-gpu"
REPS=3 THREADS="" LABELS="" TIMEOUT=7200 PREP_ARGS=() RUN_ARGS=()
PYTHON="${PYTHON:-python3}" RAPIDS_PYTHON="${RAPIDS_PYTHON:-$PYTHON}" RSCRIPT="${RSCRIPT:-Rscript}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --sizes) SIZES="$2"; shift 2 ;;
    --engines) ENGINES="$2"; shift 2 ;;
    --reps) REPS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --labels) LABELS="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;            # seconds per engine per size
    --qc) read -r -a PREP_ARGS <<< "$2"; shift 2 ;; # e.g. --qc "--max-mt 100 --max-genes 1e9"
    --skip-umap) RUN_ARGS+=(--skip-umap); shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$INPUT" && -n "$NAME" ]] || { echo "--input and --name are required" >&2; exit 2; }

RUN="$HERE/results/$NAME"
mkdir -p "$RUN"
if [[ -n "$THREADS" ]]; then
  for v in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMBA_NUM_THREADS VECLIB_MAXIMUM_THREADS RAYON_NUM_THREADS; do
    export "$v=$THREADS"
  done
  RUN_ARGS+=(--threads "$THREADS")
fi

if [[ "${SKIP_PREP:-0}" == "1" ]] && [[ -d "$RUN/data" ]] && compgen -G "$RUN/data/n*" >/dev/null; then
  echo "== prep skipped (SKIP_PREP=1; using existing $RUN/data)"
else
  if [[ "${SKIP_PREP:-0}" == "1" ]]; then
    echo "== prep -> $RUN/data (SKIP_PREP ignored; missing or empty $RUN/data)"
  else
    echo "== prep -> $RUN/data"
  fi
  "$PYTHON" "$HERE/prep.py" --input "$INPUT" --out "$RUN/data" --sizes "$SIZES" \
    ${LABELS:+--labels "$LABELS"} "${PREP_ARGS[@]}"
fi

[[ -d "$RUN/data" ]] || { echo "prep failed: no $RUN/data" >&2; exit 1; }
mapfile -t DATASETS < <(cd "$RUN/data" && ls -d n* 2>/dev/null | sort -t n -k2 -n)
[[ ${#DATASETS[@]} -gt 0 ]] || { echo "prep produced no datasets under $RUN/data" >&2; exit 1; }
IFS=',' read -r -a ENGINE_LIST <<< "$ENGINES"
declare -A STOPPED=()

for ds in "${DATASETS[@]}"; do
  for eng in "${ENGINE_LIST[@]}"; do
    [[ -n "${STOPPED[$eng]:-}" ]] && { echo "-- skip $eng $ds (failed at a smaller size)"; continue; }
    echo "== $eng $ds"
    if [[ "$eng" == "seurat" ]]; then
      cmd=("$RSCRIPT" "$HERE/run_seurat.R")
    elif [[ "$eng" == "rapids" || "$eng" == "scanpy-gpu" ]]; then
      cmd=("$RAPIDS_PYTHON" "$HERE/run_py.py" --engine "$eng")
    elif [[ "$eng" == "scfast-gpu" ]]; then
      # RAPIDS image (CuPy, cuVS, cuGraph) with the scfast built from this checkout on PYTHONPATH.
      cmd=(env PYTHONPATH="$HERE/../rust/scfast-py/python${PYTHONPATH:+:$PYTHONPATH}" "$RAPIDS_PYTHON" "$HERE/run_py.py" --engine "$eng")
    else
      cmd=("$PYTHON" "$HERE/run_py.py" --engine "$eng")
    fi
    set +e
    timeout "$TIMEOUT" "${cmd[@]}" --data "$RUN/data/$ds" --out "$RUN" --reps "$REPS" "${RUN_ARGS[@]}"
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      echo "!! $eng $ds exited with $rc; not running $eng at larger sizes"
      # Runners record their own step errors (exit 1); a timeout or a kill leaves no record, so write one.
      why=""
      if [[ $rc -eq 124 ]]; then why="timeout after ${TIMEOUT}s"
      elif [[ $rc -eq 137 ]]; then why="killed (SIGKILL, exit 137; likely out of memory)"
      elif [[ $rc -gt 128 ]]; then why="killed by signal $((rc - 128)) (exit $rc)"
      fi
      [[ -n "$why" ]] && printf '{"schema":"1.0","engine":"%s","dataset":"%s","cells":%s,"rep":-1,"warmup":false,"step":"all","kind":"compute","status":"error","error":"%s","wall_s":null}\n' \
        "$eng" "$ds" "${ds#n}" "$why" >> "$RUN/results.jsonl"
      STOPPED[$eng]=1
    fi
  done
done

echo "== parity"
"$PYTHON" "$HERE/parity.py" --run "$RUN"
echo "== report"
"$PYTHON" "$HERE/report.py" --run "$RUN"
echo "Done: $RUN/report.md"
