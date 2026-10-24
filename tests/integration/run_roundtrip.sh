#!/usr/bin/env bash
# End-to-end Seurat <-> AnnData exchange check (docs/exchange-spec.md).
#
#   tests/integration/run_roundtrip.sh [WORKDIR]
#
# Runs every step even if an earlier one fails (later steps that depend on a
# failed one are marked SKIP), prints a PASS/FAIL table with timings, and exits
# non-zero if anything failed or was skipped. Without WORKDIR a temp dir is
# used and removed on success (kept on failure, or always with KEEP_WORKDIR=1).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
RSCRIPT="${RSCRIPT:-Rscript}"
PYTHON="${PYTHON:-python3}"
N_PCS=30          # must agree with roundtrip.R / roundtrip.py
N_NEIGHBORS=15
RESOLUTION=0.5
SEED=0

if [[ $# -ge 1 ]]; then
  WORK="$1"; mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/cellwarden-roundtrip.XXXXXX")"
  CLEANUP_ON_PASS=1
fi
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$WORK/.numba}"
LOG="$WORK/logs"; mkdir -p "$LOG"

echo "cellwarden exchange round-trip"
echo "  repo:    $REPO"
echo "  workdir: $WORK"
echo

declare -a NAMES STATUS TIMES
declare -A RESULT
run_step() {   # run_step ID "description" DEPENDS_ON(space-separated or "") -- cmd...
  local id="$1" desc="$2" deps="$3"; shift 4
  local d
  for d in $deps; do
    if [[ "${RESULT[$d]:-}" != "PASS" ]]; then
      NAMES+=("$id  $desc"); STATUS+=("SKIP"); TIMES+=("-"); RESULT[$id]="SKIP"
      echo "=== [$id] $desc: SKIP (needs $d)"; echo; return
    fi
  done
  echo "=== [$id] $desc"
  local t0 t1 rc
  t0=$(date +%s.%N)
  "$@" 2>&1 | tee "$LOG/$id.log"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s.%N)
  local st="PASS"; [[ $rc -eq 0 ]] || st="FAIL"
  NAMES+=("$id  $desc"); STATUS+=("$st"); TIMES+=("$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1fs", b-a}')")
  RESULT[$id]="$st"
  echo "--- [$id] $st (exit $rc)"; echo
}

gpu_core() {
  "$PYTHON" -m cellwarden.gpu_core --in "$WORK/r_slim.h5ad" --out "$WORK/results.h5ad" \
    --n-pcs "$N_PCS" --n-neighbors "$N_NEIGHBORS" --resolution "$RESOLUTION" \
    --batch-key sample --seed "$SEED" --device cpu > "$WORK/gpu_core.stdout"
  local rc=$?
  echo "gpu_core stdout:"; cat "$WORK/gpu_core.stdout"
  [[ $rc -eq 0 ]] || { echo "gpu_core exited $rc"; return $rc; }
  [[ -s "$WORK/results.h5ad" ]] || { echo "results.h5ad not written"; return 1; }
}

run_step S0 "fixture files match expected.json"   ""    -- "$PYTHON" "$REPO/tests/fixtures/make_fixture.py" --verify
run_step S1 "R: fixture -> Seurat -> full/slim"    "S0"  -- "$RSCRIPT" "$HERE/roundtrip.R" write "$WORK"
run_step S2 "Py: read R full/slim files"           "S1"  -- "$PYTHON" "$HERE/roundtrip.py" check-r-files "$WORK"
run_step S3 "Py: gpu_core CLI (cpu) slim->results" "S1"  -- gpu_core
run_step S3b "Py: check results, build refuse files" "S3" -- "$PYTHON" "$HERE/roundtrip.py" check-results "$WORK"
run_step S4 "R: merge results + refuse cases"      "S3b" -- "$RSCRIPT" "$HERE/roundtrip.R" merge "$WORK"
run_step S5a "Py: write full file from fixture"    "S0"  -- "$PYTHON" "$HERE/roundtrip.py" write-python "$WORK"
run_step S5b "R: read Python full file"            "S5a" -- "$RSCRIPT" "$HERE/roundtrip.R" read-python "$WORK"

echo "================ round-trip summary ================"
printf "%-6s %-8s %s\n" "RESULT" "TIME" "STEP"
fail=0
for i in "${!NAMES[@]}"; do
  printf "%-6s %-8s %s\n" "${STATUS[$i]}" "${TIMES[$i]}" "${NAMES[$i]}"
  [[ "${STATUS[$i]}" == "PASS" ]] || fail=1
done
echo "===================================================="
if [[ $fail -ne 0 ]]; then
  echo "ROUND-TRIP FAILED (workdir kept: $WORK, logs: $LOG)"; exit 1
fi
echo "ROUND-TRIP PASSED"
if [[ "${CLEANUP_ON_PASS:-0}" == "1" && "${KEEP_WORKDIR:-0}" != "1" ]]; then rm -rf "$WORK"; fi
