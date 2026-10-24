#!/usr/bin/env bash
# Run the whole project's tests from one place (locally, on the node, or in the container).
#
#   tests/run_all.sh                  # every stage
#   tests/run_all.sh rust pytest      # only the named stages
#   SKIP="bench-smoke" tests/run_all.sh
#
# Stages, in order:
#   fixtures     exchange fixtures match expected.json
#   rust         cargo test --release for scfast-core and scfast-py
#   build-py     build + install the scfast Python wheel (needed by pytest and bench-smoke)
#   pytest       tests/python (exchange, gpu_core CPU path, scfast bindings, bench harness)
#   r-exchange   tests/R (Seurat <-> AnnData exchange)
#   build-r      install the scfastR package from source into a temporary library
#   r-scfast     rust/scfast-r testthat suite
#   roundtrip    cross-language round trip
#   bench-smoke  2,000-cell synthetic benchmark: Seurat, Scanpy, scfast-cpu, one repeat
#
# Every stage runs even if an earlier one fails; stages that need a failed build are skipped.
# Prints a PASS/FAIL/SKIP table and exits non-zero on any FAIL or SKIP.
# Environment: PYTHON, RSCRIPT, PIP, MATURIN (default: on PATH), WORK (scratch directory).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PYTHON="${PYTHON:-python3}"; RSCRIPT="${RSCRIPT:-Rscript}"; MATURIN="${MATURIN:-maturin}"
WORK="${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/scfast-tests.XXXXXX")}"; mkdir -p "$WORK"
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$WORK/.numba}"
RLIB="$WORK/Rlib"; mkdir -p "$RLIB"
export R_LIBS="$RLIB${R_LIBS:+:$R_LIBS}"

ALL=(fixtures rust build-py pytest r-exchange build-r r-scfast roundtrip bench-smoke)
if [[ $# -gt 0 ]]; then STAGES=("$@"); else STAGES=("${ALL[@]}"); fi
declare -A STATUS=() TIME=()
LOGDIR="$WORK/logs"; mkdir -p "$LOGDIR"

skipped() { for s in ${SKIP:-}; do [[ "$s" == "$1" ]] && return 0; done; return 1; }
needs()   { [[ "${STATUS[$1]:-}" == PASS ]] || { echo "needs stage '$1' (${STATUS[$1]:-not run})"; return 1; }; }

stage_fixtures()   { "$PYTHON" tests/fixtures/make_fixture.py --verify; }
stage_rust()       { (cd rust && cargo test --release -p scfast-core -p scfast-py); }
stage_build_py()   {
  "$MATURIN" build --release -m rust/scfast-py/Cargo.toml -o "$WORK/dist" &&
  "$PYTHON" -m pip install --force-reinstall --no-deps "$WORK"/dist/scfast-*.whl; }
stage_pytest()     { needs build-py && "$PYTHON" -m pytest -q tests/python; }
stage_r_exchange() { "$RSCRIPT" -e 'testthat::test_dir("tests/R", reporter = "summary", stop_on_failure = TRUE)'; }
stage_build_r()    { R CMD INSTALL --no-multiarch --library="$RLIB" rust/scfast-r; }
stage_r_scfast()   { needs build-r && "$RSCRIPT" -e \
  'testthat::test_dir("rust/scfast-r/tests/testthat", package = "scfastR", load_package = "installed", reporter = "summary", stop_on_failure = TRUE)'; }
stage_roundtrip()  { PYTHON="$PYTHON" RSCRIPT="$RSCRIPT" bash tests/integration/run_roundtrip.sh "$WORK/roundtrip"; }
stage_bench_smoke() {
  needs build-py || return 1
  "$PYTHON" bench/synth.py --cells 2500 --genes 4000 --types 8 --out "$WORK/smoke.h5" &&
  PYTHON="$PYTHON" RSCRIPT="$RSCRIPT" bash bench/run_bench.sh --input "$WORK/smoke.h5" --labels "$WORK/smoke.h5.labels.tsv" \
    --name smoke --sizes 2000 --engines seurat,scanpy,scfast-cpu --reps 1 --threads 2 --timeout 1800 \
    --qc "--max-mt 100 --max-genes 1e9" &&
  grep -q '"status": *"ok"' bench/results/smoke/results.jsonl &&
  ! grep -q '"status": *"error"' bench/results/smoke/results.jsonl; }

for s in "${STAGES[@]}"; do
  fn="stage_${s//-/_}"
  if ! declare -F "$fn" >/dev/null; then echo "unknown stage: $s" >&2; STATUS[$s]=FAIL; continue; fi
  if skipped "$s"; then STATUS[$s]=SKIP; continue; fi
  echo "=== $s"
  t0=$SECONDS
  "$fn" > >(tee "$LOGDIR/$s.log") 2>&1; rc=$?
  wait
  TIME[$s]=$((SECONDS - t0))
  if [[ $rc -eq 0 ]]; then STATUS[$s]=PASS; else STATUS[$s]=FAIL; echo "--- $s failed (log: $LOGDIR/$s.log)"; fi
done

echo; echo "stage         result   seconds"
bad=0
for s in "${STAGES[@]}"; do
  printf '%-13s %-8s %s\n' "$s" "${STATUS[$s]:-SKIP}" "${TIME[$s]:--}"
  [[ "${STATUS[$s]:-SKIP}" == PASS ]] || bad=1
done
echo "logs: $LOGDIR"
exit $bad
