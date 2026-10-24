#!/usr/bin/env bash
# Smoke test of the cellwarden image (run by .github/workflows/release.yml, and by hand after a pull):
#   docker run --rm --entrypoint bash ghcr.io/cellwarden/cellwarden:latest /opt/cellwarden/containers/smoke_test.sh
#   apptainer exec --nv cellwarden.sif bash /opt/cellwarden/containers/smoke_test.sh
# Preflight checks, then the whole pipeline (S0-S5) on the 500-cell test fixture with a mock LLM
# and offline Open Targets (tests/integration/run_full.sh). CPU only; a GPU is used if present.
set -euo pipefail
work="$(mktemp -d "${TMPDIR:-/tmp}/cellwarden-smoke.XXXXXX")"
trap 'rm -rf "$work"' EXIT

cellwarden doctor
python -c "import scfast; print('scfast', scfast.__version__, scfast.device_info())"
Rscript -e 'cat("scfastR", as.character(packageVersion("scfastR")), "\n")'
CELLWARDEN_DOCTOR_OFFLINE=1 PYTHON=python bash /opt/cellwarden/tests/integration/run_full.sh "$work"
echo "smoke test passed"
