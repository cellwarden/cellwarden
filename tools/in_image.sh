#!/usr/bin/env bash
# Run a command inside the CellWarden image with this checkout mounted at the same path:
#   IMAGE=ghcr.io/cellwarden/cellwarden:latest tools/in_image.sh python -m pytest -q tests/python
# Used by .github/workflows/exchange-tests.yml; needs docker. RUNNER_TEMP (CI) is mounted too.
set -euo pipefail
IMAGE="${IMAGE:-ghcr.io/cellwarden/cellwarden:latest}"
root="$(cd "$(dirname "$0")/.." && pwd)"
mounts=(-v "$root:$root")
[[ -n "${RUNNER_TEMP:-}" ]] && mounts+=(-v "$RUNNER_TEMP:$RUNNER_TEMP")
exec docker run --rm --user root -e HOME=/tmp -e OMP_NUM_THREADS=2 -e PYTHONPATH="$root/python" \
  "${mounts[@]}" -w "$root" --entrypoint "" "$IMAGE" "$@"
