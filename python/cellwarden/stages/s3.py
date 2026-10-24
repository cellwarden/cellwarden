"""S3: cluster evidence + CellTypist + consensus annotation, as one runner stage.

    python -m cellwarden.stages.s3 --run-dir DIR [--models-dir DIR]

The runner registers S3 as a single stage (fingerprint over ``annotation``,
``signatures``, ``object_format`` and the S2 object) whose process runs, in
order:

1. ``Rscript R/stages/run_stage.R --stage S3 --run-dir DIR`` (markers,
   cluster summary, signature scores, CellTypist input; ``R/stages/s3_evidence.R``);
2. :mod:`cellwarden.stages.s3_celltypist` (skips with a note when CellTypist is
   not installed or has no model offline);
3. :mod:`cellwarden.stages.s3_consensus` (``evidence_bundle.json``,
   ``annotation_consensus.tsv``).

All three print into the same ``logs/S3.log``. Each half can also be run on
its own (see their docstrings), e.g. to rebuild the consensus after editing
``markers_human.yaml`` without recomputing markers.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import StageError, log, stage_main
from . import s3_celltypist, s3_consensus

STAGE = "S3"


def run_r_step(run_dir: Path, cfg: dict[str, Any]) -> None:
    repo = Path(cfg.get("repo_root") or os.environ.get("CELLWARDEN_REPO_ROOT") or Path(__file__).resolve().parents[3])
    script = repo / "R" / "stages" / "run_stage.R"
    if not script.exists():
        raise StageError(f"R stage script not found: {script}")
    rscript = os.environ.get("CELLWARDEN_RSCRIPT") or "Rscript"
    argv = [rscript, str(script), "--stage", "S3", "--run-dir", str(run_dir)]
    log("running: " + " ".join(argv))
    sys.stdout.flush()
    r = subprocess.run(argv, stdout=sys.stdout, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise StageError(f"R step of S3 failed (exit {r.returncode}); see the lines above")


def run(run_dir: Path, cfg: dict[str, Any], models_dir: str | None = None) -> None:
    run_r_step(run_dir, cfg)
    s3_celltypist.run(run_dir, cfg, models_dir)
    s3_consensus.run(run_dir, cfg)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="python -m cellwarden.stages.s3")
    p.add_argument("--run-dir", default=os.environ.get("CELLWARDEN_RUN_DIR"), required="CELLWARDEN_RUN_DIR" not in os.environ)
    p.add_argument("--models-dir", default=None)
    a = p.parse_args(argv)
    return stage_main(STAGE, lambda rd, cfg: run(rd, cfg, a.models_dir), ["--run-dir", a.run_dir])


if __name__ == "__main__":
    raise SystemExit(main())
