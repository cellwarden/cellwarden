"""S4 driver: the R pseudobulk DGE step, then the Open Targets step.

    python -m cellwarden.stages.s4 --run-dir DIR [--skip-dge] [--refresh]

This is what the runner's ``S4`` stage runs, so both halves share one
fingerprint and one ``logs/S4.log``. The R step is
``Rscript R/stages/run_stage.R --stage S4 --run-dir DIR`` (``CELLWARDEN_RSCRIPT``
overrides ``Rscript``, ``CELLWARDEN_REPO_ROOT`` the repository); its output is
streamed into this process's stdout. ``--skip-dge`` runs only the Open
Targets step on existing DGE tables; ``--refresh`` is passed on to it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import StageError, log
from . import s4_opentargets

STAGE = "S4"


def run_r_step(run_dir: Path, repo_root: Path) -> None:
    script = repo_root / "R" / "stages" / "run_stage.R"
    if not script.exists():
        raise StageError(f"{script} not found (set CELLWARDEN_REPO_ROOT)")
    rscript = os.environ.get("CELLWARDEN_RSCRIPT") or "Rscript"
    argv = [rscript, str(script), "--stage", STAGE, "--run-dir", str(run_dir)]
    log(f"{STAGE}/dge: running {' '.join(argv)}")
    sys.stdout.flush()
    try:
        r = subprocess.run(argv, stdout=sys.stdout, stderr=subprocess.STDOUT, cwd=str(repo_root))
    except OSError as e:
        raise StageError(f"cannot start {rscript}: {e}") from e
    if r.returncode != 0:
        raise StageError(f"the R DGE step failed (exit {r.returncode}); see the lines above")
    if not (run_dir / "S4" / "dge_summary.json").exists():
        raise StageError("the R DGE step exited 0 but did not write S4/dge_summary.json")


def main(argv: list[str] | None = None) -> int:
    from ..config import REPO_ROOT, ConfigError, load_resolved

    p = argparse.ArgumentParser(prog="python -m cellwarden.stages.s4")
    p.add_argument("--run-dir", default=os.environ.get("CELLWARDEN_RUN_DIR"), required="CELLWARDEN_RUN_DIR" not in os.environ)
    p.add_argument("--skip-dge", action="store_true", help="only run the Open Targets step")
    p.add_argument("--refresh", action="store_true", help="Open Targets: ignore cached responses")
    a = p.parse_args(argv)
    run_dir = Path(a.run_dir).expanduser().resolve()
    t0 = time.monotonic()
    log(f"{STAGE}: start (run dir {run_dir}, python {sys.version.split()[0]})")
    try:
        cfg = load_resolved(run_dir)
        repo = Path(cfg.get("repo_root") or REPO_ROOT)
        if not a.skip_dge:
            run_r_step(run_dir, repo)
        s4_opentargets.run(run_dir, cfg, refresh=a.refresh)
    except (StageError, ConfigError) as e:
        print(f"{STAGE}: ERROR: {e}", file=sys.stderr, flush=True)
        return 1
    except Exception:
        traceback.print_exc()
        print(f"{STAGE}: ERROR: unexpected failure (traceback above)", file=sys.stderr, flush=True)
        return 1
    log(f"{STAGE}: done in {time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
