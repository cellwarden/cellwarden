"""Stage S5: export from the Seurat object (R), then build report.html and explorer.html.

    python -m cellwarden.stages.s5_report --run-dir DIR [--skip-export] [--report-only]

Inputs: ``S2/object.*`` (required; the R export reads S4's object when a later
stage saved one) and whatever S0-S4, gates/ and llm/ produced. Outputs:
``S5/umap.tsv.gz``, ``S5/expr_subset.tsv.gz``, ``S5/export.json`` (R),
``S5/report.html``, ``S5/explorer.html`` (Python). Sections whose inputs are
missing say "not run" rather than failing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [INFO] {msg}", flush=True)


def run_export(run_dir: Path, repo: Path) -> None:
    rscript = os.environ.get("CELLWARDEN_RSCRIPT") or "Rscript"
    script = repo / "R" / "stages" / "run_stage.R"
    argv = [rscript, str(script), "--stage", "S5", "--run-dir", str(run_dir)]
    log("S5 export: " + " ".join(argv))
    r = subprocess.run(argv, stdout=sys.stdout, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"S5: R export failed (exit {r.returncode})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m cellwarden.stages.s5_report")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--skip-export", action="store_true", help="reuse an existing S5/umap.tsv.gz")
    p.add_argument("--report-only", action="store_true", help="no explorer.html")
    a = p.parse_args(argv)
    run_dir = Path(a.run_dir).resolve()
    repo = Path(os.environ.get("CELLWARDEN_REPO_ROOT") or Path(__file__).resolve().parents[3])
    (run_dir / "S5").mkdir(parents=True, exist_ok=True)
    have_obj = any((run_dir / "S2" / f).exists() for f in ("object.rds", "object.qs2"))
    if not have_obj:
        raise SystemExit("S5: S2/object.rds (or .qs2) not found")
    if a.skip_export and (run_dir / "S5" / "umap.tsv.gz").exists():
        log("S5: reusing existing S5/umap.tsv.gz")
    else:
        run_export(run_dir, repo)
    from cellwarden.report import build

    t0 = time.monotonic()
    ctx = build.collect(run_dir)
    out = build.build_report(run_dir, ctx=ctx)
    log(f"S5: wrote {out} ({out.stat().st_size / 1024:.0f} kB; sections not run: {', '.join(ctx['missing']) or 'none'})")
    if not a.report_only:
        out = build.build_explorer(run_dir, umap=ctx.get("_umap_rows"))
        log(f"S5: wrote {out} ({out.stat().st_size / 1024:.0f} kB)")
    log(f"S5 done in {time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
