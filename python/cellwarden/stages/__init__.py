"""Python stages of the cellwarden pipeline (docs/stages-spec.md section 3).

Each stage is a module runnable as ``python -m cellwarden.stages.<module> --run-dir DIR``.
It reads ``<run-dir>/config/config.resolved.yaml``, prints progress to
stdout/stderr (the runner sends both to ``logs/<STAGE>.log``) and exits 0 on
success, 1 on a handled failure (message on stderr), 2 on bad usage.

Shared helpers live here.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

__all__ = ["StageError", "stage_main", "utc_now", "write_tsv_gz", "write_json", "read_tsv_str", "log"]


class StageError(RuntimeError):
    """A handled stage failure: the message is printed without a traceback."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def read_tsv_str(path: str | Path) -> pd.DataFrame:
    """Read a TSV keeping every value as the exact string written (no NA parsing)."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)


def write_tsv_gz(df: pd.DataFrame, path: str | Path, na_rep: str = "NA") -> None:
    """Write a deterministic gzip TSV (gzip mtime 0, so equal tables give equal bytes), atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    df.to_csv(buf, sep="\t", index=False, na_rep=na_rep, lineterminator="\n")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(buf.getvalue().encode("utf-8"))
    os.replace(tmp, path)


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def stage_main(stage_id: str, fn: Callable[[Path, dict[str, Any]], None], argv: list[str] | None = None) -> int:
    """Common CLI wrapper: parse ``--run-dir``, load the resolved config, run ``fn``."""
    from ..config import ConfigError, load_resolved

    p = argparse.ArgumentParser(prog=f"python -m cellwarden.stages ({stage_id})")
    p.add_argument("--run-dir", default=os.environ.get("CELLWARDEN_RUN_DIR"),
                   required="CELLWARDEN_RUN_DIR" not in os.environ)
    a = p.parse_args(argv)
    run_dir = Path(a.run_dir).expanduser().resolve()
    t0 = time.monotonic()
    log(f"{stage_id}: start (run dir {run_dir}, python {sys.version.split()[0]})")
    try:
        cfg = load_resolved(run_dir)
        fn(run_dir, cfg)
    except (StageError, ConfigError) as e:
        print(f"{stage_id}: ERROR: {e}", file=sys.stderr, flush=True)
        return 1
    except Exception:
        traceback.print_exc()
        print(f"{stage_id}: ERROR: unexpected failure (traceback above)", file=sys.stderr, flush=True)
        return 1
    log(f"{stage_id}: done in {time.monotonic() - t0:.1f}s")
    return 0
