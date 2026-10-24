"""S4, Open Targets step: evidence for the DGE hits (docs/agents-spec.md section 6).

    python -m cellwarden.stages.s4_opentargets --run-dir DIR [--refresh]

Runs after the R DGE step (``Rscript R/stages/run_stage.R --stage S4``; the
runner's S4 stage, ``cellwarden.stages.s4``, runs both). Reads
``S4/dge_summary.json``, the ``S4/dge/*.tsv.gz`` tables it lists and
``S4/gene_map.tsv``; writes ``S4/opentargets_summary.json`` (shape in
``cellwarden.opentargets.summary``).

Config (``opentargets:``): ``enabled`` (false: the summary is written with
``enabled: false`` and no lookups), ``max_genes_per_direction``,
``cache_dir`` (shared across runs; absolute in the resolved config),
``offline`` (cache only; misses are listed under
``genes_not_resolved_offline``), ``timeout_s``, ``retries``.
``--refresh`` ignores cached responses and refetches (not in offline mode).

A gene whose lookup fails after the retries does not fail the stage: the
error is recorded per gene under ``errors`` and the stage exits 0, so an
unreachable API never blocks a run. Rerun with ``--refresh`` later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from . import StageError, log
from ..opentargets import OpenTargetsClient
from ..opentargets.summary import build_summary, write_json

STAGE = "S4"
STEP = "S4/opentargets"


def run(run_dir: Path, cfg: dict[str, Any], *, refresh: bool = False, client: OpenTargetsClient | None = None) -> dict[str, Any]:
    s4 = run_dir / "S4"
    summ_path = s4 / "dge_summary.json"
    if not summ_path.exists():
        raise StageError(f"{summ_path} not found: run the S4 DGE step (Rscript R/stages/run_stage.R --stage S4) first")
    try:
        dge = json.loads(summ_path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise StageError(f"{summ_path} is not valid JSON: {e}") from e
    oc = dict(cfg.get("opentargets") or {})
    enabled = bool(oc.get("enabled", True))
    offline = bool(oc.get("offline", False))
    cache_dir = oc.get("cache_dir") or "refs/opentargets"
    if not os.path.isabs(cache_dir):
        cache_dir = str(Path(cfg.get("config_dir") or run_dir) / cache_dir)
    max_genes = int(oc.get("max_genes_per_direction", 50))
    if client is None:
        client = OpenTargetsClient(cache_dir, offline=offline, timeout=float(oc.get("timeout_s", 30)),
                                   retries=int(oc.get("retries", 3)), refresh=bool(refresh and not offline))
    log(f"{STEP}: enabled={enabled} offline={client.offline} refresh={client.refresh} cache={client.cache_dir} "
        f"max_genes_per_direction={max_genes}")
    n_tables = sum(1 for c in dge.get("contrasts") or [] for g in c.get("groups") or [] if g.get("file"))
    log(f"{STEP}: {len(dge.get('contrasts') or [])} contrast(s), {n_tables} DGE table(s)")
    out = build_summary(dge, run_dir, client, max_genes_per_direction=max_genes, enabled=enabled, log=log)
    out["run_id"] = cfg.get("run_id")
    out["config_hash"] = cfg.get("config_hash")
    write_json(out, s4 / "opentargets_summary.json")
    c = out["counts"]
    log(f"{STEP}: {c['genes_unique']} unique gene(s) listed, {c['genes_with_record']} with an Open Targets record, "
        f"{c['genes_without_record']} without, {c['genes_not_resolved_offline']} offline miss(es), "
        f"{c['genes_with_error']} error(s); cache hits {c['cache_hits']}, network calls {c['network_calls']}")
    return out


def main(argv: list[str] | None = None) -> int:
    from ..config import ConfigError, load_resolved

    p = argparse.ArgumentParser(prog="python -m cellwarden.stages.s4_opentargets")
    p.add_argument("--run-dir", default=os.environ.get("CELLWARDEN_RUN_DIR"), required="CELLWARDEN_RUN_DIR" not in os.environ)
    p.add_argument("--refresh", action="store_true", help="ignore cached responses and refetch")
    a = p.parse_args(argv)
    run_dir = Path(a.run_dir).expanduser().resolve()
    t0 = time.monotonic()
    log(f"{STEP}: start (run dir {run_dir}, python {sys.version.split()[0]})")
    try:
        cfg = load_resolved(run_dir)
        run(run_dir, cfg, refresh=a.refresh)
    except (StageError, ConfigError) as e:
        print(f"{STEP}: ERROR: {e}", file=sys.stderr, flush=True)
        return 1
    except Exception:
        traceback.print_exc()
        print(f"{STEP}: ERROR: unexpected failure (traceback above)", file=sys.stderr, flush=True)
        return 1
    log(f"{STEP}: done in {time.monotonic() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
