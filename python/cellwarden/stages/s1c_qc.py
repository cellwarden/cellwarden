"""S1c: rules-based QC decisions (docs/stages-spec.md section 3.4).

    python -m cellwarden.stages.s1c_qc --run-dir DIR

Inputs: the complete ``S1/qc_metrics.tsv.gz`` (after S1b) and
``config/qc_policy.yaml``. Outputs:

* ``S1/qc_decisions.tsv.gz``: ``cell_id``, ``sample_id``, ``decision``,
  ``reasons``, ``policy_hash``, ``decided_by`` (``rules``), one row per cell
  in the order of ``qc_metrics.tsv.gz``.
* ``S1/qc_summary.json``: thresholds per sample, counts of keep / exclude /
  review per sample and per reason, protected-population losses, triggered
  review rules, the per-sample / per-pre-cluster simulation, policy hash,
  run id, config hash and the input checksum.

After both are written, ``S1/metrics_counts.h5ad`` is deleted (spec section 2:
"deleted when S1c is done"); its size and checksum stay in the summary
under ``deleted_exchange``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import StageError, log, stage_main, utc_now, write_json, write_tsv_gz

STAGE = "S1c"


def run(run_dir: Path, cfg: dict[str, Any]) -> None:
    from .. import qc
    from ..config import sha256_file

    s1 = run_dir / "S1"
    tsv = s1 / "qc_metrics.tsv.gz"
    pol = Path(cfg.get("qc_policy") or run_dir / "config" / "qc_policy.yaml")
    if not tsv.exists():
        raise StageError(f"input {tsv} not found (S1a and S1b must run first)")
    try:
        policy = qc.load_policy(pol)
    except qc.QCError as e:
        raise StageError(f"QC policy: {e}") from e
    metrics = qc.read_metrics(tsv)
    for c in ("doublet_call", "pre_cluster"):
        if c not in metrics.columns:
            raise StageError(f"{tsv} lacks {c!r}: S1b has not completed it")
    log(f"{len(metrics)} cells, {metrics['sample_id'].nunique()} sample(s); policy {pol} (hash {policy.hash[:12]})")
    try:
        thr = qc.compute_thresholds(metrics, policy)
        dec = qc.decide(metrics, thr, policy)
    except qc.QCError as e:
        raise StageError(str(e)) from e

    write_tsv_gz(dec.table, s1 / "qc_decisions.tsv.gz", na_rep="")
    summary = {
        "stage": STAGE,
        "created_utc": utc_now(),
        "run_id": cfg.get("run_id"),
        "config_hash": cfg.get("config_hash"),
        "qc_metrics_sha256": sha256_file(tsv),
        "policy_file": str(pol),
        **dec.summary(),
    }
    c = summary["counts"]
    log(f"decisions: keep {c['keep']}, exclude {c['exclude']}, review {c['review']}")
    for t in dec.triggered:
        where = t.get("sample_id") or t.get("pre_cluster")
        log(f"  review trigger {t['rule']} on {where}: {t['pct_excluded']:.1f}% > {t['limit_pct']:g}%")
    for p in dec.protected:
        if p.n_flagged:
            log(f"  protected {p.name}: {p.n_protected} cells, {p.n_flagged} flagged, "
                f"{p.n_rescued} to review, {p.n_lost} lost")

    h5 = s1 / "metrics_counts.h5ad"
    if h5.exists():
        summary["deleted_exchange"] = {"path": "S1/metrics_counts.h5ad", "bytes": h5.stat().st_size,
                                       "sha256": sha256_file(h5)}
    write_json(summary, s1 / "qc_summary.json")
    if h5.exists():
        h5.unlink()
        log(f"deleted {h5} (no longer needed)")


def main(argv: list[str] | None = None) -> int:
    return stage_main(STAGE, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
