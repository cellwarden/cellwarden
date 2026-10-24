"""S3 (consensus half): evidence bundle + code-only consensus annotation.

    python -m cellwarden.stages.s3_consensus --run-dir DIR

Runs after the R step (``Rscript R/stages/run_stage.R --stage S3``) and
``cellwarden.stages.s3_celltypist``; ``cellwarden.stages.s3`` chains all three for the
runner. Re-running this module alone rebuilds the bundle and the consensus
(e.g. after editing ``markers_human.yaml``) without touching the R outputs.

Inputs (all in ``S3/``): ``markers.tsv.gz``, ``cluster_summary.json``,
``signatures.tsv.gz``, ``s3_r_summary.json``, and optionally
``celltypist.tsv.gz`` + ``celltypist_note.json``.

Outputs:

* ``S3/evidence_bundle.json`` - everything aggregated per cluster, **no
  per-cell data** (docs/agents-spec.md section 4)::

      {"stage": "S3", "run_id", "config_hash", "created_utc",
       "n_cells", "n_clusters", "samples": [...],
       "clusters": {"<cluster>": {
           "n_cells", "pct_cells", "per_sample": {sample: n},
           "median_genes", "median_umis", "median_pct_mito",
           "top_markers": [{"gene", "avg_log2fc", "pct_in", "pct_out", "p_adj", "rank"}, ...],   # <= 50
           "signatures": {set: mean_score | null},
           "signature_detail": {set: {"mean_score", "median_score", "pct_positive"}},
           "celltypist": {"majority_voting": {label: fraction}, "predicted_label": {label: fraction},
                          "mean_conf_score": x} | null,
           "consensus": {"label_broad", "label_fine", "cl_id", "confidence", "status"}}},
       "signature_sets": {set: {"category", "cell_type", "n_genes_set", "n_genes_found", "genes_found", "method"}},
       "signature_file": path, "markers": {method, ...},
       "celltypist": {"status": "ok" | "skipped", "reason", "model", ...},
       "sample_metadata_hint": {column: [values]},      # sample sheet extra columns, values only
       "annotation": {"min_confidence", "weights", "reference": path},
       "consensus_summary": {"n_consensus", "n_low_confidence", "n_conflict", "labels_broad": {label: n_clusters}}}

* ``S3/annotation_consensus.tsv`` (plain TSV, spec section 5):
  ``cluster, label_broad, label_fine, cl_id, confidence, votes_json, status``.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import StageError, log, read_tsv_str, stage_main, utc_now, write_json
from ..annotation import consensus as cons

STAGE = "S3-consensus"
BUNDLE_FILE = "evidence_bundle.json"
CONSENSUS_FILE = "annotation_consensus.tsv"


def _read_json(p: Path) -> Any:
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _num(x: str) -> float | None:
    if x is None or x == "" or x == "NA":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def load_r_outputs(s3: Path) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    need = ["markers.tsv.gz", "cluster_summary.json", "signatures.tsv.gz", "s3_r_summary.json"]
    missing = [f for f in need if not (s3 / f).exists()]
    if missing:
        raise StageError(f"missing S3 input(s) from the R step: {', '.join(missing)} (run Rscript R/stages/run_stage.R --stage S3 first)")
    markers = read_tsv_str(s3 / "markers.tsv.gz")
    for col in ("cluster", "gene", "avg_log2fc", "pct_in", "pct_out", "p_adj", "rank"):
        if col not in markers.columns:
            raise StageError(f"markers.tsv.gz lacks column {col!r}")
    summary = _read_json(s3 / "cluster_summary.json")
    sigs = read_tsv_str(s3 / "signatures.tsv.gz")
    for col in ("cluster", "signature", "mean_score", "n_genes_set", "n_genes_found"):
        if col not in sigs.columns:
            raise StageError(f"signatures.tsv.gz lacks column {col!r}")
    rsum = _read_json(s3 / "s3_r_summary.json")
    return markers, summary, sigs, rsum


def celltypist_per_cluster(s3: Path, cell_clusters: pd.Series | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Aggregate celltypist.tsv.gz per cluster (fractions); returns (note, {cluster: {...}} | None)."""
    note_p = s3 / "celltypist_note.json"
    note = _read_json(note_p) if note_p.exists() else {"status": "skipped", "reason": "celltypist step did not run"}
    tsv = s3 / "celltypist.tsv.gz"
    if note.get("status") != "ok" or not tsv.exists():
        return note, None
    if cell_clusters is None:
        note = dict(note, status="skipped", reason="no per-cell cluster map (S3/cell_clusters.tsv.gz) to aggregate votes")
        return note, None
    ct = read_tsv_str(tsv)
    ct = ct.set_index("cell_id")
    common = cell_clusters.index.intersection(ct.index)
    if len(common) == 0:
        note = dict(note, status="skipped", reason="celltypist cell ids do not match the S2 object")
        return note, None
    ct = ct.loc[common]
    cl = cell_clusters.loc[common]
    out: dict[str, Any] = {}
    for c, idx in cl.groupby(cl).groups.items():
        sub = ct.loc[idx]
        mv = (sub["majority_voting"].value_counts(normalize=True)).round(4)
        pl = (sub["predicted_label"].value_counts(normalize=True)).round(4)
        conf = pd.to_numeric(sub["conf_score"], errors="coerce")
        out[str(c)] = {"n_cells": int(len(sub)), "majority_voting": {str(k): float(v) for k, v in mv.items()},
                       "predicted_label": {str(k): float(v) for k, v in pl.items()},
                       "mean_conf_score": None if conf.isna().all() else round(float(conf.mean()), 4)}
    return note, out


def build_bundle(run_dir: Path, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    s3 = run_dir / "S3"
    markers, summary, sigs, rsum = load_r_outputs(s3)
    a = dict(cfg.get("annotation") or {})
    min_conf = float(a.get("min_confidence", 0.6))
    ref = cons.load_reference()

    clusters: dict[str, Any] = {}
    order = [str(c) for c in (rsum.get("clusters") or list(summary.keys()))]
    for c in order:
        info = dict(summary.get(c) or {})
        m = markers[markers["cluster"] == c].copy()
        m["rank_i"] = pd.to_numeric(m["rank"], errors="coerce")
        m = m.sort_values("rank_i")
        top = [{"gene": r.gene, "avg_log2fc": _num(r.avg_log2fc), "pct_in": _num(r.pct_in), "pct_out": _num(r.pct_out),
                "p_adj": _num(r.p_adj), "rank": int(r.rank_i)} for r in m.itertuples()]
        sg = sigs[sigs["cluster"] == c]
        signatures = {r.signature: _num(r.mean_score) for r in sg.itertuples()}
        detail = {r.signature: {"mean_score": _num(r.mean_score), "median_score": _num(getattr(r, "median_score", "")),
                                "pct_positive": _num(getattr(r, "pct_positive", ""))} for r in sg.itertuples()}
        clusters[c] = {
            "n_cells": info.get("n_cells"), "pct_cells": info.get("pct_cells"), "per_sample": info.get("per_sample") or {},
            "median_genes": info.get("median_genes"), "median_umis": info.get("median_umis"),
            "median_pct_mito": info.get("median_pct_mito"),
            "top_markers": top, "signatures": signatures, "signature_detail": detail, "celltypist": None,
        }

    # per-cell cluster map for celltypist aggregation (written by the R step; per-cell, never in the bundle)
    cc_path = s3 / "cell_clusters.tsv.gz"
    cell_clusters = None
    if cc_path.exists():
        cc = read_tsv_str(cc_path)
        cell_clusters = cc.set_index("cell_id")["cluster"].astype(str)
    ct_note, ct_votes = celltypist_per_cluster(s3, cell_clusters)
    if ct_votes:
        for c in clusters:
            clusters[c]["celltypist"] = ct_votes.get(c)

    sig_sets = {}
    for name, meta in ((rsum.get("signatures") or {}).get("sets") or {}).items():
        sig_sets[name] = {k: meta.get(k) for k in ("category", "cell_type", "n_genes_set", "n_genes_found", "genes_found", "method")}
    if not sig_sets:  # fall back to the TSV's own columns
        for r in sigs.drop_duplicates("signature").itertuples():
            sig_sets[r.signature] = {"category": getattr(r, "category", "lineage"), "cell_type": getattr(r, "cell_type", None),
                                     "n_genes_set": int(float(r.n_genes_set)), "n_genes_found": int(float(r.n_genes_found))}

    records = cons.annotate(clusters, sig_sets, ref, min_confidence=min_conf)
    for r in records:
        clusters[r["cluster"]]["consensus"] = {k: r[k] for k in ("label_broad", "label_fine", "cl_id", "confidence", "status")}
    labels: dict[str, int] = {}
    for r in records:
        labels[r["label_broad"]] = labels.get(r["label_broad"], 0) + 1
    bundle = {
        "stage": "S3", "run_id": cfg.get("run_id"), "config_hash": cfg.get("config_hash"), "created_utc": utc_now(),
        "n_cells": rsum.get("n_cells"), "n_genes": rsum.get("n_genes"), "n_clusters": len(clusters),
        "samples": rsum.get("samples") or [],
        "clusters": clusters,
        "signature_sets": sig_sets,
        "signature_file": (rsum.get("signatures") or {}).get("file"),
        "markers": rsum.get("markers") or {},
        "celltypist": ct_note,
        "sample_metadata_hint": rsum.get("sample_metadata_hint") or {},
        "annotation": {"min_confidence": min_conf, "weights": dict(cons.DEFAULT_WEIGHTS), "reference": ref.path,
                       "celltypist_model": a.get("celltypist_model", "Immune_All_Low.pkl")},
        "consensus_summary": {
            "n_consensus": sum(r["status"] == "consensus" for r in records),
            "n_low_confidence": sum(r["status"] == "low_confidence" for r in records),
            "n_conflict": sum(r["status"] == "conflict" for r in records),
            "labels_broad": dict(sorted(labels.items())),
        },
    }
    return bundle, records


def write_consensus_tsv(records: list[dict[str, Any]], path: Path) -> None:
    df = cons.consensus_table(records)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, sep="\t", index=False, lineterminator="\n")
    tmp.replace(path)


def run(run_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    s3 = run_dir / "S3"
    bundle, records = build_bundle(run_dir, cfg)
    write_consensus_tsv(records, s3 / CONSENSUS_FILE)
    write_json(bundle, s3 / BUNDLE_FILE)
    cs = bundle["consensus_summary"]
    log(f"consensus: {cs['n_consensus']} consensus, {cs['n_low_confidence']} low_confidence, {cs['n_conflict']} conflict; "
        f"celltypist {bundle['celltypist'].get('status')}")
    for r in records:
        log(f"  cluster {r['cluster']}: {r['label_broad']} / {r['label_fine']} ({r['cl_id']}) conf {r['confidence']:.2f} {r['status']}")
    return bundle


def main(argv: list[str] | None = None) -> int:
    return stage_main(STAGE, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
