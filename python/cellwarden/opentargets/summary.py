"""Build ``S4/opentargets_summary.json`` from the S4 DGE tables.

(docs/agents-spec.md section 6)

Per contrast x group x direction (``up`` / ``down``): up to
``max_genes_per_direction`` genes ordered by padj (ties: larger |log2fc|
first), each with its DGE statistics and a pointer to its Open Targets
evidence. **Direction always comes from our DGE table**; Open Targets has
no notion of up or down.

Shape::

    {
      "stage": "S4", "step": "opentargets", "created_utc": ..., "enabled": true,
      "offline": false, "endpoint": ..., "cache_dir": ..., "max_genes_per_direction": 50,
      "thresholds": {"padj": 0.05, "lfc": 0.5},
      "gene_id_source": "ensembl_id|symbol",
      "contrasts": [{"contrast", "numerator", "denominator",
                     "groups": [{"group", "file", "skipped_reason",
                                 "directions": {"up": {"n_de", "n_listed", "n_with_record",
                                                       "genes": [{"gene", "symbol", "ensembl_id", "rank",
                                                                  "log2fc", "padj", "baseMean", "direction",
                                                                  "has_record", "target": <gene key or null>}]},
                                                "down": {...}}}]}],
      "targets": {<gene key>: <target evidence from client.shape_target>},   # one entry per gene, deduplicated
      "genes_without_record": [<gene key>, ...],
      "genes_not_resolved_offline": [<gene key>, ...],
      "errors": [{"gene", "error"}],
      "counts": {"contrasts", "groups", "groups_with_tables", "genes_listed", "genes_unique",
                 "genes_with_record", "genes_without_record", "genes_not_resolved_offline",
                 "known_drug_rows", "cache_hits", "cache_misses", "network_calls"}
    }

Gene keys are the ``gene`` column of the DGE tables (feature names of the
Seurat object). ``S4/gene_map.tsv`` (gene, symbol, ensembl_id) maps them to
symbols and Ensembl IDs: a gene with an Ensembl ID is looked up directly,
otherwise its symbol is resolved through the search endpoint.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .client import OpenTargetsClient, OpenTargetsError

__all__ = ["read_dge_table", "read_gene_map", "build_summary", "DGE_COLUMNS"]

DGE_COLUMNS = ("gene", "baseMean", "log2fc", "lfcSE", "pvalue", "padj", "direction")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(x: str | None) -> float | None:
    if x is None or x == "" or x == "NA":
        return None
    try:
        v = float(x)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def read_dge_table(path: str | Path) -> list[dict[str, Any]]:
    """Rows of a ``dge/<contrast>__<group>.tsv.gz`` table with numeric fields parsed."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        missing = [c for c in DGE_COLUMNS if c not in (rd.fieldnames or [])]
        if missing:
            raise ValueError(f"{p}: missing column(s) {missing} (have {rd.fieldnames})")
        rows = []
        for r in rd:
            rows.append({"gene": r["gene"], "baseMean": _num(r["baseMean"]), "log2fc": _num(r["log2fc"]),
                         "lfcSE": _num(r["lfcSE"]), "pvalue": _num(r["pvalue"]), "padj": _num(r["padj"]),
                         "direction": r["direction"]})
    return rows


def read_gene_map(path: str | Path | None) -> dict[str, dict[str, str]]:
    """``{gene: {symbol, ensembl_id}}`` from ``S4/gene_map.tsv``; empty when the file is absent."""
    if path is None or not Path(path).exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            out[r["gene"]] = {"symbol": (r.get("symbol") or "").strip(), "ensembl_id": (r.get("ensembl_id") or "").strip()}
    return out


def _ranked(rows: list[dict[str, Any]], direction: str, n: int) -> list[dict[str, Any]]:
    de = [r for r in rows if r["direction"] == direction and r["padj"] is not None]
    de.sort(key=lambda r: (r["padj"], -abs(r["log2fc"] or 0.0), r["gene"]))
    return de[:n]


def _lookup(client: OpenTargetsClient, gene: str, gmap: dict[str, dict[str, str]]) -> tuple[dict[str, Any] | None, dict[str, str]]:
    """(evidence or {} or None, {symbol, ensembl_id}) for one DGE gene key."""
    info = gmap.get(gene) or {}
    symbol = info.get("symbol") or gene
    ens = info.get("ensembl_id") or (gene if gene.startswith("ENSG") else "")
    if ens:
        ev = client.target_by_ensembl(ens)
        if ev:
            ev["query_symbol"] = symbol
            ev["lookup"] = "ensembl_id"
        elif ev == {} and symbol and not symbol.startswith("ENSG"):
            ev = client.target_by_symbol(symbol)
            if ev:
                ev["lookup"] = "symbol (ensembl_id unknown to Open Targets)"
        return ev, {"symbol": symbol, "ensembl_id": ens}
    ev = client.target_by_symbol(symbol)
    if ev:
        ev["lookup"] = "symbol"
        ens = ev.get("ensembl_id") or ens
    return ev, {"symbol": symbol, "ensembl_id": ens}


def build_summary(
    dge_summary: dict[str, Any],
    run_dir: str | Path,
    client: OpenTargetsClient,
    *,
    max_genes_per_direction: int = 50,
    enabled: bool = True,
    gene_map: dict[str, dict[str, str]] | None = None,
    log=None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    say = log or (lambda *a: None)
    gmap = gene_map if gene_map is not None else read_gene_map(run_dir / "S4" / "gene_map.tsv")
    thr = dge_summary.get("thresholds") or {}
    out: dict[str, Any] = {
        "stage": "S4", "step": "opentargets", "created_utc": _utc(), "enabled": bool(enabled),
        "offline": client.offline, "endpoint": client.endpoint,
        "cache_dir": str(client.cache_dir) if client.cache_dir else None,
        "max_genes_per_direction": int(max_genes_per_direction),
        "thresholds": {"padj": thr.get("padj"), "lfc": thr.get("lfc")},
        "gene_id_source": "ensembl_id" if any(v.get("ensembl_id") for v in gmap.values()) else "symbol",
        "dge_summary": "S4/dge_summary.json",
        "contrasts": [], "targets": {}, "genes_without_record": [], "genes_not_resolved_offline": [], "errors": [],
    }
    targets: dict[str, Any] = out["targets"]
    status: dict[str, str] = {}      # gene -> record | none | offline-miss | error
    errors: dict[str, str] = {}
    listed = 0
    n_groups = n_tables = 0

    for c in dge_summary.get("contrasts") or []:
        cout = {"contrast": c.get("name"), "numerator": c.get("numerator"), "denominator": c.get("denominator"), "groups": []}
        for g in c.get("groups") or []:
            n_groups += 1
            gout: dict[str, Any] = {"group": g.get("group"), "file": g.get("file"), "skipped_reason": g.get("skipped_reason"),
                                    "directions": {}}
            rows: list[dict[str, Any]] = []
            if g.get("file"):
                n_tables += 1
                rows = read_dge_table(run_dir / g["file"])
            for direction in ("up", "down"):
                n_de = sum(1 for r in rows if r["direction"] == direction)
                genes_out = []
                for rank, r in enumerate(_ranked(rows, direction, max_genes_per_direction), start=1):
                    gene = r["gene"]
                    listed += 1
                    if enabled and gene not in status:
                        try:
                            ev, info = _lookup(client, gene, gmap)
                        except OpenTargetsError as e:
                            ev, info = None, (gmap.get(gene) or {"symbol": gene, "ensembl_id": ""})
                            errors[gene] = str(e)
                            status[gene] = "error"
                            say(f"opentargets: {gene}: {e}")
                        else:
                            if ev is None:
                                status[gene] = "offline-miss"
                            elif ev:
                                status[gene] = "record"
                                targets[gene] = ev
                            else:
                                status[gene] = "none"
                        gmap.setdefault(gene, {}).update({k: v for k, v in info.items() if v})
                    info = gmap.get(gene) or {}
                    st = status.get(gene, "not-queried")
                    genes_out.append({
                        "gene": gene, "symbol": info.get("symbol") or gene, "ensembl_id": info.get("ensembl_id") or None,
                        "rank": rank, "log2fc": r["log2fc"], "padj": r["padj"], "baseMean": r["baseMean"],
                        "direction": direction, "has_record": st == "record", "lookup_status": st,
                        "target": gene if st == "record" else None,
                    })
                gout["directions"][direction] = {"n_de": n_de, "n_listed": len(genes_out),
                                                 "n_with_record": sum(1 for x in genes_out if x["has_record"]),
                                                 "genes": genes_out}
            cout["groups"].append(gout)
        out["contrasts"].append(cout)

    out["genes_without_record"] = sorted(g for g, s in status.items() if s == "none")
    out["genes_not_resolved_offline"] = sorted(g for g, s in status.items() if s == "offline-miss")
    out["errors"] = [{"gene": g, "error": e} for g, e in sorted(errors.items())]
    stats = client.stats()
    out["counts"] = {
        "contrasts": len(out["contrasts"]), "groups": n_groups, "groups_with_tables": n_tables,
        "genes_listed": listed, "genes_unique": len(status),
        "genes_with_record": sum(1 for s in status.values() if s == "record"),
        "genes_without_record": len(out["genes_without_record"]),
        "genes_not_resolved_offline": len(out["genes_not_resolved_offline"]),
        "genes_with_error": len(errors),
        "known_drug_rows": sum(len(t.get("known_drugs") or []) for t in targets.values()),
        "cache_hits": stats["cache_hits"], "cache_misses": stats["cache_misses"], "network_calls": stats["network_calls"],
    }
    return out


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)
