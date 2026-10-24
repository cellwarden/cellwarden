"""Choose the differential expression comparisons once the cell types are known.

``dge.interactive: true`` (set by ``cellwarden run -i`` when you choose "after clustering and
annotation", or by hand) makes the runner stop just before S4, when the clusters (S2/S3) and the
reviewed cell types (G3) exist, and ask in the terminal:

* markers of each cell type against the other cells;
* two cell types or clusters (picked by number from the list, with cell counts);
* two groups of samples (any samples.csv column with two or more values: condition, donor,
  sex, age ...), within each cell type or across all cells;

and more comparisons if wanted. The answer is written into ``config/config.resolved.yaml``
(``dge.contrasts``, ``dge.markers``) and recorded in ``config/dge_choice.yaml``; a later
``cellwarden resume`` keeps it (delete ``dge_choice.yaml`` to be asked again). Without a terminal
(``--detach`` with nohup, scheduler jobs) the run stops before S4 like a waiting gate, and
``cellwarden resume`` in a terminal asks.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from .stage_names import title

__all__ = ["CHOICE_FILE", "needs_choice", "available_groups", "ask_choice", "apply_choice"]

CHOICE_FILE = "dge_choice.yaml"
_SHEET_SKIP = ("path", "format")

Ask = Callable[[str, str], str]
Say = Callable[[str], None]


def needs_choice(cfg: Mapping[str, Any], run_dir: str | Path) -> bool:
    """True when the run asked to choose the comparisons after annotation and has not yet."""
    d = cfg.get("dge") or {}
    return bool(d.get("interactive")) and not (Path(run_dir) / "config" / CHOICE_FILE).exists()


def _json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def available_groups(run_dir: str | Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """What can be compared: ``{"cell": {column: {value: n_cells}}, "sample": {column: {value: n_samples}}}``.

    Cell types come from ``gates/G3/decision.json`` (else ``S3/annotation_consensus.tsv`` is not
    needed: clusters alone are offered), cell counts from ``S3/cluster_summary.json``, sample
    groups from the run's ``samples.csv``.
    """
    rd = Path(run_dir)
    sizes = {str(k): int((v or {}).get("n_cells") or 0)
             for k, v in (_json(rd / "S3" / "cluster_summary.json") or {}).items() if isinstance(v, Mapping)}
    cell: dict[str, dict[str, int]] = {}
    dec = _json(rd / "gates" / "G3" / "decision.json") or {}
    clusters = ((dec.get("decision") or {}).get("clusters")) or dec.get("clusters") or []
    for col, key in (("cell_type_broad", "label_broad"), ("cell_type_fine", "label_fine")):
        counts: dict[str, int] = {}
        for c in clusters:
            lab = str(c.get(key) or c.get("label_broad") or "unassigned")
            counts[lab] = counts.get(lab, 0) + sizes.get(str(c.get("cluster")), 0)
        if counts:
            cell[col] = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    if sizes:
        cell["seurat_clusters"] = dict(sorted(sizes.items(), key=lambda kv: (int(kv[0]) if kv[0].isdigit() else 1e9, kv[0])))
    sample: dict[str, dict[str, int]] = {}
    try:
        with open(str(cfg.get("samples")), encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, TypeError, csv.Error):
        rows = []
    for col in (rows[0].keys() if rows else []):
        if col in _SHEET_SKIP:
            continue
        counts = {}
        for r in rows:
            v = (r.get(col) or "").strip()
            if v:
                counts[v] = counts.get(v, 0) + 1
        if len(counts) >= 2:
            sample[col] = dict(sorted(counts.items()))
    return {"cell": cell, "sample": sample}


CELL_COLUMN_NAMES = {"cell_type_broad": "cell types", "cell_type_fine": "fine cell types", "seurat_clusters": "clusters"}


def _pick(q: Any, say: Say, text: str, values: list[str], allow_rest: bool, default: str) -> list[str]:
    """Values chosen by number or name (comma-separated); ``rest`` when allowed."""
    low = {v.lower(): v for v in values}
    while True:
        got = [t.strip() for t in q.ask(text, default).split(",") if t.strip()]
        if allow_rest and [g.lower() for g in got] == ["rest"]:
            return ["rest"]
        out, bad = [], []
        for t in got:
            if t.isdigit() and 1 <= int(t) <= len(values) and t not in low:
                out.append(values[int(t) - 1])
            elif t.lower() in low:
                out.append(low[t.lower()])
            else:
                bad.append(t)
        if out and not bad:
            return list(dict.fromkeys(out))
        say(f"  not in the list: {', '.join(bad)}" if bad else "  please choose at least one")


def _name(g1: list[str], g2: list[str], used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{'+'.join(g1)}_vs_{'+'.join(g2)}").strip("-") or "contrast"
    if not re.match(r"[A-Za-z0-9]", base):
        base = "c" + base
    name, k = base[:80], 2
    while name in used:
        name, k = f"{base[:76]}-{k}", k + 1
    used.add(name)
    return name


def ask_choice(cfg: Mapping[str, Any], run_dir: str | Path, ask: Ask, say: Say) -> dict[str, Any] | None:
    """Ask for the comparisons; returns ``{"markers": bool, "contrasts": [...]}``, or None when cancelled."""
    from .wizard import _Asker

    q = _Asker(ask, say)
    d = cfg.get("dge") or {}
    pseudobulk = d.get("method", "findmarkers") != "findmarkers"
    avail = available_groups(run_dir, cfg)
    say(f"\nClustering and the {title('G3').lower()} are done. Choose the differential expression comparison(s).")
    for col, vals in avail["cell"].items():
        if col == "cell_type_fine" and list(vals) == list(avail["cell"].get("cell_type_broad", {})):
            continue
        say(f"  {CELL_COLUMN_NAMES[col]}: " + ", ".join(f"{v} ({n} cells)" for v, n in vals.items()))
    for col, vals in avail["sample"].items():
        say(f"  {col}: " + ", ".join(f"{v} ({n} sample{'s' if n > 1 else ''})" for v, n in vals.items()))

    options = []
    if not pseudobulk:
        options += [("markers", "markers of each cell type against the other cells")]
        if avail["cell"]:
            options += [("cells", "two cell types or clusters")]
    if avail["sample"]:
        options += [("samples", "two groups of samples (" + ", ".join(avail["sample"]) + ")")]
    if not options:
        say("  pseudobulk needs a samples.csv column with two or more values; none found, so no comparison")
        return {"markers": False, "contrasts": []}
    options += [("none", "no differential expression")]

    markers, contrasts, used = False, [], set()
    while True:
        what = q.choice("Compare", options, options[0][0])
        if what == "markers":
            markers = True
        elif what == "cells":
            cols = [c for c in avail["cell"] if not (c == "cell_type_fine" and list(avail["cell"][c]) ==
                                                     list(avail["cell"].get("cell_type_broad", {})))]
            col = cols[0] if len(cols) == 1 else q.choice(
                "Groups from", [(c, CELL_COLUMN_NAMES[c]) for c in cols], cols[0])
            values = list(avail["cell"][col])
            for k, v in enumerate(values, 1):
                say(f"  [{k}] {v} ({avail['cell'][col][v]} cells)")
            g1 = _pick(q, say, "Group 1 (numbers or names, comma-separated)", values, False, "1")
            g2 = _pick(q, say, "Group 2 (numbers or names, or 'rest' for all other cells)", values, True,
                       "rest" if len(values) > 2 else "2")
            contrasts.append({"name": _name(g1, g2, used), "column": col, "numerator": g1 if len(g1) > 1 else g1[0],
                              "denominator": g2 if len(g2) > 1 else g2[0], "within": "none"})
        elif what == "samples":
            cols = list(avail["sample"])
            col = cols[0] if len(cols) == 1 else q.choice("Column", [(c, c) for c in cols],
                                                          next((c for c in cols if c != "sample_id"), cols[0]))
            values = list(avail["sample"][col])
            for k, v in enumerate(values, 1):
                say(f"  [{k}] {v} ({avail['sample'][col][v]} samples)")
            g1 = _pick(q, say, "Group 1 (numbers or names, comma-separated)", values, False, str(len(values)))
            g2 = _pick(q, say, "Group 2 (numbers or names" + ("" if pseudobulk else ", or 'rest'") + ")", values,
                       not pseudobulk, "1")
            within = d.get("group_by", "cell_type_broad") if (pseudobulk or q.yes("Within each cell type separately", True)) \
                else "none"
            contrasts.append({"name": _name(g1, g2, used), "column": col, "numerator": g1 if len(g1) > 1 else g1[0],
                              "denominator": g2 if len(g2) > 1 else g2[0], "within": within})
        else:
            break
        if not q.yes("Add another comparison", False):
            break
        options = [o for o in options if o[0] != "none" and not (o[0] == "markers" and markers)] + [("none", "no more")]
    return {"markers": markers, "contrasts": contrasts}


def apply_choice(run_dir: str | Path, cfg: dict[str, Any], choice: Mapping[str, Any]) -> None:
    """Put the choice into ``cfg`` and ``config/config.resolved.yaml``; record it in ``config/dge_choice.yaml``."""
    cdir = Path(run_dir) / "config"
    dge = dict(cfg.get("dge") or {})
    dge.update(markers=bool(choice.get("markers")), contrasts=list(choice.get("contrasts") or []))
    cfg["dge"] = dge
    p = cdir / "config.resolved.yaml"
    try:
        res = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except OSError:
        res = None
    if isinstance(res, dict):
        res["dge"] = dge
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(yaml.safe_dump(res, sort_keys=False), encoding="utf-8")
        tmp.replace(p)
    rec = {"chosen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "markers": dge["markers"], "contrasts": dge["contrasts"]}
    (cdir / CHOICE_FILE).write_text("# Chosen before S4 (dge.interactive); delete this file to be asked again on resume.\n"
                                    + yaml.safe_dump(rec, sort_keys=False), encoding="utf-8")
