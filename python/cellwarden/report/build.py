"""Build ``S5/report.html``, ``S5/explorer.html`` and the gate reports
(docs/agents-spec.md section 7).

Public functions:

* :func:`collect` reads everything a run directory has (state.json, stage
  summaries, S5 exports, gates, the LLM ledger) into one context dict. Every
  input is optional: a missing one leaves its section marked ``not run``.
* :func:`render_report(ctx)` / :func:`build_report(run_dir, out)` write the
  single-file report (jinja2, inline CSS/JS, plots as inline SVG or base64
  PNG, no CDN).
* :func:`build_explorer(run_dir, out)` writes the canvas-based UMAP explorer
  with the data embedded as JSON (``report.explorer_max_cells`` points).
* :func:`render_gate_report(gate_dir) -> str` renders one gate's
  ``bundle/proposal/review/decision.json`` with the same templates; the gates
  package calls it to write ``gates/Gx/report.html``.

Shapes of the inputs are those of docs/agents-spec.md; unknown shapes are
shown as pretty-printed JSON rather than failing.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, Iterable, Mapping

from . import plots
from ..stage_names import slug as stage_slug, title as stage_title

__all__ = ["collect", "render_report", "build_report", "build_explorer", "render_gate_report", "build_all"]

TEMPLATES = Path(__file__).parent / "templates"
GATES = ("G1", "G2", "G3", "G4")
SECTIONS = ("summary", "resources", "qc", "clusters", "markers", "dge", "opentargets", "gates", "llm", "software")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Readers (all tolerant)
# --------------------------------------------------------------------------


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _tsv(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8", newline="") as fh:
            rd = csv.DictReader(fh, delimiter="\t")
            rows = []
            for i, r in enumerate(rd):
                if limit is not None and i >= limit:
                    break
                rows.append(r)
            return rows
    except (OSError, ValueError):
        return []


def _float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _fmt_s(s: Any) -> str:
    if s is None:
        return "-"
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def _fmt_bytes(b: Any) -> str:
    if b is None:
        return "-"
    b = float(b)
    for unit, div in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if b >= div:
            return f"{b / div:.2f} {unit}"
    return f"{int(b)} B"


def _yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


# --------------------------------------------------------------------------
# Collect
# --------------------------------------------------------------------------


def collect(run_dir: str | Path, *, max_cells_plot: int | None = None) -> dict[str, Any]:
    """Everything the report shows, from whatever the run directory holds."""
    rd = Path(run_dir)
    cfg = _yaml(rd / "config" / "config.resolved.yaml")
    state = _json(rd / "state.json") or {}
    rep_cfg = cfg.get("report") or {}
    ctx: dict[str, Any] = {
        "generated_utc": _utc(),
        "run": {
            "run_id": cfg.get("run_id") or state.get("run_id") or rd.name,
            "project": cfg.get("project"),
            "run_dir": str(rd),
            "config_hash": cfg.get("config_hash") or state.get("config_hash"),
            "cellwarden_version": cfg.get("cellwarden_version"),
            "execution": cfg.get("execution", "direct"),
            "site": cfg.get("site"),
            "device": cfg.get("device"),
            "samples": [s.get("sample_id") for s in cfg.get("samples_resolved") or []],
        },
        "cfg": cfg,
        "stages": _stages(state),
        "missing": [],
    }
    ctx["resources"] = _resources(rd, cfg, state)
    ctx["ingest"] = _json(rd / "S0" / "ingest_report.json")
    ctx["qc"] = _qc(rd, cfg)
    umap = _tsv(rd / "S5" / "umap.tsv.gz")
    ctx["clusters"] = _clusters(rd, umap, rep_cfg)
    ctx["markers"] = _markers(rd, int(rep_cfg.get("top_markers") or 10))
    ctx["dge"] = _dge(rd)
    ctx["opentargets"] = _opentargets(rd)
    ctx["gates"] = [_gate(rd / "gates" / g, g) for g in GATES]
    ctx["llm"] = _llm(rd, cfg)
    ctx["software"] = _software(rd, cfg)
    for key, label in (("ingest", "S0 ingest"), ("qc", "S1 QC"), ("clusters", "S2 clustering / S5 export"),
                       ("markers", "S3 markers"), ("dge", "S4 DGE"), ("opentargets", "S4 Open Targets"), ("llm", "LLM ledger")):
        if not ctx.get(key):
            ctx["missing"].append(label)
    ctx["_umap_rows"] = umap  # for the explorer; not rendered
    return ctx


def _stages(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    order = state.get("stage_order") or list((state.get("stages") or {}).keys())
    rows = []
    for sid in order:
        r = (state.get("stages") or {}).get(sid) or {}
        rows.append({"id": sid, "lang": r.get("lang", ""), "status": r.get("status", "pending"),
                     "started": (r.get("started_utc") or "-").replace("T", " ").rstrip("Z"),
                     "wall": _fmt_s(r.get("wall_s")), "peak": _fmt_bytes(r.get("peak_rss_bytes")),
                     "exit": "-" if r.get("exit_code") is None else r.get("exit_code"),
                     "note": r.get("note") or r.get("reason") or ""})
    return rows


def _resources(rd: Path, cfg: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    s2 = _json(rd / "S2" / "summary.json") or {}
    res = dict(cfg.get("resources") or {})
    peak = max([(r.get("peak_rss_bytes") or 0) for r in (state.get("stages") or {}).values()] or [0])
    total = sum((r.get("wall_s") or 0) for r in (state.get("stages") or {}).values() if r.get("status") == "done")
    return {
        "caps": res,
        "device": s2.get("device"), "plan": s2.get("plan"), "reason": s2.get("reason"), "path": s2.get("path"),
        "peak_memory": s2.get("peak_memory"),
        "runner_host": (state.get("runner") or {}).get("host"),
        "max_peak_rss": _fmt_bytes(peak or None), "total_wall": _fmt_s(total),
        "execution": cfg.get("execution", "direct"), "container": cfg.get("container"),
    }


def _qc(rd: Path, cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    summ = _json(rd / "S1" / "qc_summary.json")
    if not summ:
        return None
    per_sample = summ.get("counts_per_sample") or {}
    thr = (summ.get("thresholds") or {}).get("per_sample") or {}
    rows = []
    for sid, c in per_sample.items():
        t = thr.get(sid) or {}
        nat = t.get("adaptive_natural_scale") or {}
        tot = sum(int(c.get(k, 0) or 0) for k in ("keep", "exclude", "review"))
        rows.append({"sample": sid, "cells": tot, "keep": c.get("keep", 0), "exclude": c.get("exclude", 0),
                     "review": c.get("review", 0),
                     "pct_removed": (100.0 * int(c.get("exclude", 0) or 0) / tot) if tot else 0.0,
                     "adaptive": t.get("adaptive_applied"), "note": t.get("note") or "",
                     "min_genes": nat.get("min_genes"), "max_genes": nat.get("max_genes"),
                     "min_umis": nat.get("min_umis"), "max_umis": nat.get("max_umis"),
                     "max_mito": nat.get("max_mito_pct")})
    decisions_plot = plots.stacked_bars([(r["sample"], {"keep": r["keep"], "review": r["review"], "exclude": r["exclude"]}) for r in rows],
                                        ["keep", "review", "exclude"], title="QC decisions per sample",
                                        colours={"keep": "#009E73", "review": "#E69F00", "exclude": "#D55E00"})
    reasons = sorted(((k, v) for k, v in (summ.get("counts_per_reason") or {}).items()), key=lambda kv: -kv[1])
    reasons_plot = plots.bar_chart(reasons, title="Reason codes (a cell can have several)")
    hists = _qc_histograms(rd, thr, summ)
    return {"summary": summ, "rows": rows, "decisions_plot": decisions_plot, "reasons": reasons,
            "reasons_plot": reasons_plot, "histograms": hists,
            "protected": summ.get("protected_losses") or summ.get("protected") or [],
            "triggered": summ.get("review_triggers_fired") or summ.get("triggered_review_rules") or summ.get("triggered") or [],
            "policy_hash": summ.get("policy_hash"), "decided_by": summ.get("decided_by"),
            "hard_limits": (summ.get("thresholds") or {}).get("hard_limits") or {}}


def _qc_histograms(rd: Path, thr: Mapping[str, Any], summ: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _tsv(rd / "S1" / "qc_metrics.tsv.gz")
    if not rows:
        return []
    by_sample: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_sample.setdefault(r.get("sample_id", "?"), []).append(r)
    hard = (summ.get("thresholds") or {}).get("hard_limits") or {}
    out = []
    for sid, rs in by_sample.items():
        nat = (thr.get(sid) or {}).get("adaptive_natural_scale") or {}
        out.append({
            "sample": sid, "n": len(rs),
            "genes": plots.histogram((_float(r.get("n_genes")) for r in rs), title=f"{sid}: genes per cell", log10=True,
                                     vlines={"hard min": hard.get("min_genes"), "adaptive min": nat.get("min_genes"),
                                             "adaptive max": nat.get("max_genes")}),
            "umis": plots.histogram((_float(r.get("n_umis")) for r in rs), title=f"{sid}: UMIs per cell", log10=True,
                                    vlines={"hard min": hard.get("min_umis"), "adaptive min": nat.get("min_umis"),
                                            "adaptive max": nat.get("max_umis")}, colour="#56B4E9"),
            "mito": plots.histogram((_float(r.get("pct_mito")) for r in rs), title=f"{sid}: % mitochondrial",
                                    vlines={"hard max": hard.get("max_mito_pct"), "adaptive max": nat.get("max_mito_pct")},
                                    colour="#D55E00"),
        })
    return out


def _clusters(rd: Path, umap: list[dict[str, str]], rep_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    s2 = _json(rd / "S2" / "summary.json")
    if not s2 and not umap:
        return None
    out: dict[str, Any] = {"summary": s2 or {}, "n_cells": len(umap), "export": _json(rd / "S5" / "export.json")}
    sizes = (s2 or {}).get("cluster_sizes") or {}
    if not sizes and umap:
        for r in umap:
            sizes[r.get("seurat_clusters", "?")] = sizes.get(r.get("seurat_clusters", "?"), 0) + 1
    out["sizes"] = sorted(((str(k), int(v)) for k, v in sizes.items()), key=lambda kv: plots._level_key(kv[0]))
    out["sizes_plot"] = plots.bar_chart(out["sizes"], title="Cells per cluster")
    out["umaps"] = []
    if umap:
        x = [_float(r.get("umap_1")) for r in umap]
        y = [_float(r.get("umap_2")) for r in umap]
        ok = [i for i in range(len(x)) if not (math.isnan(x[i]) or math.isnan(y[i]))]
        if ok:
            x = [x[i] for i in ok]
            y = [y[i] for i in ok]
            for col, title in (("seurat_clusters", "UMAP by cluster"), ("cell_type_broad", "UMAP by broad cell type"),
                               ("cell_type_fine", "UMAP by fine cell type"), ("sample_id", "UMAP by sample"),
                               ("condition", "UMAP by condition")):
                if col in umap[0]:
                    labs = [umap[i].get(col) or "NA" for i in ok]
                    out["umaps"].append(plots.scatter(x, y, labs, title=title))
    # cluster summary (S3) and annotation tables
    cs = _json(rd / "S3" / "cluster_summary.json") or {}
    ann = {r["cluster"]: r for r in _tsv(rd / "S3" / "annotation.tsv") if "cluster" in r}
    cons = {r["cluster"]: r for r in _tsv(rd / "S3" / "annotation_consensus.tsv") if "cluster" in r}
    rows = []
    for cl, n in out["sizes"]:
        c = cs.get(cl) or cs.get(str(cl)) or {}
        a = ann.get(cl) or {}
        k = cons.get(cl) or {}
        rows.append({"cluster": cl, "n_cells": n, "pct": c.get("pct_cells"),
                     "median_genes": c.get("median_genes"), "median_umis": c.get("median_umis"),
                     "median_pct_mito": c.get("median_pct_mito"),
                     "top_markers": ", ".join((c.get("top_markers") or [])[:8]),
                     "label_broad": a.get("label_broad") or k.get("label_broad") or "",
                     "label_fine": a.get("label_fine") or k.get("label_fine") or "",
                     "cl_id": a.get("cl_id") or k.get("cl_id") or "",
                     "confidence": a.get("confidence") or k.get("confidence") or "",
                     "status": a.get("status") or k.get("status") or ""})
    out["rows"] = rows
    out["has_s3"] = bool(cs)
    out["has_annotation"] = bool(ann or cons)
    out["celltypist"] = (rd / "S3" / "celltypist.tsv.gz").exists()
    return out


def _markers(rd: Path, top: int) -> dict[str, Any] | None:
    rows = _tsv(rd / "S3" / "markers.tsv.gz")
    if not rows:
        return None
    per: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        try:
            rank = int(float(r.get("rank") or 0))
        except ValueError:
            rank = 0
        lst = per.setdefault(str(r.get("cluster")), [])
        if len(lst) < top and (rank == 0 or rank <= top):
            lst.append({"gene": r.get("gene"), "avg_log2fc": _float(r.get("avg_log2fc")), "pct_in": _float(r.get("pct_in")),
                        "pct_out": _float(r.get("pct_out")), "p_adj": _float(r.get("p_adj"))})
    clusters = sorted(per, key=plots._level_key)
    return {"top": top, "clusters": [{"cluster": c, "genes": per[c]} for c in clusters], "n_rows": len(rows)}


def _dge(rd: Path) -> dict[str, Any] | None:
    s = _json(rd / "S4" / "dge_summary.json")
    if not s:
        return None
    contrasts = []
    for c in s.get("contrasts") or []:
        groups = []
        for g in c.get("groups") or []:
            groups.append({"group": g.get("group"), "file": g.get("file"), "method": g.get("method_used"),
                           "n_tested": g.get("n_tested", 0), "n_up": g.get("n_up", 0), "n_down": g.get("n_down", 0),
                           "n_samples": g.get("n_samples") or {}, "skipped": g.get("skipped_reason"),
                           "top_up": _gene_list(g.get("top_up")), "top_down": _gene_list(g.get("top_down")),
                           "notes": g.get("notes") or []})
        contrasts.append({"name": c.get("name"), "numerator": c.get("numerator"), "denominator": c.get("denominator"),
                          "covariates": c.get("covariates") or [], "samples": c.get("samples") or {}, "groups": groups,
                          "plot": plots.stacked_bars([(str(g["group"]), {"up": g["n_up"], "down": g["n_down"]}) for g in groups if not g["skipped"]],
                                                     ["up", "down"], title=f"{c.get('name')}: DE genes per group",
                                                     colours={"up": "#D55E00", "down": "#0072B2"})})
    return {"summary": s, "contrasts": contrasts, "method": s.get("method"), "group_by": s.get("group_by"),
            "thresholds": s.get("thresholds") or {}, "labels": s.get("labels"), "notes": s.get("notes") or []}


def _gene_list(v: Any) -> list[dict[str, Any]]:
    out = []
    for g in v or []:
        if isinstance(g, str):
            out.append({"gene": g})
        elif isinstance(g, Mapping):
            out.append({"gene": g.get("gene"), "log2fc": g.get("log2fc", g.get("log2FoldChange")), "padj": g.get("padj")})
    return out


def _opentargets(rd: Path) -> dict[str, Any] | None:
    s = _json(rd / "S4" / "opentargets_summary.json")
    if not s:
        return None
    contrasts = []
    targets = s.get("targets") or {}
    for c in s.get("contrasts") or []:
        groups = []
        for g in c.get("groups") or []:
            dirs = []
            for d, dd in (g.get("directions") or {}).items():
                genes = []
                for ge in (dd.get("genes") or [])[:50]:
                    # S4 lists each gene with `target` = its key in the summary's targets table (None without
                    # a record); a record inlined under `target` is accepted too
                    t = ge.get("target")
                    t = (targets.get(t) or {}) if isinstance(t, str) else (t if isinstance(t, Mapping) else ge)
                    genes.append({"gene": ge.get("gene") or ge.get("symbol"), "ensembl": t.get("ensembl_id") or ge.get("ensembl_id"),
                                  "status": ge.get("status") or ge.get("lookup_status"),
                                  "drugs": [f"{x.get('drug')} (ph {x.get('phase')}, {x.get('disease')})" for x in (t.get("known_drugs") or [])[:3]],
                                  "diseases": [f"{x.get('name')} ({_float(x.get('score')):.2f})" for x in (t.get("associated_diseases") or [])[:3]],
                                  "tractability": t.get("tractability")})
                dirs.append({"direction": d, "n_de": dd.get("n_de"), "n_listed": dd.get("n_listed"), "genes": genes})
            groups.append({"group": g.get("group"), "directions": dirs})
        contrasts.append({"name": c.get("name") or c.get("contrast"), "groups": groups})
    return {"summary": s, "contrasts": contrasts, "counts": s.get("counts") or {}, "offline": s.get("offline"),
            "errors": s.get("errors") or [], "without_record": s.get("genes_without_record") or []}


def _gate(gdir: Path, gate: str) -> dict[str, Any]:
    files = {k: _json(gdir / f"{k}.json") for k in ("bundle", "proposal", "review", "decision")}
    present = any(v is not None for v in files.values())
    dec = files["decision"] or {}
    rev = files["review"] or {}
    prop = files["proposal"] or {}
    pending = _json(gdir / "pending.json")  # written by the gates package while a human decision is awaited
    present = present or pending is not None
    return {
        "gate": gate, "title": stage_title(gate), "slug": stage_slug(gate), "present": present, "dir": str(gdir),
        "has_report": (gdir / "report.html").exists(), "report_href": f"../gates/{gdir.name}/report.html",
        "pending": pending is not None and not dec,
        "mode": dec.get("mode"), "decided_by": dec.get("decided_by"), "decided_utc": dec.get("decided_utc"),
        "changes": dec.get("changes_vs_proposal") or [],
        "verdict": rev.get("verdict"), "findings": rev.get("findings") or [], "reviewer": (rev.get("reviewer") or {}).get("model"),
        "worker": (prop.get("worker") or {}).get("model"), "rationale": prop.get("rationale"),
        "fallback": prop.get("fallback"), "policy_check": prop.get("policy_check") or {},
        "decision_json": json.dumps(dec.get("decision"), indent=1, ensure_ascii=False) if dec.get("decision") is not None else None,
        "proposal_json": json.dumps(prop.get("proposal"), indent=1, ensure_ascii=False) if prop.get("proposal") is not None else None,
        "bundle_json": json.dumps(files["bundle"], indent=1, ensure_ascii=False)[:20000] if files["bundle"] is not None else None,
        "files": {k: (v is not None) for k, v in files.items()},
    }


def _llm(rd: Path, cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    led = _json(rd / "llm" / "ledger.json")
    if not led:
        return None
    entries = led.get("entries") or []
    by_role: dict[str, dict[str, Any]] = {}
    for e in entries:
        k = f"{e.get('gate', '?')}/{e.get('role', '?')}"
        b = by_role.setdefault(k, {"key": k, "calls": 0, "input": 0, "output": 0, "cached": 0, "cost": 0.0, "model": e.get("model")})
        u = e.get("usage") or {}
        b["calls"] += 1
        b["input"] += int(u.get("input_tokens") or 0)
        b["output"] += int(u.get("output_tokens") or 0)
        b["cached"] += int(u.get("cached_tokens") or 0)
        b["cost"] += float(e.get("cost_gbp") or 0)
    budget = ((cfg.get("llm") or {}).get("budget") or {})
    return {"total_gbp": float(led.get("total_gbp") or 0), "n_calls": len(entries), "rows": sorted(by_role.values(), key=lambda r: r["key"]),
            "per_run_limit": budget.get("per_run_limit"), "total_limit": budget.get("total_limit"),
            "provider": (cfg.get("llm") or {}).get("provider"), "replay": sum(1 for e in entries if e.get("replay"))}


def _software(rd: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    pk: dict[str, Any] = {}
    for st in ("S2", "S4"):
        s = _json(rd / st / "summary.json") or _json(rd / st / "dge_summary.json") or {}
        for k, v in (s.get("packages") or {}).items():
            pk.setdefault(k, v)
    s3 = _json(rd / "S3" / "s3_r_summary.json") or {}
    for k, v in (s3.get("packages") or {}).items():
        pk.setdefault(k, v)
    import sys

    return {"r": pk, "python": sys.version.split()[0], "cellwarden": cfg.get("cellwarden_version")}


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


def _env():
    import jinja2

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES)), autoescape=jinja2.select_autoescape(["html", "j2"]),
                             trim_blocks=True, lstrip_blocks=True)
    env.filters["num"] = _num
    env.filters["pct"] = lambda v: "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):.1f}%"
    env.filters["sci"] = lambda v: "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):.2e}"
    env.filters["f2"] = lambda v: "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):.2f}"
    return env


def _num(v: Any) -> str:
    if v is None or v == "":
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(f):
        return "-"
    if f == int(f) and abs(f) < 1e15:
        return f"{int(f):,}"
    return f"{f:,.3g}" if abs(f) < 1 else f"{f:,.1f}"


def render_report(ctx: Mapping[str, Any]) -> str:
    return _env().get_template("report.html.j2").render(**{k: v for k, v in ctx.items() if not k.startswith("_")})


def build_report(run_dir: str | Path, out: str | Path | None = None, ctx: Mapping[str, Any] | None = None) -> Path:
    rd = Path(run_dir)
    ctx = ctx or collect(rd)
    out = Path(out) if out else rd / "S5" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".html.tmp")
    tmp.write_text(render_report(ctx), encoding="utf-8")
    tmp.replace(out)
    return out


def render_gate_report(gate_dir: str | Path, extra_sections: Sequence[tuple[str, str]] = ()) -> str:
    """One gate's report.html (bundle, proposal, review, decision) as a string.

    ``extra_sections`` are ``(title, html)`` pairs a gate module adds (its ``report_sections``),
    rendered before the footer.
    """
    gdir = Path(gate_dir)
    gate = gdir.name.upper()
    g = _gate(gdir, gate)
    rd = gdir.parent.parent
    cfg = _yaml(rd / "config" / "config.resolved.yaml")
    run = {"run_id": cfg.get("run_id") or rd.name, "project": cfg.get("project"), "run_dir": str(rd)}
    llm = _llm(rd, cfg)
    return _env().get_template("gate.html.j2").render(gate=g, run=run, llm=llm, generated_utc=_utc(),
                                                       extra_sections=list(extra_sections))


# --------------------------------------------------------------------------
# Explorer
# --------------------------------------------------------------------------

NUMERIC_COLS = ("n_genes", "n_umis", "pct_mito", "doublet_score", "nFeature_RNA", "nCount_RNA", "percent.mt")
SKIP_COLS = ("cell_id", "umap_1", "umap_2")


def explorer_payload(run_dir: str | Path, umap: list[dict[str, str]] | None = None, *, max_cells: int = 100000,
                     seed: int = 42) -> dict[str, Any]:
    rd = Path(run_dir)
    umap = umap if umap is not None else _tsv(rd / "S5" / "umap.tsv.gz")
    rows = [r for r in umap if not math.isnan(_float(r.get("umap_1"))) and not math.isnan(_float(r.get("umap_2")))]
    expr_rows = _tsv(rd / "S5" / "expr_subset.tsv.gz")
    expr_cells = {r["cell_id"] for r in expr_rows} if expr_rows else set()
    n = len(rows)
    if n > max_cells:
        rng = random.Random(seed)
        with_expr = [i for i, r in enumerate(rows) if r["cell_id"] in expr_cells]
        rest = [i for i, r in enumerate(rows) if r["cell_id"] not in expr_cells]
        keep = with_expr[:max_cells]
        if len(keep) < max_cells:
            keep += rng.sample(rest, max_cells - len(keep))
        keep.sort()
        rows = [rows[i] for i in keep]
    index = {r["cell_id"]: i for i, r in enumerate(rows)}
    cats: dict[str, Any] = {}
    nums: dict[str, list[float | None]] = {}
    cols = [c for c in (rows[0].keys() if rows else []) if c not in SKIP_COLS]
    for c in cols:
        vals = [r.get(c) for r in rows]
        if c in NUMERIC_COLS:
            nums[c] = [None if v in (None, "") else round(_float(v), 3) for v in vals]
            continue
        levels = sorted({v if v not in (None, "") else "NA" for v in vals}, key=plots._level_key)
        if len(levels) > 200:
            continue  # free text, not a colour option
        li = {lv: i for i, lv in enumerate(levels)}
        cats[c] = {"levels": levels, "idx": [li[v if v not in (None, "") else "NA"] for v in vals]}
    genes: dict[str, dict[str, list]] = {}
    for r in expr_rows:
        i = index.get(r.get("cell_id"))
        if i is None:
            continue
        g = genes.setdefault(r["gene"], {"i": [], "v": []})
        g["i"].append(i)
        g["v"].append(round(_float(r.get("value")), 3))
    return {
        "run_id": (_yaml(rd / "config" / "config.resolved.yaml") or {}).get("run_id") or rd.name,
        "n_total": n, "n": len(rows),
        "x": [round(_float(r["umap_1"]), 3) for r in rows],
        "y": [round(_float(r["umap_2"]), 3) for r in rows],
        "ids": [r["cell_id"] for r in rows],
        "cats": cats, "nums": nums, "genes": genes,
        "expr_cells": len(expr_cells & set(index)) if expr_cells else 0,
        "palette": plots.PALETTE, "na_colour": plots.NA_COLOUR,
    }


def build_explorer(run_dir: str | Path, out: str | Path | None = None, *, max_cells: int | None = None,
                   umap: list[dict[str, str]] | None = None) -> Path:
    rd = Path(run_dir)
    cfg = _yaml(rd / "config" / "config.resolved.yaml")
    max_cells = max_cells or int((cfg.get("report") or {}).get("explorer_max_cells") or 100000)
    payload = explorer_payload(rd, umap, max_cells=max_cells, seed=int(cfg.get("seed") or 42))
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = _env().get_template("explorer.html.j2").render(data_json=data, run_id=payload["run_id"], n=payload["n"],
                                                         n_total=payload["n_total"], n_genes=len(payload["genes"]),
                                                         expr_cells=payload["expr_cells"], generated_utc=_utc())
    out = Path(out) if out else rd / "S5" / "explorer.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".html.tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(out)
    return out


def build_all(run_dir: str | Path) -> dict[str, Path]:
    rd = Path(run_dir)
    ctx = collect(rd)
    r = build_report(rd, ctx=ctx)
    e = build_explorer(rd, umap=ctx.get("_umap_rows"))
    return {"report": r, "explorer": e}
