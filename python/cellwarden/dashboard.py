"""Run dashboard: one self-contained HTML page showing every stage and gate of a run.

    cellwarden dashboard --run-dir DIR [--out DIR/dashboard.html] [--watch SECONDS]

Reads only what the run already writes (``state.json``, the stage summaries, the
gate ``decision``/``review``/``proposal`` files, ``llm/ledger.json`` and the stage
logs) and renders:

* the pipeline as a strip of stages and gates coloured by status;
* a timeline (one bar per stage, wall-clock, gaps = waiting for a human);
* one card per stage with wall time, peak memory, exit code and the key numbers
  from its summary (cells, QC counts, engine/path, clusters, contrasts, ...);
* one card per gate with mode, who decided, fallback, reviewer verdict, cost,
  changes versus the proposal, and a link to the gate's own ``report.html``;
* the tail of the log for a running or failed stage.

With ``--watch N`` the page is regenerated every N seconds and refreshes itself
in the browser until the run finishes or fails. ``cellwarden run`` also rewrites
the page after every stage, so opening ``<run_dir>/dashboard.html`` is enough.
No CDN, no JavaScript needed; inline CSS only, so it opens over ``scp``/SMB.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .stage_names import title as stage_title

DEFAULT_ORDER = ["S0", "S1a", "S1b", "S1c", "G1", "S2", "S3", "G2", "G3", "S4", "G4", "S5"]
COLORS = {"done": "#2e8b57", "running": "#1f77b4", "waiting": "#e6a700", "failed": "#c0392b",
          "pending": "#b8b8b8", "skipped": "#8c8c8c"}


def _json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _yaml(p: Path) -> dict[str, Any]:
    try:
        import yaml
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def _ts(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _fmt_s(v: Any) -> str:
    if v is None or v == "":
        return "–"
    v = float(v)
    if v < 60:
        return f"{v:.1f} s"
    if v < 3600:
        return f"{v / 60:.1f} min"
    return f"{v / 3600:.2f} h"


def _fmt_b(b: Any) -> str:
    if not b:
        return "–"
    b = float(b)
    return f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.0f} MB"


def _esc(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _tail(p: Path, n: int = 15) -> str:
    try:
        lines = p.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


# ----------------------------------------------------------------------------- collect

def collect(run_dir: Path) -> dict[str, Any]:
    rd = Path(run_dir)
    state = _json(rd / "state.json") or {}
    cfg = _yaml(rd / "config" / "config.resolved.yaml")
    order = state.get("stage_order") or DEFAULT_ORDER
    stages = state.get("stages") or {}
    rows = []
    t0 = None
    for sid in order:
        rec = stages.get(sid) or {"status": "pending"}
        st = rec.get("status", "pending")
        start, end = _ts(rec.get("started_utc")), _ts(rec.get("ended_utc"))
        if st == "running" and start and not end:
            end = time.time()
        if start and (t0 is None or start < t0):
            t0 = start
        row = {
            "id": sid, "title": stage_title(sid), "gate": sid.startswith("G"), "status": st,
            "lang": rec.get("lang", ""), "wall": rec.get("wall_s"), "rss": rec.get("peak_rss_bytes"),
            "exit": rec.get("exit_code"), "note": rec.get("note") or rec.get("skip_reason") or "",
            "start": start, "end": end, "started_utc": rec.get("started_utc"), "ended_utc": rec.get("ended_utc"),
            "facts": [], "log": "", "report": None,
        }
        _stage_facts(rd, row)
        if st in ("running", "failed", "waiting"):
            row["log"] = _tail(rd / "logs" / f"{sid}.log")
        rows.append(row)
    overall = _overall(rows)
    total = sum(float(r["wall"]) for r in rows if r.get("wall"))
    ledger = _json(rd / "llm" / "ledger.json") or {}
    spent = sum(float(e.get("cost_gbp") or e.get("cost") or 0) for e in ledger.get("entries") or [])
    s2 = _json(rd / "S2" / "summary.json") or {}
    return {
        "run_dir": str(rd), "run_id": state.get("run_id", "?"), "project": cfg.get("project", ""),
        "updated": state.get("updated_utc", ""), "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "overall": overall, "rows": rows, "t0": t0, "total_wall": total,
        "max_rss": max((float(r["rss"]) for r in rows if r.get("rss")), default=0),
        "engine": s2.get("engine") or cfg.get("engine", ""), "path": s2.get("path", ""),
        "device": s2.get("device") or cfg.get("device", ""), "llm_spent": spent, "llm_calls": len(ledger.get("entries") or []),
        "report": (rd / "S5" / "report.html").exists(), "explorer": (rd / "S5" / "explorer.html").exists(),
    }


def _overall(rows: list[dict[str, Any]]) -> str:
    sts = [r["status"] for r in rows]
    if "failed" in sts:
        return "failed"
    if "running" in sts:
        return "running"
    if "waiting" in sts:
        return "waiting"
    if all(s in ("done", "skipped") for s in sts) and sts:
        return "done"
    return "running" if any(s == "done" for s in sts) else "pending"


def _stage_facts(rd: Path, row: dict[str, Any]) -> None:
    sid, f = row["id"], row["facts"]
    if sid == "S0":
        j = _json(rd / "S0" / "ingest_report.json") or {}
        if j:
            t = j.get("totals") or {}
            f += [("samples", t.get("samples", len(j.get("samples") or []))), ("cells", t.get("cells")), ("genes", t.get("genes")),
                  ("gene key", j.get("gene_key"))]
    elif sid in ("S1a", "S1b", "S1c"):
        j = _json(rd / "S1" / "qc_summary.json") or {}
        if sid == "S1c" and j:
            c = j.get("counts") or {}
            f += [("cells", j.get("n_cells")), ("keep", c.get("keep")), ("review", c.get("review")),
                  ("exclude", c.get("exclude")), ("decided by", j.get("decided_by"))]
        elif sid == "S1a":
            log = _tail(rd / "logs" / "S1a.log", 40)
            for line in log.splitlines():
                if "engine" in line and "S1a:" in line:
                    f.append(("engine", line.split("S1a:")[-1].strip()))
                    break
    elif sid == "S2":
        j = _json(rd / "S2" / "summary.json") or {}
        if j:
            f += [("path", j.get("path")), ("engine", f"{j.get('engine')} {j.get('scfast_version') or ''}".strip()),
                  ("device / plan", f"{j.get('device')} / {j.get('plan')}"), ("cells in → out", f"{j.get('cells_in')} → {j.get('cells_out')}"),
                  ("HVG / PCs", f"{j.get('n_hvg')} / {j.get('pcs')}"), ("clusters", j.get("clusters")),
                  ("integration", (j.get("integration") or {}).get("method"))]
            t = j.get("timings_s") or {}
            if t:
                row["timings"] = t
    elif sid == "S3":
        j = _json(rd / "S3" / "cluster_summary.json") or {}
        cons = rd / "S3" / "annotation_consensus.tsv"
        if j:
            f.append(("clusters", len(j)))
        if cons.exists():
            lines = cons.read_text().splitlines()[1:]
            low = sum(1 for ln in lines if "low_confidence" in ln or "conflict" in ln)
            f += [("annotated clusters", len(lines)), ("low-confidence / conflict", low)]
        if (rd / "S3" / "celltypist_note.json").exists():
            n = _json(rd / "S3" / "celltypist_note.json") or {}
            if n.get("note"):
                f.append(("CellTypist", n["note"][:80]))
    elif sid == "S4":
        j = _json(rd / "S4" / "dge_summary.json") or {}
        if j:
            cs = j.get("contrasts") or []
            m = j.get("method")
            if isinstance(m, dict):
                m = f"{m.get('requested')} (used: {', '.join(m.get('used') or []) or 'none'})"
            f += [("contrasts", len(cs)), ("method", m), ("groups × samples", f"{j.get('n_groups')} × {j.get('n_samples')}")]
            for c in cs[:4]:
                f.append((c.get("name", "contrast"), f"{c.get('n_up', '?')} up / {c.get('n_down', '?')} down"))
        o = _json(rd / "S4" / "opentargets_summary.json") or {}
        if o:
            f.append(("Open Targets", ("offline cache" if o.get("offline") else "live") if o.get("enabled") else "disabled"))
    elif sid == "S5":
        if (rd / "S5" / "report.html").exists():
            row["report"] = "S5/report.html"
            f.append(("outputs", "report.html, explorer.html, umap.tsv.gz"))
    elif row["gate"]:
        g = rd / "gates" / sid
        dec = _json(g / "decision.json") or {}
        rev = _json(g / "review.json") or {}
        prop = _json(g / "proposal.json") or {}
        pend = _json(g / "pending.json")
        if (g / "report.html").exists():
            row["report"] = f"gates/{sid}/report.html"
        if pend and not dec:
            f.append(("state", "waiting for `cellwarden approve`"))
        if dec:
            f += [("mode", dec.get("mode")), ("decided by", dec.get("decided_by")),
                  ("fallback (rules only)", "yes" if dec.get("fallback") else "no"),
                  ("changes vs proposal", len(dec.get("changes_vs_proposal") or []))]
        if rev:
            f.append(("reviewer verdict", rev.get("verdict")))
        if prop:
            w = prop.get("worker") or {}
            if w.get("model"):
                f.append(("worker model", w["model"]))
        if sid == "G1" and dec.get("decision"):
            d = dec["decision"]
            if isinstance(d, dict) and d.get("thresholds"):
                f.append(("thresholds changed", len(d["thresholds"])))
        if sid == "G2" and isinstance(dec.get("decision"), dict):
            t = (dec["decision"].get("tissue") or {})
            f.append(("tissue", f"{t.get('label', '?')} ({t.get('uberon_id', '')})"))
        if sid == "G3" and isinstance(dec.get("decision"), dict):
            cl = dec["decision"].get("clusters") or []
            f.append(("clusters labelled", f"{len(cl)}; {sum(1 for c in cl if c.get('status') != 'consensus')} changed from consensus"))


# ----------------------------------------------------------------------------- render

CSS = """
body{font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#222}
header{background:#1b2a41;color:#fff;padding:14px 22px}header h1{margin:0;font-size:18px;font-weight:600}
header .meta{opacity:.85;font-size:12.5px;margin-top:4px}
.badge{display:inline-block;padding:2px 9px;border-radius:10px;color:#fff;font-size:12px;font-weight:600;vertical-align:middle}
main{padding:16px 22px;max-width:1400px;margin:0 auto}
.strip{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:8px 0 18px}
.node{padding:7px 11px;border-radius:7px;color:#fff;font-weight:600;font-size:12.5px;min-width:68px;text-align:center;text-decoration:none}
.node.gate{border-radius:20px;outline:2px dashed rgba(0,0,0,.25);outline-offset:-3px}
.node small{display:block;font-weight:400;opacity:.9;font-size:11px}
.arrow{color:#999}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px}
.kpi{background:#fff;border:1px solid #e3e5e8;border-radius:8px;padding:8px 14px;min-width:120px}
.kpi b{display:block;font-size:18px}.kpi span{font-size:11.5px;color:#666}
h2{font-size:15px;margin:22px 0 8px}
.timeline{background:#fff;border:1px solid #e3e5e8;border-radius:8px;padding:10px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.card{background:#fff;border:1px solid #e3e5e8;border-radius:8px;padding:10px 14px;border-top:4px solid #ccc}
.card h3{margin:0 0 4px;font-size:14px}.card h3 .badge{float:right}
.card table{width:100%;border-collapse:collapse;font-size:12.5px}.card td{padding:2px 0;vertical-align:top}
.card td:first-child{color:#666;width:48%}
.card pre{background:#f1f3f5;font-size:11px;padding:6px;overflow:auto;max-height:180px;margin:6px 0 0;border-radius:4px}
.card a{color:#1f5fa8}
.bar{display:flex;height:10px;border-radius:4px;overflow:hidden;margin-top:4px}.bar div{height:100%}
.legend{font-size:11px;color:#666;margin-top:3px}
footer{color:#777;font-size:11.5px;padding:18px 22px}
"""


def render(d: dict[str, Any], watch: int | None) -> str:
    o = d["overall"]
    refresh = f'<meta http-equiv="refresh" content="{watch}">' if watch and o in ("running", "waiting", "pending") else ""
    parts = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>CellWarden run {_esc(d['run_id'])}</title>{refresh}<style>{CSS}</style></head><body>"]
    parts.append(f"<header><h1>CellWarden run dashboard <span class='badge' style='background:{COLORS.get(o, '#888')}'>{_esc(o.upper())}</span></h1>"
                 f"<div class='meta'>run <code>{_esc(d['run_id'])}</code> · project {_esc(d['project'])} · {_esc(d['run_dir'])}<br>"
                 f"state updated {_esc(d['updated'])} · page generated {_esc(d['generated'])}{' · auto-refresh every %ds' % watch if refresh else ''}</div></header><main>")
    # strip
    parts.append("<div class='strip'>")
    for i, r in enumerate(d["rows"]):
        if i:
            parts.append("<span class='arrow'>→</span>")
        sub = _fmt_s(r["wall"]) if r["wall"] else r["status"]
        parts.append(f"<a class='node{' gate' if r['gate'] else ''}' href='#{r['id']}' style='background:{COLORS.get(r['status'], '#888')}' title='{_esc(r['title'])}: {_esc(r['status'])}'>{r['id']}<small>{_esc(sub)}</small></a>")
    parts.append("</div>")
    # kpis
    done = sum(1 for r in d["rows"] if r["status"] == "done")
    kp = [(f"{done}/{len(d['rows'])}", "stages done"), (_fmt_s(d["total_wall"]), "compute wall time"),
          (_fmt_b(d["max_rss"]), "max peak RSS"), (f"{_esc(d['engine'])} / {_esc(d['device'])}", "engine / device"),
          (_esc(d["path"] or "–"), "S2 path"), (f"£{d['llm_spent']:.2f} ({d['llm_calls']} calls)", "LLM spend")]
    parts.append("<div class='kpis'>" + "".join(f"<div class='kpi'><b>{a}</b><span>{b}</span></div>" for a, b in kp) + "</div>")
    if d["report"]:
        parts.append("<p><a href='S5/report.html'>Open report.html</a> · <a href='S5/explorer.html'>explorer.html</a></p>")
    # timeline
    parts.append("<h2>Timeline</h2><div class='timeline'>" + _timeline_svg(d) + "</div>")
    # cards
    parts.append("<h2>Stages and gates</h2><div class='cards'>")
    for r in d["rows"]:
        parts.append(_card(r))
    parts.append("</div></main>")
    parts.append("<footer>Colours: " + " ".join(f"<span class='badge' style='background:{c}'>{k}</span>" for k, c in COLORS.items())
                 + " · Regenerate with <code>cellwarden dashboard --run-dir …</code>; <code>--watch 30</code> keeps it live.</footer></body></html>")
    return "".join(parts)


def _timeline_svg(d: dict[str, Any]) -> str:
    rows = [r for r in d["rows"] if r["start"]]
    if not rows or d["t0"] is None:
        return "<p style='color:#777'>No stage has started yet.</p>"
    t0 = d["t0"]
    t1 = max(r["end"] or r["start"] for r in rows)
    span = max(t1 - t0, 1.0)
    W, L, H, rh = 1000, 60, 22, 18
    n = len(d["rows"])
    svg = [f"<svg viewBox='0 0 {W + 120} {n * H + 30}' width='100%' style='font-size:11px'>"]
    for k in range(6):
        x = L + (W - L) * k / 5
        svg.append(f"<line x1='{x:.0f}' y1='0' x2='{x:.0f}' y2='{n * H}' stroke='#eee'/><text x='{x:.0f}' y='{n * H + 14}' fill='#777' text-anchor='middle'>{_fmt_s(span * k / 5)}</text>")
    for i, r in enumerate(d["rows"]):
        y = i * H
        svg.append(f"<text x='{L - 6}' y='{y + 13}' text-anchor='end' fill='#333'>{r['id']}</text>")
        if r["start"]:
            x0 = L + (W - L) * (r["start"] - t0) / span
            x1 = L + (W - L) * ((r["end"] or r["start"]) - t0) / span
            w = max(x1 - x0, 2)
            svg.append(f"<rect x='{x0:.1f}' y='{y + 2}' width='{w:.1f}' height='{rh}' rx='3' fill='{COLORS.get(r['status'], '#888')}'><title>{_esc(r['title'])}: {_esc(r['status'])}, {_fmt_s(r['wall'])}</title></rect>")
            svg.append(f"<text x='{x0 + w + 4:.1f}' y='{y + 14}' fill='#555'>{_fmt_s(r['wall']) if r['wall'] else r['status']}</text>")
        # gap before this stage = waiting (gate) or idle
        if i and r["start"] and d["rows"][i - 1]["end"] and r["start"] - d["rows"][i - 1]["end"] > 5:
            gx0 = L + (W - L) * (d["rows"][i - 1]["end"] - t0) / span
            gx1 = L + (W - L) * (r["start"] - t0) / span
            svg.append(f"<rect x='{gx0:.1f}' y='{y + 2}' width='{max(gx1 - gx0, 1):.1f}' height='{rh}' fill='url(#h)' opacity='.6'><title>waiting {_fmt_s(r['start'] - d['rows'][i - 1]['end'])} (human decision / idle)</title></rect>")
    svg.insert(1, "<defs><pattern id='h' width='6' height='6' patternUnits='userSpaceOnUse' patternTransform='rotate(45)'><rect width='3' height='6' fill='#e6a700'/></pattern></defs>")
    svg.append("</svg><div class='legend'>Hatched = gap between stages (a gate waiting for a human, or the run was resumed later).</div>")
    return "".join(svg)


def _card(r: dict[str, Any]) -> str:
    c = COLORS.get(r["status"], "#888")
    h = [f"<div class='card' id='{r['id']}' style='border-top-color:{c}'><h3>{r['id']} · {_esc(r['title'])}<span class='badge' style='background:{c}'>{_esc(r['status'])}</span></h3><table>"]
    h.append(f"<tr><td>wall / peak RSS</td><td>{_fmt_s(r['wall'])} / {_fmt_b(r['rss'])}</td></tr>")
    if r["started_utc"]:
        h.append(f"<tr><td>started → ended</td><td>{_esc(r['started_utc'][11:19])} → {_esc((r['ended_utc'] or '…')[11:19])} UTC</td></tr>")
    if r["exit"] not in (None, 0):
        h.append(f"<tr><td>exit code</td><td><b style='color:{COLORS['failed']}'>{r['exit']}</b></td></tr>")
    for k, v in r["facts"]:
        h.append(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>")
    if r["note"]:
        h.append(f"<tr><td>note</td><td>{_esc(r['note'])}</td></tr>")
    h.append("</table>")
    if r.get("timings"):
        t = r["timings"]
        tot = sum(float(v) for v in t.values()) or 1
        pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        h.append("<div class='bar'>" + "".join(f"<div style='width:{100 * float(v) / tot:.1f}%;background:{pal[i % 10]}' title='{_esc(k)} {_fmt_s(v)}'></div>" for i, (k, v) in enumerate(t.items())) + "</div>")
        h.append("<div class='legend'>" + " · ".join(f"{_esc(k)} {_fmt_s(v)}" for k, v in t.items()) + "</div>")
    if r["report"]:
        h.append(f"<p><a href='{_esc(r['report'])}'>open {_esc(r['report'].split('/')[-1] if not r['gate'] else 'gate report')}</a></p>")
    if r["log"]:
        h.append(f"<pre>{_esc(r['log'])}</pre>")
    h.append("</div>")
    return "".join(h)


def build(run_dir: str | Path, out: str | Path | None = None, watch: int | None = None) -> Path:
    rd = Path(run_dir).expanduser().resolve()
    d = collect(rd)
    p = Path(out) if out else rd / "dashboard.html"
    tmp = p.with_suffix(".html.tmp")
    tmp.write_text(render(d, watch))
    tmp.replace(p)
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="render <run_dir>/dashboard.html for a cellwarden run")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None, help="default: <run_dir>/dashboard.html")
    ap.add_argument("--watch", type=int, default=None, metavar="SECONDS",
                    help="regenerate every N seconds until the run is done or failed")
    a = ap.parse_args(argv)
    while True:
        p = build(a.run_dir, a.out, a.watch)
        d = collect(Path(a.run_dir))
        print(f"{d['overall']:8s} {p}", file=sys.stderr)
        if not a.watch or d["overall"] in ("done", "failed"):
            return 0 if d["overall"] != "failed" else 1
        time.sleep(a.watch)


if __name__ == "__main__":
    sys.exit(main())
