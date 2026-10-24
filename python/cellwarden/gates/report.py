"""Gate ``report.html``: one self-contained page per gate (no CDN, inline CSS).

Deliberately small: the S5 report developer replaces the templates later
(agents-spec section 7, "each gate's report.html reuses the same templates"),
so everything lives in :func:`render_report` and the gate-specific tables
come in as ready-made ``(title, html)`` sections from the gate object.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["render_report", "esc", "table", "pre"]

CSS = """
body{font:14px/1.45 system-ui,sans-serif;margin:2em auto;max-width:1100px;color:#222;padding:0 1em}
h1{font-size:1.5em;margin-bottom:.2em} h2{font-size:1.15em;margin-top:1.6em;border-bottom:1px solid #ddd}
table{border-collapse:collapse;margin:.5em 0} th,td{border:1px solid #ccc;padding:3px 8px;text-align:right}
th:first-child,td:first-child{text-align:left} th{background:#f3f3f3}
pre{background:#f7f7f7;padding:.6em;overflow:auto;font-size:12px}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;color:#fff;font-weight:600}
.wait{background:#c77700} .continue{background:#2a7d2a} .warn{color:#b35c00} .block{color:#b00020} .info{color:#444}
.cmd{background:#222;color:#eee;padding:.5em .8em;font-family:monospace}
"""


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def pre(obj: Any) -> str:
    return "<pre>" + esc(json.dumps(obj, indent=1, default=str) if not isinstance(obj, str) else obj) + "</pre>"


def table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> str:
    if not rows:
        return "<p><em>none</em></p>"
    cols = list(columns or rows[0].keys())
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{esc(_fmt(r.get(c)))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return f"{v:.4g}"
    return v


def render_report(gate_id: str, run_dir: Path, bundle: Mapping[str, Any], proposal: Mapping[str, Any],
                  review: Mapping[str, Any] | None, decision: Mapping[str, Any] | None, status: str,
                  ledger_status: Mapping[str, Any] | None, notes: Sequence[str],
                  sections: Sequence[tuple[str, str]] = ()) -> str:
    parts: list[str] = []
    badge = "waiting for a human" if status == "wait" else f"decided by {decision.get('decided_by') if decision else '?'}"
    parts.append(f"<h1>Gate {esc(gate_id)} <span class='badge {esc(status)}'>{esc(badge)}</span></h1>")
    parts.append(f"<p>run <code>{esc(run_dir)}</code>; mode <b>{esc((decision or {}).get('mode') or '')}</b>"
                 f"{'; rules fallback' if proposal.get('fallback') else ''}</p>")
    if status == "wait":
        parts.append("<h2>Action needed</h2><p>Review the proposal below, then approve it (optionally after editing "
                     "<code>gates/" + esc(gate_id) + "/proposal.json</code> and passing it with <code>--from-file</code>):</p>")
        parts.append(f"<p class='cmd'>cellwarden approve --run-dir {esc(run_dir)} --stage {esc(gate_id)} [--from-file edited.json] "
                     f"[--by NAME]<br>cellwarden resume --run-dir {esc(run_dir)}</p>")
    if notes:
        parts.append("<h2>Notes</h2><ul>" + "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul>")
    for title, body in sections:
        parts.append(f"<h2>{esc(title)}</h2>{body}")
    parts.append("<h2>Worker proposal</h2>")
    w = proposal.get("worker") or {}
    pc = proposal.get("policy_check") or {}
    parts.append(f"<p>role <b>{esc(w.get('role'))}</b>, model <b>{esc(w.get('model') or 'rules')}</b>; policy check: "
                 f"<b>{'ok' if pc.get('ok') else 'FAILED'}</b></p>")
    if pc.get("violations"):
        parts.append("<ul>" + "".join(f"<li class='block'>{esc(v)}</li>" for v in pc["violations"]) + "</ul>")
    parts.append("<p><b>Rationale</b></p>" + pre(proposal.get("rationale") or ""))
    parts.append("<details><summary>proposal.json</summary>" + pre(proposal.get("proposal")) + "</details>")
    parts.append("<h2>Reviewer</h2>")
    if review:
        parts.append(f"<p>verdict <b>{esc(review.get('verdict'))}</b> (model {esc((review.get('reviewer') or {}).get('model'))})</p>")
        f = review.get("findings") or []
        parts.append("<ul>" + "".join(f"<li class='{esc(x.get('severity'))}'>[{esc(x.get('severity'))}] {esc(x.get('text'))}</li>" for x in f) + "</ul>" if f else "<p><em>no findings</em></p>")
    else:
        parts.append("<p><em>no review</em></p>")
    parts.append("<h2>Decision</h2>")
    if decision:
        parts.append(f"<p>by <b>{esc(decision.get('decided_by'))}</b> at {esc(decision.get('decided_utc'))}</p>")
        ch = decision.get("changes_vs_proposal") or []
        parts.append("<p>changes vs proposal:</p><ul>" + "".join(f"<li>{esc(c)}</li>" for c in ch) + "</ul>" if ch else "<p>no changes vs the proposal</p>")
        parts.append("<details><summary>decision.json</summary>" + pre(decision.get("decision")) + "</details>")
    else:
        parts.append("<p><em>pending</em></p>")
    parts.append("<h2>LLM cost</h2>")
    if ledger_status:
        parts.append(f"<p>this run {ledger_status.get('run_gbp', 0):.4f} / {ledger_status.get('per_run_limit')} GBP "
                     f"({ledger_status.get('run_pct')}%); all runs {ledger_status.get('total_gbp', 0):.4f} / "
                     f"{ledger_status.get('total_limit')} GBP ({ledger_status.get('total_pct')}%); "
                     f"{ledger_status.get('n_calls')} call(s) in this run</p>")
        for wmsg in ledger_status.get("warnings") or []:
            parts.append(f"<p class='warn'>{esc(wmsg)}</p>")
    else:
        parts.append("<p>no LLM calls (agents not configured)</p>")
    parts.append("<details><summary>bundle.json</summary>" + pre(bundle) + "</details>")
    return ("<!doctype html><html><head><meta charset='utf-8'><title>" + esc(f"CellWarden gate {gate_id}")
            + "</title><style>" + CSS + "</style></head><body>" + "\n".join(parts) + "</body></html>\n")
