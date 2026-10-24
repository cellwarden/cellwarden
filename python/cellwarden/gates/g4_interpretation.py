"""Gate G4 (DGE interpretation): bundle builder, rules baseline, policy check.

Spec: docs/agents-spec.md sections 2 and 6; DESIGN.md 4.4 and 6.2.

Inputs read from ``<run-dir>/S4``:

* ``dge_summary.json`` (spec 6: per contrast x group n_up, n_down, n_tested,
  skipped_reason, top 10 up/down). Accepted layout::

      {"method": "deseq2", "padj": 0.05, "lfc": 0.5, "group_by": "cell_type_broad",
       "contrasts": {"<contrast>": {"numerator", "denominator", "covariates",
                     "groups": {"<group>": {"n_up", "n_down", "n_tested", "skipped_reason",
                                            "top_up": [{gene, log2fc, padj}], "top_down": [...]}}}}}

  ``contrasts`` may also be a list of ``{"name": ..., "groups": ...}`` and
  ``groups`` a list of ``{"group": ..., ...}``.
* ``dge/<contrast>__<group>.tsv.gz`` (optional): the full table
  ``gene, baseMean, log2fc, lfcSE, pvalue, padj, direction``; when present it
  extends the gene/direction map beyond the top-10 lists.
* ``opentargets_summary.json`` (spec 6: per contrast x group x direction,
  per-gene evidence). Accepted layout::

      {"contrasts": {"<contrast>": {"<group>": {"up"|"down": {"n_genes", "n_with_evidence",
          "genes": [{"gene", "ensembl_id", "direction", "log2fc",
                     "evidence": [{"evidence_id", "type", "label", ...}]}]}}}}}

  Every evidence record carries a unique ``evidence_id``; the policy check
  collects them with a recursive walk, so nesting differences do not matter.

G4 proposal shape (spec 6)::

    {"contrasts": [{"contrast", "group", "summary": str,
                    "hypotheses": [{"gene", "direction", "evidence_ids": [...], "text"}]}]}

Policy: every cited ``evidence_id`` exists in ``opentargets_summary.json``;
every cited gene is in the DGE table for that contrast/group with the stated
direction (``up``|``down``); every hypothesis text contains the word
"hypothesis" (texts without it are prefixed with ``Hypothesis:`` by
:func:`normalize_proposal` before checking). Skipped groups may not be
interpreted.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

GATE = "G4"
WORKER_ROLE = "interpretation_worker"
DIRECTIONS = ("up", "down")
HYPOTHESIS_RE = re.compile(r"hypothes[ie]s", re.IGNORECASE)
HYPOTHESIS_PREFIX = "Hypothesis: "


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def read_dge_summary(path: str | Path) -> dict[str, Any]:
    """Normalise ``dge_summary.json`` to ``{"contrasts": {name: {"groups": {group: {...}}}}}``."""
    raw = _load_json(Path(path))
    contrasts = raw.get("contrasts", {})
    if isinstance(contrasts, list):
        contrasts = {str(c.get("name", i)): c for i, c in enumerate(contrasts)}
    out_c: dict[str, Any] = {}
    for name, c in contrasts.items():
        c = dict(c)
        groups = c.get("groups", {})
        if isinstance(groups, list):
            groups = {str(g.get("group", i)): g for i, g in enumerate(groups)}
        c["groups"] = {str(g): dict(v) for g, v in groups.items()}
        out_c[str(name)] = c
    out = dict(raw)
    out["contrasts"] = out_c
    return out


def read_opentargets_summary(path: str | Path) -> dict[str, Any]:
    raw = _load_json(Path(path))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    out = dict(raw)
    if isinstance(raw.get("contrasts"), list):
        out["contrasts"] = _by_contrast(raw)
    return out


# Evidence kept per gene for the agent (the full records stay in S4/opentargets_summary.json).
_EVIDENCE_GENES = 20      # per contrast x group x direction (the top-ranked DE genes)
_EVIDENCE_DISEASES = 5
_EVIDENCE_DRUGS = 3


def _by_contrast(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The stage's summary (``contrasts: [{contrast, groups: [{group, directions}]}]`` plus ``targets``)
    as ``{contrast: {group: {direction: {n_genes, n_with_evidence, genes: [{gene, ..., evidence}]}}}}``."""
    targets = raw.get("targets") or {}
    out: dict[str, Any] = {}
    for c in raw.get("contrasts") or []:
        cd = out.setdefault(str(c.get("contrast")), {})
        for g in c.get("groups") or []:
            gd = cd.setdefault(str(g.get("group")), {})
            for dirn, blk in (g.get("directions") or {}).items():
                genes = []
                for x in (blk.get("genes") or [])[:_EVIDENCE_GENES]:
                    t = targets.get(x.get("target") or "") or {}
                    ev = [{"evidence_id": d.get("evidence_id"), "type": "associated_disease", "label": d.get("name"),
                           "efo_id": d.get("efo_id"), "score": d.get("score")}
                          for d in (t.get("associated_diseases") or [])[:_EVIDENCE_DISEASES]]
                    ev += [{"evidence_id": k.get("evidence_id"), "type": "known_drug", "label": k.get("drug"),
                            "phase": k.get("phase"), "disease": k.get("disease"), "mechanism": k.get("mechanism")}
                           for k in (t.get("known_drugs") or [])[:_EVIDENCE_DRUGS]]
                    genes.append({"gene": x.get("gene"), "symbol": x.get("symbol"), "ensembl_id": x.get("ensembl_id"),
                                  "direction": dirn, "log2fc": x.get("log2fc"), "padj": x.get("padj"), "evidence": ev})
                gd[str(dirn)] = {"n_genes": blk.get("n_de", len(genes)),
                                 "n_with_evidence": sum(1 for x in genes if x["evidence"]), "genes": genes}
    return out


def collect_evidence_ids(node: Any, out: dict[str, dict] | None = None) -> dict[str, dict]:
    """All ``evidence_id`` -> record found anywhere in the Open Targets summary."""
    out = {} if out is None else out
    if isinstance(node, Mapping):
        eid = node.get("evidence_id")
        if isinstance(eid, str):
            out[eid] = dict(node)
        for v in node.values():
            collect_evidence_ids(v, out)
    elif isinstance(node, list):
        for v in node:
            collect_evidence_ids(v, out)
    return out


def _read_full_table(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:  # type: ignore[arg-type]
        return list(csv.DictReader(fh, delimiter="\t"))


def gene_directions(dge: Mapping[str, Any], run_dir: Path | None = None) -> dict[str, dict[str, dict[str, str]]]:
    """``{contrast: {group: {gene: direction}}}`` from the top lists plus any full DGE tables."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for cname, c in dge["contrasts"].items():
        out[cname] = {}
        for gname, g in c["groups"].items():
            d: dict[str, str] = {}
            for rec in g.get("top_up") or []:
                d[str(rec["gene"])] = "up"
            for rec in g.get("top_down") or []:
                d[str(rec["gene"])] = "down"
            if run_dir is not None:
                for suffix in (".tsv.gz", ".tsv"):
                    p = run_dir / "S4" / "dge" / f"{cname}__{gname}{suffix}"
                    if p.exists():
                        for row in _read_full_table(p):
                            dirn = (row.get("direction") or "ns").strip()
                            if dirn in DIRECTIONS:
                                d[row["gene"]] = dirn
                        break
            out[cname][gname] = d
    return out


def _ot_genes(ot: Mapping[str, Any], contrast: str, group: str, direction: str) -> list[dict]:
    block = ((ot.get("contrasts") or {}).get(contrast) or {}).get(group) or {}
    return list((block.get(direction) or {}).get("genes") or [])


def build_bundle(run_dir: str | Path, cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``gates/G4/bundle.json``: DGE summary + Open Targets evidence per contrast x group."""
    run_dir = Path(run_dir)
    dge = read_dge_summary(run_dir / "S4" / "dge_summary.json")
    ot_path = run_dir / "S4" / "opentargets_summary.json"
    ot = read_opentargets_summary(ot_path) if ot_path.exists() else {"contrasts": {}}
    dirs = gene_directions(dge, run_dir)
    contrasts = []
    for cname, c in dge["contrasts"].items():
        groups = []
        for gname, g in c["groups"].items():
            evidence = {}
            for dirn in DIRECTIONS:
                evidence[dirn] = [
                    {"gene": rec.get("gene"), "log2fc": rec.get("log2fc"),
                     "evidence": [{k: e.get(k) for k in ("evidence_id", "type", "label", "phase", "disease", "mechanism", "efo_id", "score") if k in e}
                                  for e in (rec.get("evidence") or [])]}
                    for rec in _ot_genes(ot, cname, gname, dirn)
                ]
            groups.append({
                "group": gname,
                "n_up": g.get("n_up", 0), "n_down": g.get("n_down", 0), "n_tested": g.get("n_tested", 0),
                "skipped_reason": g.get("skipped_reason"),
                "top_up": list(g.get("top_up") or [])[:10], "top_down": list(g.get("top_down") or [])[:10],
                "n_genes_with_direction": len(dirs[cname][gname]),
                "opentargets": evidence,
            })
        contrasts.append({"contrast": cname, "numerator": c.get("numerator"), "denominator": c.get("denominator"),
                          "covariates": list(c.get("covariates") or []), "groups": groups})
    # the IDs the agent can cite: those of the evidence in the bundle (not every drug row of every target)
    ids = collect_evidence_ids(contrasts)
    return {
        "gate": GATE,
        "method": dge.get("method"), "padj": dge.get("padj"), "lfc": dge.get("lfc"),
        "contrasts": contrasts,
        "gene_directions": dirs,
        "evidence_ids": sorted(ids),
        "n_evidence": len(ids),
        "opentargets_offline": bool(ot.get("offline", False)),
    }


def baseline(bundle: Mapping[str, Any], cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Rules-only decision: counts per contrast x group, no hypotheses."""
    rows = []
    for c in bundle["contrasts"]:
        for g in c["groups"]:
            if g.get("skipped_reason"):
                summary = f"Skipped: {g['skipped_reason']}."
            else:
                n_ev = sum(1 for d in DIRECTIONS for rec in g["opentargets"].get(d, []) if rec.get("evidence"))
                summary = (f"{g['n_up']} up, {g['n_down']} down of {g['n_tested']} tested; "
                           f"{n_ev} DE genes with Open Targets evidence. No interpretation (rules baseline).")
            rows.append({"contrast": c["contrast"], "group": g["group"], "summary": summary, "hypotheses": []})
    return {"contrasts": rows}


def normalize_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy where every hypothesis text carries the word 'hypothesis' (prefix added if missing)."""
    out = json.loads(json.dumps(proposal, default=str))
    for c in out.get("contrasts", []) if isinstance(out.get("contrasts"), list) else []:
        if not isinstance(c, Mapping):
            continue
        for h in c.get("hypotheses", []) if isinstance(c.get("hypotheses"), list) else []:
            if isinstance(h, Mapping) and isinstance(h.get("text"), str) and h["text"].strip() and not HYPOTHESIS_RE.search(h["text"]):
                h["text"] = HYPOTHESIS_PREFIX + h["text"].strip()
    return out


def policy_check(proposal: Mapping[str, Any], bundle: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Validate a G4 proposal in code. Returns ``(ok, violations)``; run :func:`normalize_proposal` first."""
    v: list[str] = []
    if not isinstance(proposal, Mapping) or not isinstance(proposal.get("contrasts"), list):
        return False, ["proposal must be {contrasts: [...]}"]
    known_ids = set(bundle.get("evidence_ids") or [])
    dirs = bundle.get("gene_directions") or {}
    skipped = {(c["contrast"], g["group"]): g.get("skipped_reason") for c in bundle["contrasts"] for g in c["groups"]}
    seen: set[tuple[str, str]] = set()
    for i, c in enumerate(proposal["contrasts"]):
        if not isinstance(c, Mapping):
            v.append(f"contrasts[{i}] must be an object")
            continue
        key = (str(c.get("contrast")), str(c.get("group")))
        if key not in skipped:
            v.append(f"contrasts[{i}]: unknown contrast/group {key[0]!r}/{key[1]!r}")
            continue
        if key in seen:
            v.append(f"contrast/group {key[0]}/{key[1]} listed more than once")
        seen.add(key)
        if not isinstance(c.get("summary"), str) or not c["summary"].strip():
            v.append(f"{key[0]}/{key[1]}: summary must be non-empty text")
        hyps = c.get("hypotheses", [])
        if not isinstance(hyps, list):
            v.append(f"{key[0]}/{key[1]}: hypotheses must be a list")
            continue
        if skipped[key] and hyps:
            v.append(f"{key[0]}/{key[1]}: group was skipped ({skipped[key]}); no hypotheses allowed")
        gene_dir = (dirs.get(key[0]) or {}).get(key[1]) or {}
        for j, h in enumerate(hyps):
            tag = f"{key[0]}/{key[1]} hypothesis[{j}]"
            if not isinstance(h, Mapping):
                v.append(f"{tag}: must be an object")
                continue
            gene = h.get("gene")
            dirn = h.get("direction")
            if dirn not in DIRECTIONS:
                v.append(f"{tag}: direction must be up|down (got {dirn!r})")
            if not isinstance(gene, str) or gene not in gene_dir:
                v.append(f"{tag}: gene {gene!r} is not in the DGE table for this contrast/group")
            elif dirn in DIRECTIONS and gene_dir[gene] != dirn:
                v.append(f"{tag}: gene {gene} is {gene_dir[gene]} in our data, not {dirn}")
            ids = h.get("evidence_ids", [])
            if not isinstance(ids, list):
                v.append(f"{tag}: evidence_ids must be a list")
            else:
                for eid in ids:
                    if not isinstance(eid, str) or eid not in known_ids:
                        v.append(f"{tag}: unknown evidence_id {eid!r}")
            text = h.get("text")
            if not isinstance(text, str) or not text.strip():
                v.append(f"{tag}: text must be non-empty")
            elif not HYPOTHESIS_RE.search(text):
                v.append(f"{tag}: text must be labelled as a hypothesis")
    for key in proposal:
        if key != "contrasts":
            v.append(f"unexpected field {key!r}")
    return (not v), v


def policy(bundle: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The policy as shown to the reviewer (what :func:`policy_check` enforces, in words)."""
    return {
        "gate": GATE,
        "hypotheses": "every statement is a hypothesis, never a finding; each hypothesis text says so",
        "direction": "up/down comes from our DGE table for that contrast/group; a cited gene must be DE there with that direction",
        "evidence": "evidence_ids are cited only from the Open Targets summary (evidence_ids in the bundle); "
                    "an empty list is allowed only when the hypothesis rests on the expression change alone",
        "skipped_groups": "a group with a skipped_reason gets a one-line summary and no hypotheses",
        "completeness": "one entry per contrast/group, each with a non-empty summary; no contrast/group twice",
        "outside_knowledge": "no drug or trial claims beyond the evidence records; general biology only as a hypothesis",
    }


def report_sections(bundle: Mapping[str, Any], proposal: Mapping[str, Any], review: Mapping[str, Any] | None,
                    decision: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    from cellwarden.gates.report import esc, table

    content = (decision or {}).get("decision") or proposal.get("proposal") or {}
    counts = [{"contrast": c["contrast"], "group": g["group"], "n_up": g.get("n_up"), "n_down": g.get("n_down"),
               "n_tested": g.get("n_tested"), "skipped_reason": g.get("skipped_reason"),
               "genes with evidence": sum(1 for d in DIRECTIONS for r in (g.get("opentargets") or {}).get(d, []) if r.get("evidence"))}
              for c in bundle.get("contrasts") or [] for g in c.get("groups") or []]
    parts = []
    for c in content.get("contrasts") or []:
        if not isinstance(c, Mapping):
            continue
        parts.append(f"<h3>{esc(c.get('contrast'))} / {esc(c.get('group'))}</h3><p>{esc(c.get('summary'))}</p>")
        hyps = [{"gene": h.get("gene"), "direction": h.get("direction"), "evidence_ids": ", ".join(h.get("evidence_ids") or []),
                 "text": h.get("text")} for h in c.get("hypotheses") or [] if isinstance(h, Mapping)]
        parts.append(table(hyps) if hyps else "<p><em>no hypotheses</em></p>")
    return [("DGE per contrast x group", table(counts)),
            ("Interpretation (every statement is a hypothesis)", "".join(parts) or "<p><em>none</em></p>"),
            ("Open Targets", f"<p>{bundle.get('n_evidence', 0)} evidence record(s)"
                             f"{' (offline cache)' if bundle.get('opentargets_offline') else ''}</p>")]


def make_worker(provider, ledger=None, **kw):
    from cellwarden.agents.interpretation import InterpretationAgent

    return InterpretationAgent(provider, ledger, **kw)


def main(argv: list[str] | None = None) -> int:
    from cellwarden.gates.protocol import gate_main

    return gate_main(GATE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
