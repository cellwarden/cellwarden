"""G3 worker agent: cell-type annotation of unresolved clusters (DESIGN 4.3, spec section 5).

Tools (plain Python, input-validated):

* ``list_low_confidence_clusters()`` - clusters with consensus status ``low_confidence|conflict``
* ``get_cluster_evidence(cluster)`` - markers, signatures, CellTypist votes, consensus votes
* ``lookup_cell_type(label)`` - a label from ``markers_human.yaml`` (or the built-in table):
  CL id, level, markers; unknown labels get close suggestions
* ``propose_labels(clusters, rationale)`` - the proposal tool; the rows are merged
  over the rules baseline (consensus) and validated by
  :func:`cellwarden.gates.g3_annotation.policy_check` (only unresolved clusters may change)

Entry point: ``AnnotationAgent(provider, ledger).run(bundle)`` -> ``proposal.json`` dict.
The model only has to decide the unresolved clusters; consensus rows are
copied through unchanged. ``Unassigned`` is a valid label.
"""

from __future__ import annotations

import difflib
from typing import Any, Mapping

from cellwarden.agents._compat import Agent
from cellwarden.agents.context import render_bundle_message, tool
from cellwarden.gates import g3_annotation as G3


from cellwarden.gates.g2_context import marker_names  # noqa: E402

def lookup_cell_type(label: str, table: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Look a label up in the label table; case-insensitive, with suggestions when absent."""
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
    table = table if table is not None else G3.load_label_table()
    key = label.strip()
    if key not in table:
        lower = {k.lower(): k for k in table}
        key = lower.get(key.lower(), key)
    if key in table:
        e = table[key]
        return {"found": True, "label": key, "cl_id": e.get("cl_id"), "level": e.get("level"),
                "parent": e.get("parent"), "markers": list(e.get("markers") or [])}
    sugg = difflib.get_close_matches(label, list(table), n=5, cutoff=0.4)
    return {"found": False, "label": label, "suggestions": sugg,
            "note": "label not in the table; use one of the suggestions or 'Unassigned'"}


class AnnotationAgent(Agent):
    role = G3.WORKER_ROLE
    gate = G3.GATE

    def user_message(self, bundle: Mapping[str, Any]) -> str:
        return render_bundle_message(
            bundle, "Evidence bundle for gate G3 (annotation). Only clusters listed in low_confidence_clusters may be relabelled.")

    def build_tools(self, bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
        by_id = {str(c["cluster"]): c for c in bundle.get("clusters", [])}
        table = G3.load_label_table()
        unresolved = [cid for cid, c in by_id.items() if (c.get("consensus") or {}).get("status") in G3.CHANGEABLE]

        def list_low_confidence_clusters() -> list[dict]:
            return [{"cluster": cid, "status": by_id[cid]["consensus"].get("status"),
                     "consensus": {k: by_id[cid]["consensus"].get(k) for k in ("label_broad", "label_fine", "cl_id", "confidence")},
                     "votes": by_id[cid]["consensus"].get("votes", {}),
                     "top_markers": marker_names(by_id[cid].get("top_markers"), 10)} for cid in unresolved]

        def get_cluster_evidence(cluster: str) -> dict:
            cid = str(cluster)
            if cid not in by_id:
                raise ValueError(f"unknown cluster {cid!r}; known: {sorted(by_id)}")
            c = by_id[cid]
            return {k: c.get(k) for k in ("cluster", "n_cells", "pct_cells", "median_genes", "median_umis", "median_pct_mito",
                                          "top_markers", "signatures", "celltypist", "consensus")}

        def _lookup(label: str) -> dict:
            return lookup_cell_type(label, table)

        def propose_labels(clusters: list, rationale: str = "") -> dict:
            if not isinstance(clusters, list):
                raise ValueError("clusters must be a list of cluster rows")
            if not isinstance(rationale, str):
                raise ValueError("rationale must be a string")
            base = {r["cluster"]: dict(r) for r in G3.baseline(bundle)["clusters"]}
            proposed: dict[str, dict] = {}
            for r in clusters:
                if not isinstance(r, Mapping) or "cluster" not in r:
                    raise ValueError("each cluster row must be an object with a 'cluster' field")
                proposed[str(r["cluster"])] = dict(r, cluster=str(r["cluster"]))
            merged = []
            for cid, row0 in base.items():
                row = proposed.pop(cid, None)
                if row is None:
                    merged.append(row0)
                    continue
                full = dict(row0)
                full.update({k: row[k] for k in ("label_broad", "label_fine", "cl_id", "confidence", "status") if k in row})
                merged.append(full)
            merged.extend(proposed.values())  # unknown clusters are reported by policy_check
            proposal = {"clusters": merged}
            ok, violations = G3.policy_check(proposal, bundle, table)
            return {"accepted": ok, "violations": violations, "proposal": proposal, "rationale": rationale}

        row_schema = {"type": "object", "properties": {
            "cluster": {"type": "string"}, "label_broad": {"type": "string"}, "label_fine": {"type": "string"},
            "cl_id": {"type": ["string", "null"]}, "confidence": {"type": "number"},
            "status": {"type": "string", "enum": ["consensus", "agent", "review"]}},
            "required": ["cluster", "label_broad", "label_fine", "cl_id", "confidence", "status"]}
        return [
            tool("list_low_confidence_clusters", "Clusters the agent may relabel (consensus status low_confidence or conflict), with their votes.",
                     {"type": "object", "properties": {}, "required": []}, list_low_confidence_clusters),
            tool("get_cluster_evidence", "Markers, signature scores, CellTypist votes and consensus for one cluster.",
                     {"type": "object", "properties": {"cluster": {"type": "string"}}, "required": ["cluster"]}, get_cluster_evidence),
            tool("lookup_cell_type", "Check a label against the cell-type table; returns its CL id and markers, or suggestions.",
                     {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]}, _lookup),
            tool("propose_labels",
                     "Submit labels for the unresolved clusters (other clusters are copied from the consensus). Relabelled rows need status 'agent'; rows you cannot resolve keep 'review' or become 'Unassigned'.",
                     {"type": "object", "properties": {"clusters": {"type": "array", "items": row_schema}, "rationale": {"type": "string"}},
                      "required": ["clusters", "rationale"]},
                 propose_labels, proposal=True),
        ]
