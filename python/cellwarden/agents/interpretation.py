"""G4 worker agent: DGE interpretation with Open Targets evidence (DESIGN 4.4, spec section 6).

Tools (plain Python, input-validated):

* ``list_contrasts()`` - contrast x group rows with DE counts and skip reasons
* ``get_dge_top(contrast, group, direction, n)`` - top DE genes (n <= 50) with log2fc/padj
  and whether Open Targets evidence exists for each
* ``get_target_evidence(gene)`` - the Open Targets evidence records for a gene,
  one per ``evidence_id`` with the contrast/group/direction contexts the gene
  is DE in (a record found under several groups is listed once)
* ``propose_interpretation(contrasts, rationale)`` - the proposal tool; validated by
  :func:`cellwarden.gates.g4_interpretation.policy_check` after
  :func:`cellwarden.gates.g4_interpretation.normalize_proposal` labels every text
  as a hypothesis

Entry point: ``InterpretationAgent(provider, ledger).run(bundle)`` -> ``proposal.json`` dict.
The G4 bundle holds only aggregated DGE and Open Targets tables; there is no
free text from samples, so nothing needs ``<data>`` wrapping (the renderer
still wraps ``sample_metadata_hint`` if a future bundle carries one).
"""

from __future__ import annotations

from typing import Any, Mapping

from cellwarden.agents._compat import Agent
from cellwarden.agents.context import render_bundle_message, tool
from cellwarden.gates import g4_interpretation as G4

MAX_TOP = 50


class InterpretationAgent(Agent):
    role = G4.WORKER_ROLE
    gate = G4.GATE
    # one summary and hypotheses per contrast x group: 2,048 tokens ran out with six cell types
    max_tokens = 4096
    # evidence is fetched gene by gene; markers of every cell type need more than the default 12 lookups
    max_tool_calls = 24

    def user_message(self, bundle: Mapping[str, Any]) -> str:
        # The per-gene Open Targets evidence, the full gene directions and the evidence ID list are
        # reached through the tools; sending them inline made the first turn hundreds of thousands of
        # tokens for a markers-per-cell-type run.
        view = {k: v for k, v in bundle.items() if k not in ("gene_directions", "evidence_ids")}
        view["contrasts"] = [
            {**{k: v for k, v in c.items() if k != "groups"},
             "groups": [{k: v for k, v in g.items() if k != "opentargets"} for g in c.get("groups", [])]}
            for c in bundle.get("contrasts", [])]
        return render_bundle_message(
            view,
            "Evidence bundle for gate G4 (DGE interpretation). Directions come from our data; Open Targets "
            "evidence is cited by evidence_id. Per-gene evidence is not shown here: use the tools.",
        )

    def build_tools(self, bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str], dict] = {}
        for c in bundle.get("contrasts", []):
            for g in c.get("groups", []):
                groups[(str(c["contrast"]), str(g["group"]))] = g
        # One record per (gene, evidence_id); the same Open Targets record found under several
        # contrast/group/direction blocks is listed once, with every context the gene is DE in.
        evidence_by_gene: dict[str, dict[str, dict]] = {}
        for (cname, gname), g in groups.items():
            for dirn in G4.DIRECTIONS:
                for rec in (g.get("opentargets") or {}).get(dirn, []):
                    gene = str(rec.get("gene"))
                    for e in rec.get("evidence") or []:
                        eid = str(e.get("evidence_id"))
                        entry = evidence_by_gene.setdefault(gene, {}).setdefault(eid, dict(e, contexts=[]))
                        entry["contexts"].append({"contrast": cname, "group": gname, "direction": dirn,
                                                  "log2fc": rec.get("log2fc")})
        gene_dirs = bundle.get("gene_directions") or {}

        def _group(contrast: str, group: str) -> dict:
            key = (str(contrast), str(group))
            if key not in groups:
                raise ValueError(f"unknown contrast/group {key!r}; known: {sorted(groups)}")
            return groups[key]

        def list_contrasts() -> list[dict]:
            return [{"contrast": c, "group": g, "n_up": v.get("n_up"), "n_down": v.get("n_down"),
                     "n_tested": v.get("n_tested"), "skipped_reason": v.get("skipped_reason")}
                    for (c, g), v in groups.items()]

        def get_dge_top(contrast: str, group: str, direction: str, n: int = 10) -> dict:
            if direction not in G4.DIRECTIONS:
                raise ValueError(f"direction must be one of {G4.DIRECTIONS}")
            if not isinstance(n, int) or isinstance(n, bool) or n < 1 or n > MAX_TOP:
                raise ValueError(f"n must be an integer in [1, {MAX_TOP}]")
            g = _group(contrast, group)
            if g.get("skipped_reason"):
                return {"contrast": contrast, "group": group, "direction": direction, "genes": [],
                        "skipped_reason": g["skipped_reason"]}
            top = list(g.get("top_up" if direction == "up" else "top_down") or [])
            seen = {str(r.get("gene")) for r in top}
            # Extend from the full table's direction map (names only) when the summary top list is short.
            for gene, d in ((gene_dirs.get(str(contrast)) or {}).get(str(group)) or {}).items():
                if d == direction and gene not in seen:
                    top.append({"gene": gene})
                    seen.add(gene)
            with_ev = {str(r.get("gene")) for r in (g.get("opentargets") or {}).get(direction, []) if r.get("evidence")}
            out = [dict(r, has_opentargets_evidence=str(r.get("gene")) in with_ev) for r in top[:n]]
            return {"contrast": contrast, "group": group, "direction": direction, "genes": out}

        def get_target_evidence(gene: str) -> dict:
            if not isinstance(gene, str) or not gene.strip():
                raise ValueError("gene must be a non-empty string")
            recs = list(evidence_by_gene.get(gene.strip(), {}).values())
            return {"gene": gene, "n_evidence": len(recs), "evidence": recs,
                    "note": None if recs else "no Open Targets evidence recorded for this gene"}

        def propose_interpretation(contrasts: list, rationale: str = "") -> dict:
            if not isinstance(contrasts, list):
                raise ValueError("contrasts must be a list")
            if not isinstance(rationale, str):
                raise ValueError("rationale must be a string")
            proposal = G4.normalize_proposal({"contrasts": contrasts})
            ok, violations = G4.policy_check(proposal, bundle)
            out = {"accepted": ok, "violations": violations, "proposal": proposal, "rationale": rationale}
            if not ok:
                # name the valid IDs of the genes cited, so the retry can fix unknown evidence_ids
                valid = {}
                for c in proposal.get("contrasts") or []:
                    for h in c.get("hypotheses") or []:
                        gene = str(h.get("gene"))
                        bad = [e for e in h.get("evidence_ids") or [] if e not in (evidence_by_gene.get(gene) or {})]
                        if bad and gene not in valid:
                            valid[gene] = list((evidence_by_gene.get(gene) or {}))[:8]
                if valid:
                    out["valid_evidence_ids"] = valid
                    out["hint"] = ("cite only these evidence_ids (from get_target_evidence); an empty list means "
                                   "the gene has no Open Targets evidence, so its hypothesis cites none")
            return out

        hyp_schema = {"type": "object", "properties": {
            "gene": {"type": "string"}, "direction": {"type": "string", "enum": list(G4.DIRECTIONS)},
            "evidence_ids": {"type": "array", "items": {"type": "string"}}, "text": {"type": "string"}},
            "required": ["gene", "direction", "evidence_ids", "text"]}
        contrast_schema = {"type": "object", "properties": {
            "contrast": {"type": "string"}, "group": {"type": "string"}, "summary": {"type": "string"},
            "hypotheses": {"type": "array", "items": hyp_schema}},
            "required": ["contrast", "group", "summary", "hypotheses"]}
        return [
            tool("list_contrasts", "Contrast x cell-type group rows with DE counts and skip reasons.",
                     {"type": "object", "properties": {}, "required": []}, list_contrasts),
            tool("get_dge_top", "Top differentially expressed genes for a contrast/group/direction (n <= 50), flagged when Open Targets evidence exists.",
                     {"type": "object", "properties": {"contrast": {"type": "string"}, "group": {"type": "string"},
                                                       "direction": {"type": "string", "enum": list(G4.DIRECTIONS)},
                                                       "n": {"type": "integer", "minimum": 1, "maximum": MAX_TOP}},
                      "required": ["contrast", "group", "direction"]}, get_dge_top),
            tool("get_target_evidence", "All Open Targets evidence records for a gene (drugs, tractability, disease associations), each with an evidence_id to cite.",
                     {"type": "object", "properties": {"gene": {"type": "string"}}, "required": ["gene"]}, get_target_evidence),
            tool("propose_interpretation",
                     "Submit the interpretation: one entry per contrast/group with a summary and hypotheses, each citing evidence_ids from get_target_evidence. Rejected with violations if a gene, direction or evidence_id does not match our data.",
                     {"type": "object", "properties": {"contrasts": {"type": "array", "items": contrast_schema}, "rationale": {"type": "string"}},
                      "required": ["contrasts", "rationale"]},
                 propose_interpretation, proposal=True),
        ]
