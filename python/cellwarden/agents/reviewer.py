"""Reviewer agent (DESIGN 4.5, agents-spec section 2 step 4).

A separate prompt (``prompts/reviewer.md``) and, when configured, a
different model (``llm.roles.reviewer``). It sees the policy, the evidence
bundle, the proposal's decision content and the code's policy check, but
**not** the worker's rationale, so it cannot anchor on it. It has no tools
and cannot change the proposal: its only output is the verdict JSON of
``review.json``::

    {"verdict": "approve" | "request_changes" | "escalate",
     "findings": [{"severity": "info" | "warn" | "block", "text": str}],
     "reviewer": {"model": str}}

The verdict is validated in code: an unparseable answer or an unknown
verdict becomes ``escalate`` with a ``block`` finding saying so, and an
``approve`` of a proposal whose policy check failed is downgraded to
``escalate`` (the reviewer cannot approve beyond policy). The same reviewer
instance is used for the revision round, with its earlier findings shown.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from .base import Agent, wrap_data

__all__ = ["VERDICTS", "SEVERITIES", "VERDICT_SCHEMA", "Reviewer", "normalise_verdict"]

VERDICTS = ("approve", "request_changes", "escalate")
SEVERITIES = ("info", "warn", "block")
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "findings": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"severity": {"type": "string", "enum": list(SEVERITIES)}, "text": {"type": "string"}},
                      "required": ["severity", "text"]},
        },
    },
    "required": ["verdict", "findings"],
}


def normalise_verdict(raw: Any, model: str, policy_ok: bool) -> dict[str, Any]:
    """Validate the model's answer and coerce it into the ``review.json`` shape."""
    findings: list[dict[str, str]] = []
    verdict = None
    if isinstance(raw, Mapping):
        v = raw.get("verdict")
        if isinstance(v, str) and v.strip().lower() in VERDICTS:
            verdict = v.strip().lower()
        for f in raw.get("findings") or []:
            if isinstance(f, Mapping):
                sev = str(f.get("severity", "info")).lower()
                findings.append({"severity": sev if sev in SEVERITIES else "info", "text": str(f.get("text", "")).strip()})
            elif isinstance(f, str):
                findings.append({"severity": "info", "text": f.strip()})
    if verdict is None:
        verdict = "escalate"
        findings.append({"severity": "block", "text": "reviewer output was not a valid verdict; escalated to a human"})
    if verdict == "approve" and not policy_ok:
        verdict = "escalate"
        findings.append({"severity": "block", "text": "the proposal fails the policy check, so it cannot be approved"})
    return {"verdict": verdict, "findings": findings, "reviewer": {"model": model}}


class Reviewer(Agent):
    role = "reviewer"
    prompt = "reviewer"
    max_tokens = 1024

    def review(self, *, gate: str, bundle: Mapping[str, Any], proposal: Mapping[str, Any],
               policy: Mapping[str, Any], policy_check: Mapping[str, Any],
               previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Ask for a verdict on ``proposal`` (the decision content only, never the rationale)."""
        payload = {
            "gate": gate,
            "policy": policy,
            "bundle": bundle,
            "proposal": proposal,
            "policy_check": policy_check,
        }
        if previous is not None:
            payload["your_previous_findings"] = previous.get("findings")
        msg = (f"Review this {gate} proposal and answer with the verdict JSON only.\n\n"
               + wrap_data(json.dumps(payload, indent=1, default=str)))
        raw = self.ask_json(msg, VERDICT_SCHEMA)  # budget / provider errors propagate: the gate falls back
        return normalise_verdict(raw, self.provider.model, bool(policy_check.get("ok", True)))
