"""Gate protocol (docs/agents-spec.md section 2): rules first, agents advise,
reviewer checks, humans approve, everything logged.

A gate is a runner stage (``python -m cellwarden.gates.<module> --run-dir D``)
that writes ``<run-dir>/gates/<G>/{bundle,proposal,review,decision}.json``
and ``report.html``. :func:`run_gate` is the one flow every gate uses; the
gate-specific parts come from a *gate object* with this interface (G1:
:class:`cellwarden.gates.g1_qc.G1Gate`; G2-G4 are modules exposing ``GATE``,
``WORKER_ROLE``, ``build_bundle``, ``baseline``, ``policy_check`` and
``make_worker``, wrapped by :class:`ModuleGate`):

* ``gate_id``, ``worker_role``
* ``build_bundle() -> dict`` - aggregated evidence, deterministic (its hash
  ties a human decision to the evidence it was taken on)
* ``baseline(bundle) -> proposal`` - the rules-only decision content
* ``policy_check(proposal, bundle) -> (ok, violations)``
* ``make_worker(provider, ledger, **kw) -> Agent``
* optional ``normalise(proposal, bundle) -> proposal`` (canonical form,
  derived fields such as a summary), ``policy(bundle) -> dict`` (what the
  reviewer is shown as the policy), ``apply(decision, bundle)`` (write the
  decision into stage outputs; G1 rewrites ``S1/qc_decisions.tsv.gz``) and
  ``report_sections(bundle, proposal, review, decision) -> [(title, html)]``.

Flow of :func:`run_gate`
------------------------
1. ``bundle.json`` is built; a ``decision.json`` written by ``cellwarden approve``
   for this very bundle (same ``bundle_sha256``) is honoured straight away:
   applied, reported, and the run continues (this is how ``cellwarden resume``
   gets past a gate a human approved).
2. Rules first: the baseline. Without an ``llm:`` section the baseline is the
   decision (``decided_by: rules``, ``fallback: true``) and the gate obeys
   its configured mode (``auto`` continues; the stage-test configs have no
   ``llm`` and must not stop). If the budget guard refuses, the provider
   fails or the worker never makes a valid proposal (one retry), the baseline
   is the decision and the mode is forced to ``review`` (a human looks).
3. Worker proposes; code re-validates (``policy_check``) whatever the agent
   returned.
4. Reviewer (separate prompt, never sees the rationale). ``request_changes``
   gives the worker one revision round in the same conversation, then the
   reviewer looks again; a second ``request_changes`` counts as ``escalate``.
5. Mode from ``gates: {G1: auto|review|escalate}`` (defaults G1 auto, G2
   review, G3 auto, G4 review): ``auto`` + ``approve`` continues
   (``decided_by: agent``), anything else waits; ``review`` runs the agents
   and always waits (the human sees their advice); ``escalate`` goes straight
   to the human with the rules baseline and never calls a model (spec step 5
   says only "auto + approve continues", so this is how the third mode
   differs from ``review``). Waiting means ``report.html`` plus
   ``pending.json`` and **no** ``decision.json``; the stage exits 3 and the
   runner prints the ``cellwarden approve`` command.
6. :func:`approve` (``cellwarden approve --run-dir D --stage G1 [--from-file F]
   [--by NAME]``) validates the (edited) decision content against the policy
   and writes ``decision.json`` with ``decided_by: human:<name>`` and
   ``changes_vs_proposal``; ``cellwarden resume`` reruns the gate stage, which
   applies it (step 1).

``decision.json`` carries two keys beyond the spec shape: ``bundle_sha256``
and ``proposal_sha256`` (provenance for step 1 and the report).
"""

from __future__ import annotations

import argparse
import getpass
import importlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Any, Mapping

from ..config import ConfigError, canonical_json, llm_settings, load_resolved
from ..llm import AgentLog, BudgetExceeded, Ledger, ProviderError, get_provider, sha256_text

__all__ = ["GATE_MODULES", "DEFAULT_MODES", "MODES", "GateError", "GateResult", "ModuleGate", "load_gate",
           "gate_mode", "run_gate", "approve", "write_gate_files", "diff_json", "gate_main"]

GATE_MODULES = {
    "G1": "cellwarden.gates.g1_qc",
    "G2": "cellwarden.gates.g2_context",
    "G3": "cellwarden.gates.g3_annotation",
    "G4": "cellwarden.gates.g4_interpretation",
}
DEFAULT_MODES = {"G1": "auto", "G2": "review", "G3": "auto", "G4": "review"}
MODES = ("auto", "review", "escalate")
EXIT_WAITING = 3
FILES = ("bundle.json", "proposal.json", "review.json", "decision.json", "report.html", "pending.json")


class GateError(RuntimeError):
    """A gate could not run or a decision was rejected (message is for the user)."""


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class GateResult:
    gate: str
    mode: str
    status: str  # "continue" | "wait"
    decision: dict[str, Any] | None
    proposal: dict[str, Any]
    review: dict[str, Any]
    files: dict[str, Path] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def waiting(self) -> bool:
        return self.status == "wait"

    def __getitem__(self, key: str) -> Any:  # dict-like access for callers written against the shim
        if key == "continue":
            return not self.waiting
        return getattr(self, key)


class ModuleGate:
    """Adapter for module-style gates (``GATE``, ``WORKER_ROLE``, ``build_bundle``, ...)."""

    def __init__(self, mod: Any, run_dir: Path, cfg: Mapping[str, Any]):
        self.mod = mod
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.gate_id = str(getattr(mod, "GATE"))
        self.worker_role = str(getattr(mod, "WORKER_ROLE"))

    def build_bundle(self) -> dict[str, Any]:
        return self.mod.build_bundle(self.run_dir, self.cfg)

    def baseline(self, bundle: Mapping[str, Any]) -> dict[str, Any]:
        return self.mod.baseline(bundle, self.cfg)

    def policy_check(self, proposal: Mapping[str, Any], bundle: Mapping[str, Any]) -> tuple[bool, list[str]]:
        ok, v = self.mod.policy_check(proposal, bundle)
        return bool(ok), list(v)

    def make_worker(self, provider, ledger, **kw):
        return self.mod.make_worker(provider, ledger, **kw)

    def normalise(self, proposal: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
        fn = getattr(self.mod, "normalize_proposal", None) or getattr(self.mod, "normalise", None)
        return dict(fn(proposal)) if fn is not None else dict(proposal)

    def policy(self, bundle: Mapping[str, Any]) -> dict[str, Any]:
        fn = getattr(self.mod, "policy", None)
        if fn is not None:
            return dict(fn(bundle))
        p = bundle.get("policy")
        return dict(p) if isinstance(p, Mapping) else {"policy": getattr(self.mod, "__doc__", "") or ""}

    def apply(self, decision: Mapping[str, Any], bundle: Mapping[str, Any]) -> None:
        fn = getattr(self.mod, "apply", None)
        if fn is not None:
            fn(self.run_dir, self.cfg, decision, bundle)

    def report_sections(self, bundle, proposal, review, decision) -> list[tuple[str, str]]:
        fn = getattr(self.mod, "report_sections", None)
        return list(fn(bundle, proposal, review, decision)) if fn is not None else []


def load_gate(gate: Any, run_dir: str | Path, cfg: Mapping[str, Any]):
    """Gate object for ``gate`` (an id like ``G1``, a module, or an object already shaped like one)."""
    if isinstance(gate, str):
        from ..stage_names import gate_id, slug

        name = GATE_MODULES.get(gate_id(gate).upper())
        if name is None:
            raise GateError(f"unknown gate {gate!r} (gates: {', '.join(f'{slug(g)} ({g})' for g in GATE_MODULES)})")
        gate = importlib.import_module(name)
    if hasattr(gate, "make_gate"):
        return gate.make_gate(Path(run_dir), cfg)
    if hasattr(gate, "build_bundle") and hasattr(gate, "GATE"):
        return ModuleGate(gate, Path(run_dir), cfg)
    if hasattr(gate, "gate_id") and hasattr(gate, "build_bundle"):
        return gate
    raise GateError(f"{gate!r} is not a gate module or gate object")


def gate_mode(cfg: Mapping[str, Any], gate_id: str) -> str:
    mode = str((cfg.get("gates") or {}).get(gate_id) or DEFAULT_MODES.get(gate_id, "review"))
    if mode not in MODES:
        raise GateError(f"gates.{gate_id} must be one of {list(MODES)}, got {mode!r}")
    return mode


def write_gate_files(gate_dir: str | Path, bundle: Mapping[str, Any] | None, proposal: Mapping[str, Any] | None,
                     review: Mapping[str, Any] | None, decision: Mapping[str, Any] | None,
                     report_html: str | None = None) -> dict[str, Path]:
    """Write whichever of the gate files are given (``None`` leaves that file alone)."""
    gate_dir = Path(gate_dir)
    gate_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, obj in (("bundle", bundle), ("proposal", proposal), ("review", review), ("decision", decision)):
        if obj is not None:
            out[name] = _write_json(gate_dir / f"{name}.json", obj)
    if report_html is not None:
        p = gate_dir / "report.html"
        p.write_text(report_html, encoding="utf-8")
        out["report"] = p
    return out


def diff_json(old: Any, new: Any, path: str = "") -> list[str]:
    """Human-readable differences between two JSON values (``changes_vs_proposal``)."""
    out: list[str] = []
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        for k in sorted(set(old) | set(new), key=str):
            p = f"{path}.{k}" if path else str(k)
            if k not in old:
                out.append(f"{p}: added {json.dumps(new[k], default=str)[:80]}")
            elif k not in new:
                out.append(f"{p}: removed")
            else:
                out += diff_json(old[k], new[k], p)
        return out
    if isinstance(old, list) and isinstance(new, list):
        if old != new:
            if all(isinstance(x, (str, int, float)) for x in old + new):
                added, removed = sorted(set(map(str, new)) - set(map(str, old))), sorted(set(map(str, old)) - set(map(str, new)))
                out.append(f"{path}: {len(old)} -> {len(new)} items (+{len(added)}, -{len(removed)})")
            else:
                out.append(f"{path}: list changed ({len(old)} -> {len(new)} items)")
        return out
    if old != new:
        out.append(f"{path}: {json.dumps(old, default=str)[:60]} -> {json.dumps(new, default=str)[:60]}")
    return out


def _fallback_proposal(gate_id: str, role: str, base: Mapping[str, Any], reason: str,
                       model: str | None = None, violations: list[str] | None = None) -> dict[str, Any]:
    return {"gate": gate_id, "worker": {"model": model, "role": role}, "proposal": dict(base),
            "rationale": f"rules baseline: {reason}", "policy_check": {"ok": True, "violations": list(violations or [])},
            "fallback": True, "fallback_reason": reason}


def _rules_review(verdict: str, text: str, severity: str = "info") -> dict[str, Any]:
    return {"verdict": verdict, "findings": [{"severity": severity, "text": text}], "reviewer": {"model": "rules"}}


def _upgrade_report(gate_dir: Path, sections: Sequence[tuple[str, str]] = (), notes: Sequence[str] = ()) -> None:
    """Re-render report.html with the shared report templates once the gate files are on disk.

    The simple renderer in :mod:`cellwarden.gates.report` is always written first, so a failure
    here (templates missing, jinja2 absent) leaves a usable report and is only logged.
    """
    try:
        import html as _html
        from cellwarden.report.build import render_gate_report
        extra = list(sections)
        if notes:
            extra.append(("Notes", "<ul>" + "".join(f"<li>{_html.escape(str(n))}</li>" for n in notes) + "</ul>"))
        (gate_dir / "report.html").write_text(render_gate_report(gate_dir, extra), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - never fail a gate over its pretty report
        _log(f"{gate_dir.name}: shared report template unavailable ({exc}); kept the plain report")


def approve_command(run_dir: Path, gate_id: str) -> str:
    from ..stage_names import slug

    return f"cellwarden approve --run-dir {run_dir} --stage {slug(gate_id)}"


def run_gate(gate: Any, run_dir: str | Path, cfg: Mapping[str, Any]) -> GateResult:
    """Run one gate (see the module docstring). Writes the gate files; never raises for agent trouble."""
    from ..agents.reviewer import Reviewer
    from .report import render_report

    run_dir = Path(run_dir)
    g = load_gate(gate, run_dir, cfg)
    gate_id = g.gate_id
    gate_dir = run_dir / "gates" / gate_id
    gate_dir.mkdir(parents=True, exist_ok=True)
    mode = gate_mode(cfg, gate_id)
    notes: list[str] = []

    bundle = g.build_bundle()
    bundle_sha = sha256_text(canonical_json(bundle))
    bundle["bundle_sha256"] = bundle_sha
    base = g.baseline(bundle)
    _log(f"{gate_id}: bundle built (sha {bundle_sha[:12]}), mode {mode}")

    # 1. A human decision taken on this very bundle: apply and continue.
    dpath = gate_dir / "decision.json"
    if dpath.exists():
        try:
            prev = _read_json(dpath)
        except ValueError:
            prev = {}
        if str(prev.get("decided_by", "")).startswith("human:") and prev.get("bundle_sha256") == bundle_sha:
            _log(f"{gate_id}: applying the decision by {prev['decided_by']} ({prev.get('decided_utc')})")
            proposal = _read_json(gate_dir / "proposal.json") if (gate_dir / "proposal.json").exists() else \
                _fallback_proposal(gate_id, g.worker_role, base, "no proposal recorded")
            review = _read_json(gate_dir / "review.json") if (gate_dir / "review.json").exists() else \
                _rules_review("approve", "no review recorded")
            prev["mode"] = mode
            g.apply(prev, bundle)
            ledger_status = _ledger_status(run_dir, cfg)
            sections = g.report_sections(bundle, proposal, review, prev)
            html = render_report(gate_id, run_dir, bundle, proposal, review, prev, "continue", ledger_status, notes,
                                 sections)
            files = write_gate_files(gate_dir, bundle, None, None, prev, html)
            (gate_dir / "pending.json").unlink(missing_ok=True)
            _upgrade_report(gate_dir, sections, notes)
            return GateResult(gate_id, mode, "continue", prev, proposal, review, files, notes)

    # 2-4. Rules, worker, reviewer.
    llm = llm_settings(cfg)
    proposal: dict[str, Any]
    review: dict[str, Any]
    fallback = False
    ledger_status: dict[str, Any] | None = None
    if not llm or mode == "escalate":
        why = "llm not configured" if not llm else "gate mode escalate: straight to a human, agents not consulted"
        proposal = _fallback_proposal(gate_id, g.worker_role, base, why)
        review = _rules_review("approve" if not llm else "escalate", f"rules baseline used: {why}")
        decided_by = "rules"
        fallback = True
        wait = mode != "auto"
        notes.append("LLM agents disabled (no llm section): rules-only decision" if not llm else why)
        if llm:
            ledger_status = _ledger_status(run_dir, cfg)
    else:
        ledger = Ledger(run_dir / "llm" / "ledger.json", run_id=str(cfg.get("run_id") or "run"),
                        budget=llm.get("budget") or {}, total_path=(llm.get("budget") or {}).get("ledger_path") or llm.get("ledger_path"))
        log = AgentLog(run_dir / "llm" / "agent_log.jsonl")
        replay = bool(llm.get("replay", False))
        proposal, review, fallback_reason = _agents(g, gate_id, bundle, base, llm, ledger, log, replay, Reviewer, notes)
        ledger_status = ledger.status()
        notes += ledger_status.get("warnings") or []
        if fallback_reason is not None:
            fallback = True
            proposal = dict(proposal, proposal=dict(base), fallback=True, fallback_reason=fallback_reason,
                            rationale=f"rules baseline: {fallback_reason}" + (f"\n\nworker said: {proposal.get('rationale')}" if proposal.get("rationale") else ""))
            if review is None:
                review = _rules_review("escalate", f"rules baseline used: {fallback_reason}", "warn")
            decided_by = "rules"
            mode_effective = "review"
            notes.append(f"fallback to the rules baseline ({fallback_reason}); mode forced to review")
            mode = mode_effective
            wait = True
        else:
            decided_by = "agent"
            verdict = review["verdict"]
            if verdict == "approve":
                wait = mode == "review"
            else:
                wait = True
                notes.append(f"reviewer verdict {verdict}: a human must decide")

    decision_content = g.normalise(proposal["proposal"], bundle) if hasattr(g, "normalise") else dict(proposal["proposal"])
    proposal_sha = sha256_text(canonical_json(decision_content))
    files = write_gate_files(gate_dir, bundle, proposal, review, None)
    if wait:
        dpath.unlink(missing_ok=True)
        pending = {"gate": gate_id, "mode": mode, "status": "waiting", "since_utc": _utc(), "fallback": fallback,
                   "reviewer_verdict": review.get("verdict"), "approve": approve_command(run_dir, gate_id),
                   "bundle_sha256": bundle_sha, "proposal_sha256": proposal_sha}
        files["pending"] = _write_json(gate_dir / "pending.json", pending)
        sections = g.report_sections(bundle, proposal, review, None)
        html = render_report(gate_id, run_dir, bundle, proposal, review, None, "wait", ledger_status, notes, sections)
        files.update(write_gate_files(gate_dir, None, None, None, None, html))
        _upgrade_report(gate_dir, sections, notes)
        _log(f"{gate_id}: waiting for a human; {approve_command(run_dir, gate_id)}")
        return GateResult(gate_id, mode, "wait", None, proposal, review, files, notes)

    decision = {"gate": gate_id, "mode": mode, "decided_by": decided_by, "decided_utc": _utc(),
                "decision": decision_content, "changes_vs_proposal": [], "bundle_sha256": bundle_sha,
                "proposal_sha256": proposal_sha, "fallback": fallback}
    g.apply(decision, bundle)
    sections = g.report_sections(bundle, proposal, review, decision)
    html = render_report(gate_id, run_dir, bundle, proposal, review, decision, "continue", ledger_status, notes,
                         sections)
    files.update(write_gate_files(gate_dir, None, None, None, decision, html))
    (gate_dir / "pending.json").unlink(missing_ok=True)
    _upgrade_report(gate_dir, sections, notes)
    _log(f"{gate_id}: decided by {decided_by} (mode {mode}); run continues")
    return GateResult(gate_id, mode, "continue", decision, proposal, review, files, notes)


def _agents(g, gate_id, bundle, base, llm, ledger, log, replay, Reviewer, notes) -> tuple[dict, dict | None, str | None]:
    """Worker + reviewer rounds. Returns (proposal, review or None, fallback reason or None)."""
    from ..llm.mock_provider import MockScriptExhausted

    worker_role = g.worker_role
    try:
        wprov = get_provider(llm, worker_role)
        rprov = get_provider(llm, "reviewer")
    except (ProviderError, NotImplementedError, OSError, ValueError) as e:
        return _fallback_proposal(gate_id, worker_role, base, f"provider unavailable: {e}"), None, f"provider unavailable: {e}"
    worker = g.make_worker(wprov, ledger, log=log, gate=gate_id, replay=replay)
    reviewer = Reviewer(rprov, ledger, log=log, gate=gate_id, replay=replay)
    policy = g.policy(bundle) if hasattr(g, "policy") else dict(bundle.get("policy") or {})
    review: dict[str, Any] | None = None
    try:
        proposal = worker.run(bundle)
        proposal = _recheck(g, proposal, bundle)
        if not proposal["policy_check"]["ok"]:
            return proposal, None, "the worker made no valid proposal: " + "; ".join(proposal["policy_check"]["violations"][:3])
        review = reviewer.review(gate=gate_id, bundle=bundle, proposal=proposal["proposal"], policy=policy,
                                 policy_check=proposal["policy_check"])
        _log(f"{gate_id}: worker proposed ({worker.n_tool_calls} tool calls); reviewer: {review['verdict']}")
        if review["verdict"] == "request_changes":
            first_review = review
            revised = _recheck(g, worker.revise(review["findings"]), bundle)
            if revised["policy_check"]["ok"]:
                proposal = revised
                review = reviewer.review(gate=gate_id, bundle=bundle, proposal=proposal["proposal"], policy=policy,
                                         policy_check=proposal["policy_check"], previous=first_review)
                _log(f"{gate_id}: revision round; reviewer: {review['verdict']}")
            else:
                review = dict(first_review)
                review["findings"] = list(first_review["findings"]) + [
                    {"severity": "block", "text": "the revised proposal was invalid: " + "; ".join(revised["policy_check"]["violations"][:3])}]
                notes.append("revised proposal invalid; the first (valid) proposal is kept and escalated")
            if review["verdict"] == "request_changes":
                review = dict(review, verdict="escalate")
                review["findings"] = list(review["findings"]) + [
                    {"severity": "warn", "text": "changes were requested twice; escalated to a human (one revision round)"}]
            review["rounds"] = [first_review]
        proposal["worker"] = {"model": worker.model, "role": worker.role}
        proposal["n_calls"] = worker.n_calls
        proposal["n_tool_calls"] = worker.n_tool_calls
        proposal["cost_gbp"] = round(worker.cost_gbp + reviewer.cost_gbp, 6)
        return proposal, review, None
    except BudgetExceeded as e:
        reason = f"budget guard: {e}"
    except (ProviderError, MockScriptExhausted, NotImplementedError) as e:
        reason = f"provider failure: {e}"
    except Exception as e:  # noqa: BLE001 - an agent bug must not kill the run; it falls back to rules
        traceback.print_exc()
        reason = f"agent error: {type(e).__name__}: {e}"
    _log(f"{gate_id}: {reason}")
    return _fallback_proposal(gate_id, worker_role, base, reason, model=getattr(wprov, "model", None)), review, reason


def _recheck(g, proposal: dict[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Defence in depth: the gate code validates whatever the worker returned."""
    proposal = dict(proposal)
    content = proposal.get("proposal")
    if not isinstance(content, Mapping) or not content:
        proposal["policy_check"] = {"ok": False, "violations": list((proposal.get("policy_check") or {}).get("violations") or ["empty proposal"])}
        return proposal
    ok, violations = g.policy_check(content, bundle)
    prior = list((proposal.get("policy_check") or {}).get("violations") or [])
    proposal["policy_check"] = {"ok": bool(ok) and not prior, "violations": prior + [v for v in violations if v not in prior]}
    if proposal["policy_check"]["ok"] and hasattr(g, "normalise"):
        proposal["proposal"] = g.normalise(content, bundle)
    return proposal


def _ledger_status(run_dir: Path, cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    llm = llm_settings(cfg)
    if not llm:
        return None
    ledger = Ledger(run_dir / "llm" / "ledger.json", run_id=str(cfg.get("run_id") or "run"),
                    budget=llm.get("budget") or {}, total_path=(llm.get("budget") or {}).get("ledger_path") or llm.get("ledger_path"))
    return ledger.status()


def approve(gate: str, run_dir: str | Path, cfg: Mapping[str, Any], *, from_file: str | Path | None = None,
            by: str | None = None) -> dict[str, Any]:
    """Human approval: validate the (edited) decision content and write ``decision.json``."""
    run_dir = Path(run_dir)
    g = load_gate(gate, run_dir, cfg)
    gate_id = g.gate_id
    gate_dir = run_dir / "gates" / gate_id
    ppath, bpath = gate_dir / "proposal.json", gate_dir / "bundle.json"
    if not ppath.exists() or not bpath.exists():
        raise GateError(f"{gate_id} has not run yet in {run_dir} (no {ppath.name}); run or resume the pipeline first")
    proposal = _read_json(ppath)
    bundle = _read_json(bpath)
    bundle_sha = bundle.get("bundle_sha256") or sha256_text(canonical_json({k: v for k, v in bundle.items() if k != "bundle_sha256"}))
    if from_file is not None:
        try:
            edited = _read_json(Path(from_file))
        except (OSError, ValueError) as e:
            raise GateError(f"cannot read {from_file}: {e}") from e
        if isinstance(edited, Mapping) and isinstance(edited.get("decision"), Mapping):
            content = edited["decision"]
        elif isinstance(edited, Mapping) and isinstance(edited.get("proposal"), Mapping) and "gate" in edited:
            content = edited["proposal"]
        else:
            content = edited
    else:
        content = proposal.get("proposal") or {}
    if not isinstance(content, Mapping) or not content:
        raise GateError(f"{gate_id}: nothing to approve (empty decision content); use --from-file with the decision")
    ok, violations = g.policy_check(content, bundle)
    if not ok:
        raise GateError(f"{gate_id}: the decision violates the policy:\n  - " + "\n  - ".join(violations))
    content = g.normalise(content, bundle) if hasattr(g, "normalise") else dict(content)
    name = by or os.environ.get("CELLWARDEN_APPROVER") or getpass.getuser()
    decision = {
        "gate": gate_id,
        "mode": gate_mode(cfg, gate_id),
        "decided_by": f"human:{name}",
        "decided_utc": _utc(),
        "decision": content,
        "changes_vs_proposal": diff_json(proposal.get("proposal") or {}, content),
        "bundle_sha256": bundle_sha,
        "proposal_sha256": sha256_text(canonical_json(content)),
        "fallback": bool(proposal.get("fallback")),
        "from_file": str(from_file) if from_file is not None else None,
    }
    _write_json(gate_dir / "decision.json", decision)
    (gate_dir / "pending.json").unlink(missing_ok=True)
    return decision


def gate_main(gate: str, argv: list[str] | None = None) -> int:
    """CLI of a gate stage: ``--run-dir``; exit 0 (continue), 3 (waiting for a human) or 1 (error)."""
    p = argparse.ArgumentParser(prog=f"python -m cellwarden.gates ({gate})")
    p.add_argument("--run-dir", default=os.environ.get("CELLWARDEN_RUN_DIR"), required="CELLWARDEN_RUN_DIR" not in os.environ)
    a = p.parse_args(argv)
    run_dir = Path(a.run_dir).expanduser().resolve()
    t0 = time.monotonic()
    _log(f"{gate}: start (run dir {run_dir}, python {sys.version.split()[0]})")
    try:
        cfg = load_resolved(run_dir)
        res = run_gate(gate, run_dir, cfg)
    except (GateError, ConfigError) as e:
        print(f"{gate}: ERROR: {e}", file=sys.stderr, flush=True)
        return 1
    except Exception:
        traceback.print_exc()
        print(f"{gate}: ERROR: unexpected failure (traceback above)", file=sys.stderr, flush=True)
        return 1
    for n in res.notes:
        _log(f"{gate}: note: {n}")
    if res.waiting:
        _log(f"{gate}: waiting for human approval after {time.monotonic() - t0:.1f}s; "
             f"report: {res.files.get('report')}")
        return EXIT_WAITING
    _log(f"{gate}: done in {time.monotonic() - t0:.1f}s")
    return 0
