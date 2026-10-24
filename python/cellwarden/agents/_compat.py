"""Thin compatibility shim for the G2/G3/G4 worker agents.

The agents in ``context.py``, ``annotation.py`` and ``interpretation.py`` import
exactly five names from this module:

    get_provider, Agent, Ledger, run_gate, write_gate_files

Each is taken from the real module when it exists (``cellwarden.llm``,
``cellwarden.agents.base``, ``cellwarden.gates.protocol``; docs/agents-spec.md
sections 2 and 3) and otherwise from the minimal fallback below, so the
agents and their tests run before those modules land. The real modules now
exist and are what every normal import gets; the fallbacks are kept only as
a documented contract (``test_shim_fallback_loop_matches_contract`` loads
them with the real modules hidden) and can be deleted together with that
test. Nothing outside this file depends on fallback-only names.

Semantics that differ between the real protocol and the fallback
``run_gate``: the real one runs a reviewer agent (``llm.roles.reviewer``),
writes ``pending.json`` instead of ``decision.json`` while waiting for a
human, and returns a ``GateResult`` (dict-like: ``res["continue"]``); the
fallback has a rules-only reviewer and always writes ``decision.json``.

Interfaces assumed (they mirror ``cellwarden.agents.base`` / ``cellwarden.llm``):

Provider (spec 3)
    ``provider.complete(*, system, messages, tools, max_tokens, json_schema=None)``
    -> ``Response`` with ``.text``, ``.tool_calls`` (objects with ``.id``,
    ``.name``, ``.arguments``), ``.usage``, ``.stop_reason``;
    ``provider.model``. Messages are Anthropic-shaped
    (``{"role", "content": [blocks]}``). ``get_provider(cfg_llm, role)``; the
    ``mock`` provider takes ``llm.roles.<role>.script`` (list of
    ``{"text", "tool_calls": [{"name", "arguments"}]}`` responses, one per call).

Agent (spec 3)
    ``Agent(provider, ledger=None, *, log=None, log_path=None, gate=None,
    replay=False, max_tokens=None, max_tool_calls=None, tools=None,
    system=None)``; subclasses set ``role`` (prompt file ``prompts/<role>.md``)
    and ``gate`` and implement ``build_tools(bundle)`` (tool dicts
    ``{"name", "description", "input_schema", "fn", "proposal": bool}``) and
    ``user_message(bundle)``. The proposal tool's function returns
    ``{"accepted", "violations", "proposal", "rationale"}``; the loop stops on
    the first accepted proposal or after the second rejected one (one retry).
    ``run(bundle)`` returns the ``proposal.json`` dict of spec section 2;
    ``revise(findings)`` runs one more round for the reviewer's request_changes.

Gate modules (``cellwarden.gates.g2_context`` etc.)
    expose ``GATE``, ``WORKER_ROLE``, ``build_bundle(run_dir, cfg)``,
    ``baseline(bundle, cfg)``, ``policy_check(proposal, bundle)`` and
    ``make_worker(provider, ledger=None, **kw)`` (``worker.run(bundle)``). ``run_gate(gate,
    run_dir, cfg)`` drives them and writes the four gate files through
    ``write_gate_files(gate_dir, bundle, proposal, review, decision,
    report_html=None)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

__all__ = ["get_provider", "Agent", "Ledger", "run_gate", "write_gate_files"]

MAX_TOOL_CALLS = 12
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Provider + Ledger (cellwarden.llm)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only once the real module exists
    from cellwarden.llm import Ledger, get_provider  # type: ignore
except ImportError:  # fallback

    @dataclass
    class _ToolCall:
        id: str
        name: str
        arguments: dict

    @dataclass
    class _Response:
        text: str = ""
        tool_calls: list = field(default_factory=list)
        usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0})
        stop_reason: str = "end_turn"

    class _MockProvider:
        """Scripted provider: one scripted response per ``complete`` call, then silence."""

        name = "mock"

        def __init__(self, script: list[dict] | None = None, model: str = "mock", role: str = ""):
            self.script = [dict(s) for s in (script or [])]
            self.model = model
            self.role = role
            self.calls: list[dict] = []
            self.pos = 0

        def complete(self, *, system, messages, tools, max_tokens, json_schema=None):
            self.calls.append({"system": system, "messages": json.loads(json.dumps(messages, default=str)),
                               "tools": [t["name"] for t in tools or []]})
            if self.pos >= len(self.script):
                return _Response(text="", stop_reason="end_turn")
            step = self.script[self.pos]
            self.pos += 1
            calls = []
            if "tool" in step:
                calls.append(_ToolCall(step.get("id") or f"mock_{self.pos}", str(step["tool"]), dict(step.get("arguments") or {})))
            for k, tc in enumerate(step.get("tool_calls") or []):
                calls.append(_ToolCall(tc.get("id") or f"mock_{self.pos}_{k}", str(tc["name"]), dict(tc.get("arguments") or {})))
            usage = {"input_tokens": 1000, "output_tokens": 200, "cached_tokens": 0}
            return _Response(text=str(step.get("text") or ""), tool_calls=calls, usage=usage,
                             stop_reason="tool_use" if calls else "end_turn")

    def get_provider(cfg_llm: Mapping[str, Any] | None, role: str = "default"):  # type: ignore[no-redef]
        cfg_llm = dict(cfg_llm or {})
        s = dict(cfg_llm.get("default") or {})
        s.update({k: v for k, v in ((cfg_llm.get("roles") or {}).get(role) or {}).items() if v is not None})
        provider = s.get("provider", "mock")
        if provider != "mock":
            raise NotImplementedError(f"fallback get_provider supports only 'mock', not {provider!r}")
        return _MockProvider(s.get("script") or [], model=s.get("model", "mock"), role=role)

    class Ledger:  # type: ignore[no-redef]
        """Minimal budget ledger: USD prices per MTok, GBP totals, hard limits."""

        PRICES = {"mock": (0.0, 0.0), "claude-haiku-4-5": (1.0, 5.0)}

        def __init__(self, path: str | Path | None = None, *, run_id: str = "", budget: Mapping[str, Any] | None = None):
            b = dict(budget or {})
            self.path = Path(path) if path else None
            self.run_id = run_id
            self.per_run_limit = b.get("per_run_limit")
            self.total_limit = b.get("total_limit")
            self.usd_to_gbp = float(b.get("usd_to_gbp", 0.78))
            self.run_total_gbp = 0.0
            self.total_gbp = 0.0
            self.entries: list[dict] = []

        def _cost(self, model: str, in_tok: float, out_tok: float) -> float:
            pin, pout = self.PRICES.get(model, (1.0, 5.0))
            return (in_tok * pin + out_tok * pout) / 1e6 * self.usd_to_gbp

        @staticmethod
        def estimate_tokens(text: str) -> int:
            return len(text) // 4

        def charge(self, role: str, model: str, usage: Mapping[str, Any], *, gate: str | None = None) -> float:
            u = usage if isinstance(usage, Mapping) else getattr(usage, "to_dict", lambda: {})()
            cost = self._cost(model, u.get("input_tokens", 0), u.get("output_tokens", 0))
            self.run_total_gbp += cost
            self.total_gbp += cost
            self.entries.append({"ts": _utc(), "gate": gate, "role": role, "model": model, "usage": dict(u), "cost_gbp": cost})
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps({"run_id": self.run_id, "total_gbp": self.run_total_gbp,
                                                 "entries": self.entries}, indent=1))
            return cost

        def would_exceed(self, est_input_tokens: int, est_output_tokens: int, model: str | None = None) -> bool:
            est = self._cost(model or "mock", est_input_tokens, est_output_tokens)
            if self.per_run_limit is not None and self.run_total_gbp + est > float(self.per_run_limit):
                return True
            if self.total_limit is not None and self.total_gbp + est > float(self.total_limit):
                return True
            return False


# ---------------------------------------------------------------------------
# Agent base (cellwarden.agents.base)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from cellwarden.agents.base import Agent  # type: ignore
except ImportError:
    MAX_REJECTIONS = 2  # first proposal + one retry

    class Agent:  # type: ignore[no-redef]
        """Fallback tool loop with the surface of ``cellwarden.agents.base.Agent``.

        Subclasses set ``role`` (prompt file) and ``gate``, implement
        ``build_tools(bundle)`` (tool dicts with ``fn`` and ``proposal``) and
        ``user_message(bundle)``; ``run(bundle)`` returns the ``proposal.json`` dict.
        """

        role: str = "agent"
        gate: str = ""
        max_tokens: int = 2048
        max_tool_calls: int = MAX_TOOL_CALLS

        def __init__(self, provider, ledger=None, *, log=None, log_path: str | Path | None = None,
                     gate: str | None = None, replay: bool = False, max_tokens: int | None = None,
                     max_tool_calls: int | None = None, tools: list | None = None, system: str | None = None):
            self.provider = provider
            self.ledger = ledger
            self.log_path = Path(log_path) if log_path else None
            if gate:
                self.gate = gate
            self.replay = replay
            if max_tokens is not None:
                self.max_tokens = int(max_tokens)
            if max_tool_calls is not None:
                self.max_tool_calls = int(max_tool_calls)
            self.tools: list[dict] = [dict(t) for t in (tools or [])]
            self.system = system if system is not None else self.system_prompt()
            self.messages: list[dict] = []
            self.n_calls = 0
            self.n_tool_calls = 0
            self.cost_gbp = 0.0
            self.trace: list[dict] = []

        @property
        def model(self) -> str:
            return getattr(self.provider, "model", "unknown")

        def system_prompt(self) -> str:
            return (PROMPT_DIR / f"{self.role}.md").read_text(encoding="utf-8")

        def build_tools(self, bundle: Mapping[str, Any]):
            return None

        def user_message(self, bundle: Mapping[str, Any]) -> str:
            return "Evidence bundle:\n<data>\n" + json.dumps(bundle, indent=1, default=str, sort_keys=True) + "\n</data>"

        def _log(self, entry: dict) -> None:
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, default=str) + "\n")

        def _tool(self, name: str):
            return next((t for t in self.tools if t["name"] == name), None)

        def followup(self, user_message: str):
            self.messages.append({"role": "user", "content": [{"type": "text", "text": user_message}]})
            accepted = rejected = None
            rejections = 0
            final_text = ""
            for _turn in range(self.max_tool_calls + 2):
                allow = self.n_tool_calls < self.max_tool_calls
                defs = [{k: t[k] for k in ("name", "description", "input_schema")} for t in self.tools] if allow else []
                if self.ledger is not None and self.ledger.would_exceed(
                        self.ledger.estimate_tokens(self.system + json.dumps(self.messages, default=str)),
                        self.max_tokens, self.model):
                    raise RuntimeError(f"{self.role}: budget exceeded; no call was made")
                resp = self.provider.complete(system=self.system, messages=self.messages, tools=defs or None,
                                              max_tokens=self.max_tokens)
                self.n_calls += 1
                cost = self.ledger.charge(self.role, self.model, resp.usage, gate=self.gate or None) if self.ledger is not None else 0.0
                self.cost_gbp += cost
                self._log({"ts": _utc(), "gate": self.gate, "role": self.role, "model": self.model,
                           "tool_calls": [{"id": t.id, "name": t.name, "arguments": t.arguments} for t in resp.tool_calls],
                           "output": resp.text, "usage": resp.usage, "cost_gbp": cost})
                blocks = ([{"type": "text", "text": resp.text}] if resp.text else []) + \
                         [{"type": "tool_use", "id": t.id, "name": t.name, "input": t.arguments} for t in resp.tool_calls]
                self.messages.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
                if resp.text:
                    final_text = resp.text
                if not resp.tool_calls:
                    break
                results, stop = [], False
                for tc in resp.tool_calls:
                    t = self._tool(tc.name)
                    args = tc.arguments if tc.arguments is not None else {}
                    if t is None:
                        value, err = {"error": f"unknown tool {tc.name!r}"}, True
                    elif self.n_tool_calls >= self.max_tool_calls:
                        value, err = {"error": "tool budget spent"}, True
                    elif not isinstance(args, Mapping):
                        value, err = {"error": "arguments must be an object"}, True
                    else:
                        self.n_tool_calls += 1
                        try:
                            value, err = t["fn"](**args), False
                        except Exception as e:  # noqa: BLE001 - reported to the model
                            value, err = {"error": f"{type(e).__name__}: {e}"}, True
                    self.trace.append({"tool": tc.name, "arguments": args, "error": err})
                    results.append({"type": "tool_result", "tool_use_id": tc.id, "is_error": err,
                                    "content": json.dumps(value, default=str)})
                    if t is not None and t.get("proposal") and not err and isinstance(value, Mapping):
                        if value.get("accepted", True):
                            accepted, stop = dict(value), True
                        else:
                            rejections += 1
                            rejected = dict(value)
                            if rejections >= MAX_REJECTIONS:
                                stop = True
                self.messages.append({"role": "user", "content": results})
                if stop:
                    break
            return {"text": final_text, "accepted": accepted, "rejected": rejected}

        def _proposal_dict(self, res: dict) -> dict:
            worker = {"model": self.model, "role": self.role}
            if res["accepted"] is not None:
                a = res["accepted"]
                return {"gate": self.gate, "worker": worker, "proposal": a.get("proposal", {}),
                        "rationale": str(a.get("rationale", "") or res["text"]),
                        "policy_check": {"ok": True, "violations": []}, "fallback": False}
            if res["rejected"] is not None:
                r = res["rejected"]
                return {"gate": self.gate, "worker": worker, "proposal": r.get("proposal", {}),
                        "rationale": str(r.get("rationale", "")),
                        "policy_check": {"ok": False, "violations": list(r.get("violations") or [])}, "fallback": True}
            return {"gate": self.gate, "worker": worker, "proposal": {}, "rationale": res["text"],
                    "policy_check": {"ok": False, "violations": ["the worker made no proposal"]}, "fallback": True}

        def run(self, bundle: Mapping[str, Any]) -> dict:
            built = self.build_tools(bundle)
            if built is not None:
                self.tools = [dict(t) for t in built]
            self.messages = []
            self.last_result = self.followup(self.user_message(bundle))
            return self._proposal_dict(self.last_result)

        def revise(self, findings: list) -> dict:
            text = "\n".join(f"- [{f.get('severity', 'warn')}] {f.get('text', '')}" for f in findings) or "- (no details)"
            self.last_result = self.followup("The reviewer asked for changes. Address these findings and submit the proposal again:\n<data>\n" + text + "\n</data>")
            return self._proposal_dict(self.last_result)


# ---------------------------------------------------------------------------
# Gate protocol (cellwarden.gates.protocol)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from cellwarden.gates.protocol import run_gate, write_gate_files  # type: ignore
except ImportError:
    GATE_DEFAULT_MODES = {"G1": "auto", "G2": "review", "G3": "auto", "G4": "review"}

    def write_gate_files(gate_dir: str | Path, bundle: Mapping[str, Any], proposal: Mapping[str, Any],
                         review: Mapping[str, Any], decision: Mapping[str, Any],
                         report_html: str | None = None) -> dict[str, Path]:  # type: ignore[no-redef]
        gate_dir = Path(gate_dir)
        gate_dir.mkdir(parents=True, exist_ok=True)
        out = {}
        for name, obj in (("bundle", bundle), ("proposal", proposal), ("review", review), ("decision", decision)):
            p = gate_dir / f"{name}.json"
            p.write_text(json.dumps(obj, indent=1, default=str, sort_keys=True))
            out[name] = p
        if report_html is None:
            report_html = _simple_report(decision.get("gate", gate_dir.name), proposal, review, decision)
        p = gate_dir / "report.html"
        p.write_text(report_html)
        out["report"] = p
        return out

    def _simple_report(gate: str, proposal: Mapping[str, Any], review: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
        def esc(s: Any) -> str:
            return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(gate)} gate</title></head><body>"
                f"<h1>Gate {esc(gate)}</h1><h2>Decision ({esc(decision.get('mode'))}, by {esc(decision.get('decided_by'))})</h2>"
                f"<pre>{esc(json.dumps(decision.get('decision'), indent=1))}</pre>"
                f"<h2>Worker rationale</h2><pre>{esc(proposal.get('rationale', ''))}</pre>"
                f"<h2>Reviewer</h2><pre>{esc(json.dumps(review, indent=1))}</pre></body></html>")

    def _gate_module(gate: str):
        import importlib
        names = {"G2": "g2_context", "G3": "g3_annotation", "G4": "g4_interpretation"}
        if gate not in names:
            raise ValueError(f"fallback run_gate knows only G2-G4, not {gate!r}")
        return importlib.import_module(f"cellwarden.gates.{names[gate]}")

    def _make_ledger(run_dir: Path, llm_cfg: Mapping[str, Any]):
        path = llm_cfg.get("ledger_path") or (run_dir / "llm" / "ledger.json")
        budget = dict(llm_cfg.get("budget") or {})
        try:
            return Ledger(path, run_id=run_dir.name, budget=budget)
        except TypeError:  # a Ledger with another signature: run without the budget guard
            return None

    def run_gate(gate: str, run_dir: str | Path, cfg: Mapping[str, Any]) -> dict:  # type: ignore[no-redef]
        """Spec section 2 flow with a rules-only reviewer (no reviewer LLM in the fallback)."""
        run_dir = Path(run_dir)
        mod = _gate_module(gate)
        bundle = mod.build_bundle(run_dir, cfg)
        base = mod.baseline(bundle, cfg)
        mode = str((cfg.get("gates") or {}).get(gate, GATE_DEFAULT_MODES.get(gate, "review")))
        llm_cfg = cfg.get("llm")
        worker_role = mod.WORKER_ROLE
        if not llm_cfg:
            proposal = {"gate": gate, "worker": {"model": None, "role": worker_role}, "proposal": base,
                        "rationale": "rules baseline (llm not configured)",
                        "policy_check": {"ok": True, "violations": []}, "fallback": True}
        else:
            provider = get_provider(llm_cfg, worker_role)
            worker = mod.make_worker(provider, _make_ledger(run_dir, llm_cfg), log_path=run_dir / "llm" / "agent_log.jsonl")
            try:
                proposal = worker.run(bundle)
            except Exception as exc:  # noqa: BLE001 - provider/budget trouble: baseline + review
                proposal = {"gate": gate, "worker": {"model": getattr(provider, "model", None), "role": worker_role},
                            "proposal": {}, "rationale": "", "fallback": True,
                            "policy_check": {"ok": False, "violations": [f"worker failed: {type(exc).__name__}: {exc}"]}}
            if not proposal["policy_check"]["ok"]:
                proposal = dict(proposal, proposal=base, fallback=True)
        if proposal["fallback"] and mode == "auto":
            mode = "review"
        ok, viol = mod.policy_check(proposal["proposal"], bundle)
        findings = [{"severity": "block", "text": v} for v in viol]
        if proposal["fallback"]:
            findings.append({"severity": "info", "text": "rules baseline used (fallback)"})
        review = {"verdict": "approve" if ok else "escalate", "findings": findings, "reviewer": {"model": "rules"}}
        decision = {"gate": gate, "mode": mode, "decided_by": "rules" if proposal["fallback"] else "agent",
                    "decided_utc": _utc(), "decision": proposal["proposal"], "changes_vs_proposal": []}
        approved = mode == "auto" and review["verdict"] == "approve"
        files = write_gate_files(run_dir / "gates" / gate, bundle, proposal, review, decision)
        return {"gate": gate, "mode": mode, "continue": approved, "decision": decision,
                "proposal": proposal, "review": review, "files": files}
