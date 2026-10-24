"""Agent base class (docs/agents-spec.md section 3, "Agent base").

An agent is a system prompt file (``prompts/<role>.md``), a list of tools
(plain Python functions that validate their own inputs) and a tool-use loop
with a hard cap of ``max_tool_calls`` (12). Agents never touch data: a tool
returns JSON, and the only durable effect of a run is the *proposal* that
the gate code validates afterwards.

Subclass contract
-----------------
Set ``role`` (prompt file name) and ``gate``. Provide tools either through
the constructor (``tools=[Tool(...)]``) or by overriding
``build_tools(bundle)``; a tool is a :class:`Tool` or a plain dict
``{"name", "description", "input_schema", "fn", "proposal": bool}``. The
tool marked ``proposal`` (``Tool.terminal``) submits the answer: its
function returns ``{"accepted": bool, "violations": [...], "proposal": {...},
"rationale": str}``. ``user_message(bundle)`` renders the first user turn
(default: the bundle as JSON inside ``<data>``).

:meth:`Agent.run` (bundle) -> proposal dict in the ``proposal.json`` shape of
spec section 2 (``gate, worker, proposal, rationale, policy_check,
fallback``); :meth:`Agent.revise` (findings) -> the same after one revision
round in the same conversation (the reviewer's ``request_changes``).

Loop
----
1. The user message is appended; the provider is asked for a turn with the
   tool definitions.
2. Every tool call in the turn is executed; exceptions become error results
   (``is_error: true``) so the model can correct itself.
3. The loop stops on the first accepted proposal, after the second rejected
   one (one retry, spec section 2 step 3), when the model stops calling
   tools, or when the tool budget is spent (one last turn without tools).

Every provider call goes through :meth:`Agent.call`: the budget ledger is
asked first (:class:`~cellwarden.llm.ledger.BudgetExceeded` is raised and no
call is made when a limit would be crossed), a logged output is reused in
replay mode, the ledger is charged and ``agent_log.jsonl`` is appended.

Untrusted text: anything from data (sample sheet columns, file names, free
text) is passed through :func:`wrap_data`, and every system prompt states
that ``<data>`` content is never an instruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..config import canonical_json
from ..llm import AgentLog, BudgetExceeded, Ledger, Provider, Response, sha256_text

__all__ = ["PROMPTS_DIR", "MAX_TOOL_CALLS", "MAX_REJECTIONS", "AgentError", "Tool", "AgentResult", "Agent",
           "wrap_data", "load_prompt"]

PROMPTS_DIR = Path(__file__).with_name("prompts")
MAX_TOOL_CALLS = 12
MAX_REJECTIONS = 2  # first proposal + one retry


class AgentError(RuntimeError):
    """The agent could not produce a result (provider trouble, script exhausted, ...)."""


@dataclass
class Tool:
    """A tool the model may call. ``fn(**arguments)`` returns a JSON-serialisable value or raises."""

    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    terminal: bool = False  # the proposal tool

    def definition(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}

    @classmethod
    def coerce(cls, t: "Tool | Mapping[str, Any]") -> "Tool":
        if isinstance(t, Tool):
            return t
        return cls(str(t["name"]), str(t.get("description", "")), dict(t.get("input_schema") or {"type": "object"}),
                   t["fn"], bool(t.get("proposal") or t.get("terminal")))


@dataclass
class AgentResult:
    """Outcome of one loop: the accepted proposal-tool result (or the last rejected one) and the trace."""

    text: str
    n_calls: int
    n_tool_calls: int
    cost_gbp: float
    accepted: dict[str, Any] | None = None
    rejected: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)


def wrap_data(text: Any) -> str:
    """Mark text that comes from data (never instructions) for the model."""
    s = "" if text is None else str(text)
    return "<data>\n" + s.replace("</data>", "<\\/data>") + "\n</data>"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


class Agent:
    """See the module docstring."""

    role: str = "agent"
    gate: str = ""
    max_tokens: int = 2048
    max_tool_calls: int = MAX_TOOL_CALLS

    def __init__(self, provider: Provider, ledger: Ledger | None = None, *, log: AgentLog | None = None,
                 log_path: str | Path | None = None, gate: str | None = None, replay: bool = False,
                 max_tokens: int | None = None, max_tool_calls: int | None = None,
                 tools: list[Tool | Mapping[str, Any]] | None = None, system: str | None = None):
        self.provider = provider
        self.ledger = ledger
        self.log = log if log is not None else (AgentLog(log_path) if log_path else None)
        if gate:
            self.gate = gate
        self.replay = replay
        if max_tokens is None:  # llm.roles.<role>.max_tokens (or llm.default) overrides the class default
            max_tokens = (getattr(provider, "settings", None) or {}).get("max_tokens")
        if max_tokens is not None:
            self.max_tokens = int(max_tokens)
        if max_tool_calls is not None:
            self.max_tool_calls = int(max_tool_calls)
        self.tools: list[Tool] = [Tool.coerce(t) for t in (tools or [])]
        self.system = system if system is not None else self.system_prompt()
        self.messages: list[dict[str, Any]] = []
        self.n_calls = 0
        self.n_tool_calls = 0
        self.cost_gbp = 0.0
        self.trace: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return getattr(self.provider, "model", "unknown")

    # -- subclass hooks -------------------------------------------------------

    def system_prompt(self) -> str:
        return load_prompt(self.role)

    def build_tools(self, bundle: Mapping[str, Any]) -> list[Tool | Mapping[str, Any]] | None:
        """Tools for this bundle; ``None`` keeps the tools given to the constructor."""
        return None

    def user_message(self, bundle: Mapping[str, Any]) -> str:
        return ("Evidence bundle. Inspect it with the tools, then answer with the proposal tool only.\n\n"
                + wrap_data(json.dumps(bundle, indent=1, default=str, sort_keys=True)))

    # -- one provider call ----------------------------------------------------

    def call(self, messages: list[dict[str, Any]], *, tools: list[Tool] | None = None,
             json_schema: dict[str, Any] | None = None, max_tokens: int | None = None) -> Response:
        """Budget guard, replay lookup, provider call, ledger charge and log entry."""
        max_tokens = int(max_tokens or self.max_tokens)
        defs = [t.definition() for t in (tools if tools is not None else self.tools)]
        prompt_sha = sha256_text(self.system + "\n" + canonical_json(defs) + "\n" + canonical_json(json_schema))
        input_sha = sha256_text(canonical_json(messages))
        model = self.model
        if self.replay and self.log is not None:
            hit = self.log.find(prompt_sha, input_sha)
            if hit is not None:
                resp = Response.from_dict(hit["output"])
                self.n_calls += 1
                self.log.append(gate=self.gate, role=self.role, model=model, prompt_sha256=prompt_sha,
                                input_sha256=input_sha, response=resp, cost_gbp=0.0, replay=True,
                                extra={"replayed_from_ts": hit.get("ts")})
                self.trace.append({"call": self.n_calls, "replay": True, "stop_reason": resp.stop_reason,
                                   "tool_calls": [t.name for t in resp.tool_calls], "cost_gbp": 0.0})
                return resp
        if self.ledger is not None:
            est_in = self.ledger.estimate_tokens(self.system + canonical_json(defs) + canonical_json(messages))
            if self.ledger.would_exceed(est_in, max_tokens, model):
                st = self.ledger.status()
                raise BudgetExceeded(
                    f"{self.role}: a call of ~{est_in} input + {max_tokens} output tokens would exceed the budget "
                    f"(run {st['run_gbp']:.3f}/{st['per_run_limit']:.2f} GBP, total {st['total_gbp']:.3f}/"
                    f"{st['total_limit']:.2f} GBP); no call was made"
                )
        resp = self.provider.complete(system=self.system, messages=messages, tools=defs or None,
                                      max_tokens=max_tokens, json_schema=json_schema)
        self.n_calls += 1
        cost = self.ledger.charge(self.role, model, resp.usage, gate=self.gate or None) if self.ledger is not None else 0.0
        self.cost_gbp += cost
        if self.log is not None:
            self.log.append(gate=self.gate, role=self.role, model=model, prompt_sha256=prompt_sha,
                            input_sha256=input_sha, response=resp, cost_gbp=cost)
        self.trace.append({"call": self.n_calls, "replay": False, "stop_reason": resp.stop_reason,
                           "tool_calls": [t.name for t in resp.tool_calls], "cost_gbp": cost})
        return resp

    # -- tool loop --------------------------------------------------------------

    def _tool(self, name: str) -> Tool | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def _execute(self, name: str, arguments: Any) -> tuple[Any, bool]:
        tool = self._tool(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}; available: {[t.name for t in self.tools]}"}, True
        if self.n_tool_calls >= self.max_tool_calls and not tool.terminal:
            return {"error": f"tool budget of {self.max_tool_calls} calls is spent; submit the proposal now"}, True
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            return {"error": "arguments must be an object"}, True
        self.n_tool_calls += 1
        try:
            result = tool.fn(**arguments)
        except TypeError as e:  # wrong argument names
            return {"error": f"bad arguments for {name}: {e}"}, True
        except Exception as e:  # noqa: BLE001 - every tool failure goes back to the model, never up
            return {"error": f"{type(e).__name__}: {e}"}, True
        return result, False

    def start(self, user_message: str) -> AgentResult:
        """Fresh conversation; run the loop on ``user_message``."""
        self.messages = []
        return self.followup(user_message)

    def followup(self, user_message: str) -> AgentResult:
        """Append ``user_message`` to the conversation and run the loop again."""
        self.messages.append({"role": "user", "content": [{"type": "text", "text": user_message}]})
        accepted: dict[str, Any] | None = None
        rejected: dict[str, Any] | None = None
        rejections = 0
        nudged = False
        final_text = ""
        for _turn in range(self.max_tool_calls + 3):
            allow_tools = self.n_tool_calls < self.max_tool_calls
            # with the budget spent only the proposal tool stays, so the agent can still answer
            resp = self.call(self.messages, tools=self.tools if allow_tools else [t for t in self.tools if t.terminal])
            self.messages.append({"role": "assistant", "content": resp.assistant_content()})
            if resp.text:
                final_text = resp.text
            if not resp.tool_calls:
                proposal = next((t.name for t in self.tools if t.terminal), None)
                if proposal and accepted is None and rejected is None and not nudged:
                    # some models answer in prose (or with an empty turn) first; ask once for the tool call
                    nudged = True
                    self.messages.append({"role": "user", "content": [{"type": "text", "text":
                        f"Submit your answer by calling the {proposal} tool (inspect with the other tools first if needed)."}]})
                    continue
                break
            results: list[dict[str, Any]] = []
            stop = False
            for tc in resp.tool_calls:
                value, is_error = self._execute(tc.name, tc.arguments)
                self.trace.append({"tool": tc.name, "arguments": tc.arguments, "error": is_error,
                                   "result_preview": json.dumps(value, default=str)[:300]})
                results.append({"type": "tool_result", "tool_use_id": tc.id, "is_error": is_error,
                                "content": json.dumps(value, default=str)})
                tool = self._tool(tc.name)
                if tool is not None and tool.terminal and not is_error and isinstance(value, Mapping):
                    if value.get("accepted", True):
                        accepted = dict(value)
                        stop = True
                    else:
                        rejections += 1
                        rejected = dict(value)
                        if rejections >= MAX_REJECTIONS:
                            stop = True
            self.messages.append({"role": "user", "content": results})
            if stop:
                break
        return AgentResult(text=final_text, n_calls=self.n_calls, n_tool_calls=self.n_tool_calls,
                           cost_gbp=self.cost_gbp, accepted=accepted, rejected=rejected, trace=list(self.trace))

    # -- proposal API -----------------------------------------------------------

    def _proposal_dict(self, res: AgentResult) -> dict[str, Any]:
        worker = {"model": self.model, "role": self.role}
        if res.accepted is not None:
            return {"gate": self.gate, "worker": worker, "proposal": res.accepted.get("proposal", {}),
                    "rationale": str(res.accepted.get("rationale", "") or res.text),
                    "policy_check": {"ok": True, "violations": []}, "fallback": False}
        if res.rejected is not None:
            return {"gate": self.gate, "worker": worker, "proposal": res.rejected.get("proposal", {}),
                    "rationale": str(res.rejected.get("rationale", "")),
                    "policy_check": {"ok": False, "violations": list(res.rejected.get("violations") or [])},
                    "fallback": True}
        return {"gate": self.gate, "worker": worker, "proposal": {}, "rationale": res.text,
                "policy_check": {"ok": False, "violations": ["the worker made no proposal"]}, "fallback": True}

    def run(self, bundle: Mapping[str, Any]) -> dict[str, Any]:
        """Propose on ``bundle``; returns the ``proposal.json`` dict (spec section 2)."""
        built = self.build_tools(bundle)
        if built is not None:
            self.tools = [Tool.coerce(t) for t in built]
        self.last_result = self.start(self.user_message(bundle))
        return self._proposal_dict(self.last_result)

    def revise(self, findings: list[Mapping[str, Any]]) -> dict[str, Any]:
        """One revision round: the reviewer's findings go back in the same conversation."""
        text = "\n".join(f"- [{f.get('severity', 'warn')}] {f.get('text', '')}" for f in findings) or "- (no details)"
        msg = ("The reviewer asked for changes. Address these findings and submit the proposal again "
               "(or re-submit it with a better rationale if the findings are mistaken):\n" + wrap_data(text))
        self.last_result = self.followup(msg)
        return self._proposal_dict(self.last_result)

    def ask_json(self, user_message: str, json_schema: dict[str, Any]) -> Any:
        """One structured-output call without tools (used by the reviewer)."""
        from ..llm.provider import parse_json_text

        messages = [{"role": "user", "content": [{"type": "text", "text": user_message}]}]
        resp = self.call(messages, tools=[], json_schema=json_schema)
        return resp.json if resp.json is not None else parse_json_text(resp.text)
