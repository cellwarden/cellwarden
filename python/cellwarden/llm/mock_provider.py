"""Mock provider: scripted responses for tests, no network.

Config::

    llm:
      default: {provider: mock, model: mock-model}
      roles:
        qc_worker: {script: [...responses...]}         # inline list, or a path
        reviewer:  {script: tests/fixtures/reviewer.json}

A script is a list of responses, consumed in order; one call takes one
response, and the provider raises :class:`MockScriptExhausted` when the list
runs out (a test that makes more calls than it scripted fails loudly instead
of looping). Each response is a mapping with any of:

* ``text``: the assistant text,
* ``tool_calls``: ``[{"name", "arguments", "id"?}]``, or the shorthand
  ``tool`` + ``arguments`` for a single call,
* ``json``: the structured answer (for ``json_schema`` calls; also serialised
  into ``text``),
* ``usage``: ``{input_tokens, output_tokens, cached_tokens}`` (default
  ``default_usage`` from the settings, else 1000 / 200 / 0),
* ``stop_reason`` (derived: ``tool_use`` when there are tool calls).

``calls`` keeps every request the provider saw (in-process tests inspect it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .provider import Provider, Response, ToolCall, Usage

DEFAULT_USAGE = {"input_tokens": 1000, "output_tokens": 200, "cached_tokens": 0}


class MockScriptExhausted(RuntimeError):
    """The scripted responses ran out."""


def _load_script(spec: Any) -> list[dict[str, Any]]:
    if spec is None:
        return []
    if isinstance(spec, (str, Path)):
        p = Path(spec)
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    else:
        data = spec
    if not isinstance(data, list):
        raise ValueError("mock script must be a list of responses")
    return [dict(x) for x in data]


class MockProvider(Provider):
    name = "mock"

    def __init__(self, settings: Mapping[str, Any], script: list[dict[str, Any]] | None = None, role: str = ""):
        super().__init__(settings)
        self.role = role
        self.script: list[dict[str, Any]] = list(script if script is not None else _load_script(settings.get("script")))
        self.default_usage = dict(DEFAULT_USAGE, **(settings.get("default_usage") or {}))
        self.calls: list[dict[str, Any]] = []
        self.pos = 0

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any], role: str = "") -> "MockProvider":
        return cls(settings, role=role)

    @property
    def remaining(self) -> int:
        return len(self.script) - self.pos

    def complete(self, *, system, messages, tools, max_tokens, json_schema=None) -> Response:
        self.calls.append({"system": system, "messages": json.loads(json.dumps(messages)),
                           "tools": [t["name"] for t in tools or []], "json_schema": json_schema is not None,
                           "max_tokens": max_tokens})
        if self.pos >= len(self.script):
            raise MockScriptExhausted(
                f"mock provider ({self.role or 'default'}): script exhausted after {self.pos} response(s)"
            )
        spec = self.script[self.pos]
        self.pos += 1
        calls: list[ToolCall] = []
        if "tool" in spec:
            calls.append(ToolCall(spec.get("id") or f"mock_{self.pos}", str(spec["tool"]), dict(spec.get("arguments") or {})))
        for i, t in enumerate(spec.get("tool_calls") or []):
            calls.append(ToolCall(t.get("id") or f"mock_{self.pos}_{i}", str(t["name"]), dict(t.get("arguments") or {})))
        text = str(spec.get("text") or "")
        parsed = spec.get("json")
        if parsed is not None and not text:
            text = json.dumps(parsed)
        stop = spec.get("stop_reason") or ("tool_use" if calls else "end_turn")
        usage = Usage.from_dict(dict(self.default_usage, **(spec.get("usage") or {})))
        return Response(text=text, tool_calls=calls, usage=usage, stop_reason=str(stop), json=parsed,
                        model=self.model, raw=spec)
