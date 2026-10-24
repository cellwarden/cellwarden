"""Provider interface for the LLM adapter (docs/agents-spec.md section 3).

Agents talk to one interface, :class:`Provider`, and every backend is a small
adapter that translates tool definitions and tool calls:

* :mod:`cellwarden.llm.anthropic_provider` (Messages API over ``requests``),
* :mod:`cellwarden.llm.openai_provider` (``chat/completions`` with a ``base_url``,
  which also covers vLLM, Ollama and other OpenAI-compatible servers),
* :mod:`cellwarden.llm.mock_provider` (scripted responses for tests, no network),
* ``bedrock`` and ``vertex`` stubs that raise :class:`NotImplementedError`.

Internal message format (Anthropic-shaped, the adapters translate)
------------------------------------------------------------------
``messages`` is a list of ``{"role": "user" | "assistant", "content": ...}``
where ``content`` is a string or a list of blocks:

* ``{"type": "text", "text": str}``
* ``{"type": "tool_use", "id": str, "name": str, "input": dict}`` (assistant)
* ``{"type": "tool_result", "tool_use_id": str, "content": str, "is_error": bool}`` (user)

``tools`` is a list of ``{"name", "description", "input_schema"}`` (JSON
schema). ``json_schema`` asks for a structured answer: the adapters implement
it as a forced call of a synthetic tool named :data:`JSON_TOOL`, or with the
backend's native JSON mode, and the parsed object is returned in
``Response.json``.

The API key is read from the environment variable named by ``api_key_env``
and is held only on the provider instance; it is never logged, never written
to any file, and never included in error messages.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "JSON_TOOL",
    "ProviderError",
    "ToolCall",
    "Usage",
    "Response",
    "Provider",
    "role_settings",
    "get_provider",
]

JSON_TOOL = "respond_json"
PROVIDERS = ("auto", "anthropic", "openai", "nvidia", "openai_compatible", "mock", "bedrock", "vertex")
DEFAULT_MODEL = "claude-haiku-4-5"
# Named providers: the adapter they use and the settings filled in when the config leaves them out.
# `auto` picks the first of AUTO_ORDER whose key variable is set (docs/PIPELINE.md section 5).
PRESETS: dict[str, dict[str, Any]] = {
    "anthropic": {"adapter": "anthropic", "api_key_env": "ANTHROPIC_API_KEY", "model": DEFAULT_MODEL},
    "openai": {"adapter": "openai_compatible", "api_key_env": "OPENAI_API_KEY", "model": "gpt-4.1-mini",
               "base_url": "https://api.openai.com/v1"},
    # NVIDIA NIM (build.nvidia.com): OpenAI-compatible; json_object, since json_schema is not on every NIM model.
    # (meta/llama-3.3-70b-instruct, the earlier default, reached end of life on NIM on 2026-08-26.)
    "nvidia": {"adapter": "openai_compatible", "api_key_env": "NVIDIA_API_KEY",
               "model": "nvidia/nemotron-3-ultra-550b-a55b",
               "base_url": "https://integrate.api.nvidia.com/v1", "json_mode": "object"},
}
AUTO_ORDER = ("anthropic", "openai", "nvidia")
FLAT_KEYS = ("provider", "model", "api_key_env", "base_url", "max_tokens", "timeout_s", "max_retries",
             "temperature", "json_mode", "default_usage", "extra_body")
# Nemotron models reason inside their answer unless told not to, which spends the agent's token
# budget before the JSON and slows every call; their chat template takes this switch.
NEMOTRON_NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


class ProviderError(RuntimeError):
    """The backend could not produce a response (network, HTTP error, bad payload)."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"input_tokens": int(self.input_tokens), "output_tokens": int(self.output_tokens),
                "cached_tokens": int(self.cached_tokens)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "Usage":
        d = d or {}
        return cls(int(d.get("input_tokens", 0) or 0), int(d.get("output_tokens", 0) or 0),
                   int(d.get("cached_tokens", 0) or 0))


@dataclass
class Response:
    """One model turn. ``stop_reason`` is ``end_turn``, ``tool_use``, ``max_tokens`` or ``error``."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "end_turn"
    json: Any = None
    model: str = ""
    raw: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (what the agent log stores as ``output``)."""
        return {"text": self.text, "tool_calls": [t.to_dict() for t in self.tool_calls],
                "usage": self.usage.to_dict(), "stop_reason": self.stop_reason, "json": self.json,
                "model": self.model}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Response":
        return cls(
            text=str(d.get("text") or ""),
            tool_calls=[ToolCall(str(t.get("id") or f"call_{i}"), str(t["name"]), dict(t.get("arguments") or {}))
                        for i, t in enumerate(d.get("tool_calls") or [])],
            usage=Usage.from_dict(d.get("usage")),
            stop_reason=str(d.get("stop_reason") or "end_turn"),
            json=d.get("json"),
            model=str(d.get("model") or ""),
        )

    def assistant_content(self) -> list[dict[str, Any]]:
        """This turn as assistant content blocks, to append to the conversation."""
        blocks: list[dict[str, Any]] = []
        if self.text:
            blocks.append({"type": "text", "text": self.text})
        for t in self.tool_calls:
            blocks.append({"type": "tool_use", "id": t.id, "name": t.name, "input": t.arguments})
        return blocks or [{"type": "text", "text": ""}]


class Provider:
    """Base class. Subclasses implement :meth:`complete`."""

    name = "base"

    def __init__(self, settings: Mapping[str, Any]):
        self.settings = dict(settings)
        self.model: str = str(settings.get("model") or DEFAULT_MODEL)
        self.timeout: float = float(settings.get("timeout_s", 120))
        self.max_retries: int = int(settings.get("max_retries", 3))
        env = settings.get("api_key_env")
        self._api_key: str | None = os.environ.get(str(env)) if env else None

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    def complete(self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                 max_tokens: int, json_schema: dict[str, Any] | None = None) -> Response:
        raise NotImplementedError

    def __repr__(self) -> str:  # never shows the key
        return f"<{type(self).__name__} model={self.model!r}>"


def parse_json_text(text: str) -> Any:
    """Parse JSON from model text, tolerating a ``json`` code fence around it."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s[3:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if 0 <= a < b:
            try:
                return json.loads(s[a:b + 1])
            except json.JSONDecodeError:
                pass
    return None


def auto_provider(env: Mapping[str, str] | None = None) -> str | None:
    """First named provider (Claude, then OpenAI, then NVIDIA) whose API key variable is set."""
    env = os.environ if env is None else env
    for name in AUTO_ORDER:
        if env.get(PRESETS[name]["api_key_env"]):
            return name
    return None


def apply_preset(settings: Mapping[str, Any], env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Settings with ``auto`` resolved and the named provider's defaults filled in where unset.

    A role override that switches provider gets that provider's key variable, base URL and
    model unless it names its own.
    """
    s = {k: v for k, v in settings.items() if v is not None}
    prov = s.get("provider") or "auto"
    if prov == "auto":
        prov = auto_provider(env) or "anthropic"
    s["provider"] = prov
    for k, v in PRESETS.get(prov, {}).items():
        if k != "adapter":
            s.setdefault(k, v)
    s.setdefault("model", DEFAULT_MODEL)
    if prov == "nvidia" and "nemotron" in str(s["model"]).lower():
        s.setdefault("extra_body", NEMOTRON_NO_THINKING)
    return s


def role_settings(cfg_llm: Mapping[str, Any], role: str) -> dict[str, Any]:
    """``llm.default`` with ``llm.roles.<role>`` layered on top, provider defaults filled in."""
    base = dict(cfg_llm.get("default") or {})
    for k in FLAT_KEYS:  # llm: {provider: ..., model: ...} is a shorthand for llm.default
        if k in cfg_llm and k not in base:
            base[k] = cfg_llm[k]
    over = {k: v for k, v in ((cfg_llm.get("roles") or {}).get(role) or {}).items() if v is not None}
    if over.get("provider") and over["provider"] != base.get("provider"):
        # a different provider: the default's endpoint, key and model belong to the other one
        base = {k: v for k, v in base.items() if k not in ("api_key_env", "base_url", "model", "json_mode")}
    base.update(over)
    return apply_preset(base)


def get_provider(cfg_llm: Mapping[str, Any], role: str = "default") -> Provider:
    """Build the provider for ``role`` from the ``llm:`` config section."""
    s = role_settings(cfg_llm, role)
    kind = str(s.get("provider"))
    if kind == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(s)
    if kind in ("openai_compatible", "openai", "nvidia"):
        from .openai_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider(s)
    if kind == "mock":
        from .mock_provider import MockProvider

        if s.get("script") is None:  # scripts may also sit at llm.script (a list, or a dict keyed by role)
            script = cfg_llm.get("script")
            if isinstance(script, Mapping):
                script = script.get(role)
            s["script"] = script
        return MockProvider.from_settings(s, role)
    if kind == "bedrock":
        raise NotImplementedError(
            "llm.provider: bedrock is not implemented yet; use provider anthropic (direct API) or "
            "openai_compatible (any OpenAI-style endpoint), or mock for tests"
        )
    if kind == "vertex":
        raise NotImplementedError(
            "llm.provider: vertex is not implemented yet; use provider anthropic (direct API) or "
            "openai_compatible (any OpenAI-style endpoint), or mock for tests"
        )
    raise ProviderError(f"unknown llm provider {kind!r} (allowed: {list(PROVIDERS)})")
