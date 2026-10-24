"""Anthropic Messages API adapter (``requests``, no SDK).

POST ``{base_url}/v1/messages`` with ``anthropic-version: 2023-06-01``. The
system prompt goes as one text block with ``cache_control: ephemeral`` so the
fixed prompt and tool definitions are read from the prompt cache on repeated
calls (the agents' prompts are long and constant within a gate).

Structured output (``json_schema``) is a forced call of the synthetic tool
:data:`~cellwarden.llm.provider.JSON_TOOL`; its arguments are returned as
``Response.json`` and also serialised into ``Response.text``.

Usage mapping: ``input_tokens`` = uncached input + cache-creation tokens (both
billed at the input price here; cache writes cost a little more at the
provider, which keeps the ledger conservative only by a few percent),
``cached_tokens`` = ``cache_read_input_tokens``.

Retries: 429, 5xx and connection errors are retried with exponential backoff
(``max_retries``, default 3). The API key never appears in logs or errors.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .provider import JSON_TOOL, Provider, ProviderError, Response, ToolCall, Usage

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, settings):
        super().__init__(settings)
        self.base_url = str(settings.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        if not self._api_key:
            raise ProviderError(
                f"llm: environment variable {settings.get('api_key_env')!r} is not set (the Anthropic API key)"
            )

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key or "", "anthropic-version": API_VERSION, "content-type": "application/json"}

    def _payload(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
                 max_tokens: int, json_schema: dict[str, Any] | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(max_tokens),
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
        }
        if "temperature" in self.settings:
            body["temperature"] = self.settings["temperature"]
        tdefs = [dict(t) for t in (tools or [])]
        if json_schema is not None:
            tdefs.append({"name": JSON_TOOL, "description": "Return the answer as a structured object.",
                          "input_schema": json_schema})
            body["tool_choice"] = {"type": "tool", "name": JSON_TOOL}
        if tdefs:
            body["tools"] = tdefs
        return body

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            import requests
        except ImportError as e:  # not a hard dependency of cellwarden (pip install requests / cellwarden[llm])
            raise ProviderError("the 'requests' package is needed for LLM providers: pip install requests") from e

        url = f"{self.base_url}/v1/messages"
        last: str = "no attempt"
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(url, headers=self._headers(), json=body, timeout=self.timeout)
            except requests.RequestException as e:
                last = f"connection error: {type(e).__name__}"
            else:
                if r.status_code == 200:
                    try:
                        return r.json()
                    except ValueError as e:
                        raise ProviderError(f"anthropic: response is not JSON: {e}") from e
                detail = ""
                try:
                    j = r.json()
                    detail = str((j.get("error") or {}).get("message", "")) if isinstance(j, dict) else str(j)[:200]
                except ValueError:
                    detail = r.text[:200]
                last = f"HTTP {r.status_code}: {detail}"
                if r.status_code not in (408, 409, 429) and r.status_code < 500:
                    raise ProviderError(f"anthropic: {last}")
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 20))
        raise ProviderError(f"anthropic: giving up after {self.max_retries + 1} attempts ({last})")

    def complete(self, *, system, messages, tools, max_tokens, json_schema=None) -> Response:
        body = self._payload(system, messages, tools, max_tokens, json_schema)
        data = self._post(body)
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        parsed: Any = None
        for block in data.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text") or "")
            elif block.get("type") == "tool_use":
                args = block.get("input") or {}
                if block.get("name") == JSON_TOOL and json_schema is not None:
                    parsed = args
                    text_parts.append(json.dumps(args))
                else:
                    calls.append(ToolCall(str(block.get("id")), str(block.get("name")), dict(args)))
        u = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(u.get("input_tokens", 0) or 0) + int(u.get("cache_creation_input_tokens", 0) or 0),
            output_tokens=int(u.get("output_tokens", 0) or 0),
            cached_tokens=int(u.get("cache_read_input_tokens", 0) or 0),
        )
        stop = str(data.get("stop_reason") or "end_turn")
        if stop == "tool_use" and not calls:
            stop = "end_turn"  # the only tool call was the structured-output tool
        return Response(text="".join(text_parts), tool_calls=calls, usage=usage, stop_reason=stop,
                        json=parsed, model=str(data.get("model") or self.model), raw=data)
