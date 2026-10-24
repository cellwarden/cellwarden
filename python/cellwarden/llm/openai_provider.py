"""OpenAI-compatible ``chat/completions`` adapter (OpenAI, vLLM, Ollama, ...).

POST ``{base_url}/chat/completions`` with tools as
``{"type": "function", "function": {...}}``. The internal (Anthropic-shaped)
conversation is translated: ``tool_use`` blocks become the assistant's
``tool_calls`` and ``tool_result`` blocks become ``role: tool`` messages.
Structured output uses ``response_format: json_schema`` when the server
supports it (``json_mode: schema``, the default) or plain ``json_object``
mode with the schema pasted into the system prompt (``json_mode: object``,
for servers that reject ``json_schema``).

``api_key_env`` is optional: local servers usually need none. The default
``base_url`` is ``https://api.openai.com/v1``.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .provider import Provider, ProviderError, Response, ToolCall, Usage, parse_json_text

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the internal conversation into chat/completions messages."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        texts: list[str] = []
        calls: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for b in content or []:
            t = b.get("type")
            if t == "text":
                texts.append(b.get("text") or "")
            elif t == "tool_use":
                calls.append({"id": b["id"], "type": "function",
                              "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}})
            elif t == "tool_result":
                c = b.get("content")
                results.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                "content": c if isinstance(c, str) else json.dumps(c)})
        if m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            out.extend(results)
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
    return out


class OpenAICompatibleProvider(Provider):
    name = "openai_compatible"

    def __init__(self, settings):
        super().__init__(settings)
        self.base_url = str(settings.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self.json_mode = str(settings.get("json_mode") or "schema")

    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self._api_key:
            h["authorization"] = f"Bearer {self._api_key}"
        return h

    def _payload(self, system, messages, tools, max_tokens, json_schema) -> dict[str, Any]:
        if json_schema is not None and self.json_mode == "object":
            system = system + "\n\nAnswer with one JSON object matching this schema:\n" + json.dumps(json_schema)
        body: dict[str, Any] = {"model": self.model, "max_tokens": int(max_tokens),
                                "messages": to_openai_messages(system, messages)}
        if "temperature" in self.settings:
            body["temperature"] = self.settings["temperature"]
        # extra request fields the endpoint understands (e.g. chat_template_kwargs on NIM)
        body.update(self.settings.get("extra_body") or {})
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                               "parameters": t.get("input_schema") or {"type": "object"}}}
                             for t in tools]
        if json_schema is not None:
            if self.json_mode == "object":
                body["response_format"] = {"type": "json_object"}
            else:
                body["response_format"] = {"type": "json_schema",
                                           "json_schema": {"name": "answer", "schema": json_schema, "strict": False}}
        return body

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            import requests
        except ImportError as e:  # not a hard dependency of cellwarden (pip install requests / cellwarden[llm])
            raise ProviderError("the 'requests' package is needed for LLM providers: pip install requests") from e

        url = f"{self.base_url}/chat/completions"
        last = "no attempt"
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
                        raise ProviderError(f"openai_compatible: response is not JSON: {e}") from e
                try:
                    j = r.json()
                    # OpenAI style {"error": {"message"}}; NIM answers {"detail"} (e.g. a retired model's 410)
                    detail = (str((j.get("error") or {}).get("message", "") or j.get("detail", ""))[:200]
                              if isinstance(j, dict) else str(j)[:200])
                except ValueError:
                    detail = r.text[:200]
                last = f"HTTP {r.status_code}: {detail}"
                if r.status_code not in (408, 409, 429) and r.status_code < 500:
                    raise ProviderError(f"openai_compatible: {last}")
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 20))
        raise ProviderError(f"openai_compatible: giving up after {self.max_retries + 1} attempts ({last})")

    def complete(self, *, system, messages, tools, max_tokens, json_schema=None) -> Response:
        data = self._post(self._payload(system, messages, tools, max_tokens, json_schema))
        try:
            choice = data["choices"][0]
            msg = choice.get("message") or {}
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"openai_compatible: unexpected response shape: {e}") from e
        text = msg.get("content") or ""
        calls = []
        for i, c in enumerate(msg.get("tool_calls") or []):
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments")}
            calls.append(ToolCall(str(c.get("id") or f"call_{i}"), str(fn.get("name")), args if isinstance(args, dict) else {"value": args}))
        u = data.get("usage") or {}
        cached = int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        usage = Usage(input_tokens=max(0, int(u.get("prompt_tokens", 0) or 0) - cached),
                      output_tokens=int(u.get("completion_tokens", 0) or 0), cached_tokens=cached)
        finish = str(choice.get("finish_reason") or "stop")
        stop = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}.get(finish, finish)
        if calls and stop == "end_turn":
            stop = "tool_use"
        parsed = parse_json_text(text) if json_schema is not None else None
        return Response(text=text, tool_calls=calls, usage=usage, stop_reason=stop, json=parsed,
                        model=str(data.get("model") or self.model), raw=data)
