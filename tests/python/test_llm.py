"""Tests for the LLM adapter, budget ledger, agent log and agent base class (docs/agents-spec.md 3).

All offline: HTTP providers are exercised against a fake ``requests.post``;
the ``mock`` provider scripts everything else.
"""

from __future__ import annotations

import json
import time

import pytest

from cellwarden.agents.base import Agent, Tool, load_prompt, wrap_data
from cellwarden.llm import (AgentLog, BudgetExceeded, Ledger, Provider, ProviderError, Response, ToolCall, Usage,
                        get_provider, load_prices, price_for, role_settings, sha256_text)
from cellwarden.llm import anthropic_provider as AP
from cellwarden.llm import openai_provider as OP
from cellwarden.llm.mock_provider import MockProvider, MockScriptExhausted
from cellwarden.llm.provider import JSON_TOOL, parse_json_text

KEY = "sk-ant-test-secret-0123456789"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class FakeResp:
    def __init__(self, status: int, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def fake_post(answers):
    """``requests.post`` replacement returning the queued answers; records every request."""
    seen = []

    def post(url, headers=None, json=None, timeout=None):
        seen.append({"url": url, "headers": headers, "body": json, "timeout": timeout})
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    post.seen = seen
    return post


def mock_cfg(role, script, **budget):
    cfg = {"default": {"provider": "mock", "model": "mock-model"}, "roles": {role: {"script": script}}}
    if budget:
        cfg["budget"] = budget
    return cfg


# ---------------------------------------------------------------------------
# provider selection and settings
# ---------------------------------------------------------------------------


def test_role_settings_layers_roles_over_default_and_flat_shorthand():
    cfg = {"default": {"provider": "openai_compatible", "model": "m0", "base_url": "http://x"},
           "roles": {"reviewer": {"model": "m1", "provider": None}}}
    assert role_settings(cfg, "qc_worker") == {"provider": "openai_compatible", "model": "m0", "base_url": "http://x"}
    assert role_settings(cfg, "reviewer")["model"] == "m1" and role_settings(cfg, "reviewer")["provider"] == "openai_compatible"
    # defaults: anthropic + claude-haiku-4-5 + ANTHROPIC_API_KEY
    s = role_settings({}, "x")
    assert s == {"provider": "anthropic", "model": "claude-haiku-4-5", "api_key_env": "ANTHROPIC_API_KEY"}
    # flat shorthand (llm: {provider: mock}) is llm.default
    assert role_settings({"provider": "mock", "model": "flat"}, "x")["model"] == "flat"
    assert isinstance(get_provider({"provider": "mock"}, "x"), MockProvider)


def test_get_provider_kinds(monkeypatch):
    monkeypatch.setenv("MY_KEY", KEY)
    p = get_provider({"default": {"provider": "anthropic", "model": "claude-haiku-4-5", "api_key_env": "MY_KEY"}}, "w")
    assert isinstance(p, AP.AnthropicProvider) and p.has_api_key
    assert KEY not in repr(p) and KEY not in str(p.settings)  # the key lives only on the instance
    assert isinstance(get_provider({"default": {"provider": "openai_compatible", "model": "llama"}}, "w"), OP.OpenAICompatibleProvider)
    for kind in ("bedrock", "vertex"):
        with pytest.raises(NotImplementedError, match=kind):
            get_provider({"default": {"provider": kind}}, "w")
    with pytest.raises(ProviderError, match="unknown llm provider"):
        get_provider({"default": {"provider": "carrier-pigeon"}}, "w")


def test_anthropic_requires_the_key_and_never_prints_it(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        get_provider({"default": {"provider": "anthropic"}}, "w")
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    p = get_provider({}, "w")
    assert KEY not in repr(p)
    with pytest.raises(NotImplementedError):
        Provider({}).complete(system="", messages=[], tools=None, max_tokens=1)


# ---------------------------------------------------------------------------
# anthropic adapter
# ---------------------------------------------------------------------------


def test_anthropic_payload_and_response(monkeypatch):
    monkeypatch.setenv("K", KEY)
    p = AP.AnthropicProvider({"model": "claude-haiku-4-5", "api_key_env": "K", "temperature": 0})
    body = {"id": "msg", "model": "claude-haiku-4-5-20251001", "stop_reason": "tool_use",
            "content": [{"type": "text", "text": "Let me check."},
                        {"type": "tool_use", "id": "tu_1", "name": "simulate_filter", "input": {"thresholds": {"n_mads": 2}}}],
            "usage": {"input_tokens": 100, "output_tokens": 20, "cache_creation_input_tokens": 50, "cache_read_input_tokens": 400}}
    post = fake_post([FakeResp(200, body)])
    monkeypatch.setattr("requests.post", post)
    tools = [{"name": "simulate_filter", "description": "d", "input_schema": {"type": "object"}}]
    r = p.complete(system="SYS", messages=[{"role": "user", "content": "hi"}], tools=tools, max_tokens=99)
    req = post.seen[0]
    assert req["url"] == "https://api.anthropic.com/v1/messages"
    assert req["headers"]["anthropic-version"] == "2023-06-01" and req["headers"]["x-api-key"] == KEY
    assert req["body"]["system"] == [{"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}]
    assert req["body"]["tools"] == tools and req["body"]["max_tokens"] == 99 and req["body"]["temperature"] == 0
    assert r.text == "Let me check." and r.stop_reason == "tool_use"
    assert [t.to_dict() for t in r.tool_calls] == [{"id": "tu_1", "name": "simulate_filter", "arguments": {"thresholds": {"n_mads": 2}}}]
    assert r.usage.to_dict() == {"input_tokens": 150, "output_tokens": 20, "cached_tokens": 400}
    assert r.model == "claude-haiku-4-5-20251001"
    assert r.assistant_content()[1]["type"] == "tool_use"


def test_anthropic_json_schema_forces_the_synthetic_tool(monkeypatch):
    monkeypatch.setenv("K", KEY)
    p = AP.AnthropicProvider({"api_key_env": "K"})
    body = {"stop_reason": "tool_use", "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": [{"type": "tool_use", "id": "x", "name": JSON_TOOL, "input": {"verdict": "approve", "findings": []}}]}
    post = fake_post([FakeResp(200, body)])
    monkeypatch.setattr("requests.post", post)
    schema = {"type": "object", "properties": {"verdict": {"type": "string"}}}
    r = p.complete(system="s", messages=[{"role": "user", "content": "q"}], tools=None, max_tokens=10, json_schema=schema)
    sent = post.seen[0]["body"]
    assert sent["tool_choice"] == {"type": "tool", "name": JSON_TOOL} and sent["tools"][-1]["input_schema"] == schema
    assert r.json == {"verdict": "approve", "findings": []} and json.loads(r.text) == r.json
    assert r.tool_calls == [] and r.stop_reason == "end_turn"


def test_anthropic_retries_then_fails_cleanly(monkeypatch):
    monkeypatch.setenv("K", KEY)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    p = AP.AnthropicProvider({"api_key_env": "K", "max_retries": 2})
    ok = {"stop_reason": "end_turn", "content": [{"type": "text", "text": "fine"}], "usage": {}}
    post = fake_post([FakeResp(429, {"error": {"message": "slow down"}}), FakeResp(500, "oops"), FakeResp(200, ok)])
    monkeypatch.setattr("requests.post", post)
    assert p.complete(system="s", messages=[], tools=None, max_tokens=5).text == "fine"
    assert len(post.seen) == 3
    # a 4xx other than 408/409/429 is not retried; the message never carries the key
    post = fake_post([FakeResp(401, {"error": {"message": "bad key"}})])
    monkeypatch.setattr("requests.post", post)
    with pytest.raises(ProviderError, match="HTTP 401") as e:
        p.complete(system="s", messages=[], tools=None, max_tokens=5)
    assert KEY not in str(e.value) and len(post.seen) == 1
    # retries exhausted
    import requests

    post = fake_post([requests.ConnectionError("x")] * 3)
    monkeypatch.setattr("requests.post", post)
    with pytest.raises(ProviderError, match="giving up after 3 attempts"):
        p.complete(system="s", messages=[], tools=None, max_tokens=5)


# ---------------------------------------------------------------------------
# openai-compatible adapter
# ---------------------------------------------------------------------------


def test_openai_message_translation():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "calling"},
                                          {"type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "{\"ok\": true}", "is_error": False},
                                     {"type": "text", "text": "and then"}]},
    ]
    out = OP.to_openai_messages("SYS", msgs)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "hello"}
    assert out[2]["tool_calls"][0] == {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "{\"ok\": true}"}
    assert out[4] == {"role": "user", "content": "and then"}


def test_openai_complete_with_tools_and_json(monkeypatch):
    p = OP.OpenAICompatibleProvider({"model": "llama3", "base_url": "http://localhost:11434/v1/"})
    body = {"model": "llama3", "choices": [{"finish_reason": "tool_calls", "message": {
        "content": None, "tool_calls": [{"id": "call_1", "function": {"name": "f", "arguments": "{\"x\": 2}"}}]}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 20}}}
    post = fake_post([FakeResp(200, body)])
    monkeypatch.setattr("requests.post", post)
    r = p.complete(system="s", messages=[{"role": "user", "content": "q"}],
                   tools=[{"name": "f", "description": "", "input_schema": {"type": "object"}}], max_tokens=5)
    assert post.seen[0]["url"] == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in post.seen[0]["headers"]  # no key configured: local server
    assert post.seen[0]["body"]["tools"][0]["function"]["name"] == "f"
    assert r.tool_calls[0].arguments == {"x": 2} and r.stop_reason == "tool_use"
    assert r.usage.to_dict() == {"input_tokens": 100, "output_tokens": 7, "cached_tokens": 20}
    # structured output
    body = {"choices": [{"finish_reason": "stop", "message": {"content": "```json\n{\"verdict\": \"approve\"}\n```"}}], "usage": {}}
    post = fake_post([FakeResp(200, body)])
    monkeypatch.setattr("requests.post", post)
    r = p.complete(system="s", messages=[], tools=None, max_tokens=5, json_schema={"type": "object"})
    assert post.seen[0]["body"]["response_format"]["type"] == "json_schema"
    assert r.json == {"verdict": "approve"} and r.stop_reason == "end_turn"


def test_parse_json_text_tolerates_fences_and_prose():
    assert parse_json_text("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert parse_json_text("Sure: {\"a\": [1, 2]} done") == {"a": [1, 2]}
    assert parse_json_text("not json") is None


# ---------------------------------------------------------------------------
# mock provider
# ---------------------------------------------------------------------------


def test_mock_provider_scripts_and_exhausts(tmp_path):
    script = [{"text": "thinking", "tool": "get_qc_summary", "arguments": {}},
              {"tool_calls": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"k": 1}, "id": "idb"}]},
              {"json": {"verdict": "approve"}, "usage": {"input_tokens": 5}}]
    p = get_provider(mock_cfg("w", script), "w")
    r = p.complete(system="s", messages=[], tools=[{"name": "get_qc_summary"}], max_tokens=1)
    assert r.text == "thinking" and r.tool_calls[0].name == "get_qc_summary" and r.stop_reason == "tool_use"
    r = p.complete(system="s", messages=[], tools=None, max_tokens=1)
    assert [t.id for t in r.tool_calls] == ["mock_2_0", "idb"]
    r = p.complete(system="s", messages=[], tools=None, max_tokens=1, json_schema={})
    assert r.json == {"verdict": "approve"} and r.usage.input_tokens == 5 and r.usage.output_tokens == 200
    assert p.remaining == 0 and len(p.calls) == 3 and p.calls[2]["json_schema"] is True
    with pytest.raises(MockScriptExhausted):
        p.complete(system="s", messages=[], tools=None, max_tokens=1)
    # script from a file, and llm.script keyed by role
    f = tmp_path / "s.json"
    f.write_text(json.dumps([{"text": "from file"}]))
    assert get_provider({"default": {"provider": "mock", "script": str(f)}}, "w").complete(
        system="", messages=[], tools=None, max_tokens=1).text == "from file"
    p = get_provider({"default": {"provider": "mock"}, "script": {"w": [{"text": "role"}]}}, "w")
    assert p.complete(system="", messages=[], tools=None, max_tokens=1).text == "role"


def test_response_roundtrip():
    r = Response(text="t", tool_calls=[ToolCall("i", "n", {"a": 1})], usage=Usage(1, 2, 3), stop_reason="tool_use", json=None)
    assert Response.from_dict(json.loads(json.dumps(r.to_dict()))).to_dict() == r.to_dict()


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------


def test_prices_table():
    prices = load_prices()
    assert prices["claude-haiku-4-5"] == {"input": 1.0, "output": 5.0, "cached": 0.1}
    assert price_for(prices, "claude-haiku-4-5-20251001")[1] == "prefix:claude-haiku-4-5"
    assert price_for(prices, "never-heard-of")[1] == "default" and price_for(prices, "never-heard-of")[0]["input"] >= 15


def test_ledger_charges_and_limits(tmp_path):
    total = tmp_path / "home" / "ledger.json"
    led = Ledger(tmp_path / "run" / "ledger.json", run_id="r1", budget={"per_run_limit": 2.0, "total_limit": 3.0},
                 total_path=total)
    assert led.usd_to_gbp == 0.78 and led.warn_at_pct == 80
    cost = led.charge("qc_worker", "claude-haiku-4-5", Usage(1_000_000, 100_000, 1_000_000), gate="G1")
    assert cost == pytest.approx((1.0 + 0.5 + 0.1) * 0.78)
    assert led.run_total() == pytest.approx(cost) and led.grand_total() == pytest.approx(cost)
    run = json.loads((tmp_path / "run" / "ledger.json").read_text())
    assert run["entries"][0]["role"] == "qc_worker" and run["entries"][0]["gate"] == "G1" and run["total_gbp"] == pytest.approx(cost)
    assert json.loads(total.read_text())["runs"] == {"r1": pytest.approx(cost)}
    # hard stop: a call that would cross the per-run limit
    assert led.would_exceed(0, 0) is False
    assert led.would_exceed(100_000, 20_000, "claude-haiku-4-5") is False
    assert led.would_exceed(1_000_000, 0, "claude-haiku-4-5") is True  # 1.248 + 0.78 > 2
    # warn at 80 %
    assert led.warnings() == []
    led.charge("w", "mock-model", Usage(0, 0, 0))
    led.charge("w", "claude-haiku-4-5", Usage(0, 0, 0))
    led.run["total_gbp"] = 1.62
    assert any("run budget" in w and "81%" in w for w in led.warnings())
    st = led.status()
    assert st["run_pct"] == 81.0 and st["n_calls"] == 3 and st["warnings"]


def test_ledger_total_across_runs_without_double_counting(tmp_path):
    total = tmp_path / "ledger.json"
    a = Ledger(tmp_path / "a.json", run_id="a", budget={"total_limit": 1.0}, total_path=total)
    a.charge("w", "claude-haiku-4-5", Usage(400_000, 0, 0))  # 0.312 GBP
    b = Ledger(tmp_path / "b.json", run_id="b", budget={"total_limit": 1.0}, total_path=total)
    b.charge("w", "claude-haiku-4-5", Usage(400_000, 0, 0))
    assert b.grand_total() == pytest.approx(0.624)
    # reopening run "a" replaces its total instead of adding to it
    a2 = Ledger(tmp_path / "a.json", run_id="a", budget={"total_limit": 1.0}, total_path=total)
    assert a2.run_total() == pytest.approx(0.312) and a2.grand_total() == pytest.approx(0.624)
    assert a2.would_exceed(600_000, 0, "claude-haiku-4-5") is True  # 0.624 + 0.468 > 1
    assert a2.would_exceed(100_000, 0, "claude-haiku-4-5") is False


def test_agent_log_append_find(tmp_path):
    log = AgentLog(tmp_path / "llm" / "agent_log.jsonl")
    assert list(log.entries()) == [] and log.find("a", "b") is None
    r = Response(text="x", tool_calls=[ToolCall("1", "t", {})], usage=Usage(1, 2, 0))
    log.append(gate="G1", role="w", model="m", prompt_sha256="p", input_sha256="i", response=r, cost_gbp=0.1)
    log.append(gate="G1", role="w", model="m", prompt_sha256="p", input_sha256="i", response=r, cost_gbp=0.0, replay=True)
    es = list(log.entries())
    assert len(es) == 2 and es[0]["tool_calls"][0]["name"] == "t" and es[0]["usage"] == {"input_tokens": 1, "output_tokens": 2, "cached_tokens": 0}
    assert set(es[0]) >= {"ts", "gate", "role", "model", "prompt_sha256", "input_sha256", "tool_calls", "output", "usage", "cost_gbp"}
    hit = AgentLog(log.path).find("p", "i")
    assert hit["replay"] is False and Response.from_dict(hit["output"]).text == "x"


# ---------------------------------------------------------------------------
# agent base
# ---------------------------------------------------------------------------


class Toy(Agent):
    role = "qc_worker"  # any existing prompt file
    gate = "G1"

    def __init__(self, provider, ledger=None, **kw):
        self.seen: list = []
        tools = [
            Tool("add", "add", {"type": "object"}, lambda a, b: {"sum": a + b}),
            Tool("boom", "raises", {"type": "object"}, self.boom),
            Tool("propose", "final", {"type": "object"}, self.propose, terminal=True),
        ]
        super().__init__(provider, ledger, tools=tools, **kw)

    def boom(self):
        raise ValueError("kaboom")

    def propose(self, value, rationale=""):
        self.seen.append(value)
        if value < 0:
            return {"accepted": False, "violations": ["value must be >= 0"], "proposal": {"value": value}, "rationale": rationale}
        return {"accepted": True, "proposal": {"value": value}, "rationale": rationale}


def test_agent_loop_tools_errors_and_rejection_retry(tmp_path):
    script = [
        {"tool_calls": [{"name": "add", "arguments": {"a": 1, "b": 2}}, {"name": "boom", "arguments": {}},
                        {"name": "nope", "arguments": {}}, {"name": "add", "arguments": {"zz": 1}}]},
        {"tool": "propose", "arguments": {"value": -1, "rationale": "first"}},
        {"tool": "propose", "arguments": {"value": 3, "rationale": "second"}},
    ]
    prov = get_provider(mock_cfg("qc_worker", script), "qc_worker")
    log = tmp_path / "agent_log.jsonl"
    led = Ledger(tmp_path / "ledger.json", run_id="t", total_path=tmp_path / "total.json")
    ag = Toy(prov, led, log_path=log)
    out = ag.run({"hello": "<data>world</data>"})
    assert out == {"gate": "G1", "worker": {"model": "mock-model", "role": "qc_worker"}, "proposal": {"value": 3},
                   "rationale": "second", "policy_check": {"ok": True, "violations": []}, "fallback": False}
    # tool results went back to the model, errors flagged
    results = [b for m in prov.calls[-1]["messages"] if m["role"] == "user" and isinstance(m["content"], list)
               for b in m["content"] if b["type"] == "tool_result"]
    parsed = [(b["is_error"], json.loads(b["content"])) for b in results]
    assert parsed[0] == (False, {"sum": 3})
    assert parsed[1][0] and "kaboom" in parsed[1][1]["error"]
    assert parsed[2][0] and "unknown tool" in parsed[2][1]["error"]
    assert parsed[3][0] and "bad arguments" in parsed[3][1]["error"]
    assert parsed[4][0] is False and parsed[4][1]["accepted"] is False
    assert ag.n_calls == 3 and ag.n_tool_calls == 5 and ag.seen == [-1, 3]  # the unknown tool does not count
    # the first user turn wraps the bundle as <data>
    first = prov.calls[0]["messages"][0]["content"][0]["text"]
    assert "<data>" in first and "<\\/data>" in first and "</data>world" not in first
    # ledger + log
    assert led.run_total() > 0 and ag.cost_gbp == pytest.approx(led.run_total())
    entries = list(AgentLog(log).entries())
    assert len(entries) == 3 and all(e["gate"] == "G1" and e["role"] == "qc_worker" for e in entries)
    assert entries[0]["prompt_sha256"] == entries[1]["prompt_sha256"] and entries[0]["input_sha256"] != entries[1]["input_sha256"]


def test_agent_two_rejections_then_fallback_and_no_proposal():
    script = [{"tool": "propose", "arguments": {"value": -1, "rationale": "a"}},
              {"tool": "propose", "arguments": {"value": -2, "rationale": "b"}},
              {"text": "I give up"}]
    ag = Toy(get_provider(mock_cfg("qc_worker", script), "qc_worker"))
    out = ag.run({})
    assert out["fallback"] is True and out["policy_check"] == {"ok": False, "violations": ["value must be >= 0"]}
    assert out["proposal"] == {"value": -2} and ag.n_calls == 2  # stops after the second rejection
    ag = Toy(get_provider(mock_cfg("qc_worker", [{"text": "nothing to do"}, {"text": "still nothing"}]), "qc_worker"))
    out = ag.run({})
    assert out["fallback"] and out["rationale"] == "still nothing" and "no proposal" in out["policy_check"]["violations"][0]
    # one nudge to use the proposal tool, then it stops
    assert ag.n_calls == 2 and "calling the propose tool" in ag.provider.calls[1]["messages"][-1]["content"][0]["text"]


def test_agent_is_nudged_once_then_proposes():
    script = [{"text": "Blood, probably."}, {"tool": "propose", "arguments": {"value": 1, "rationale": "r"}}]
    ag = Toy(get_provider(mock_cfg("qc_worker", script), "qc_worker"))
    out = ag.run({})
    assert out["fallback"] is False and out["proposal"] == {"value": 1} and ag.n_calls == 2


def test_proposal_tool_stays_after_the_budget_is_spent():
    script = [{"tool_calls": [{"name": "add", "arguments": {"a": 1, "b": 1}}] * 2},
              {"tool": "propose", "arguments": {"value": 2, "rationale": "after the budget"}}]
    ag = Toy(get_provider(mock_cfg("qc_worker", script), "qc_worker"), max_tool_calls=2)
    out = ag.run({})
    assert out["fallback"] is False and out["proposal"] == {"value": 2}
    assert ag.provider.calls[1]["tools"] == ["propose"]  # only the proposal tool is offered


def test_agent_tool_budget_is_capped():
    script = [{"tool_calls": [{"name": "add", "arguments": {"a": 1, "b": 1}}] * 5} for _ in range(3)] + [{"text": "done"}] * 2
    ag = Toy(get_provider(mock_cfg("qc_worker", script), "qc_worker"), max_tool_calls=12)
    prov = ag.provider
    out = ag.run({})
    assert ag.n_tool_calls == 12 and out["fallback"]
    assert prov.calls[-1]["tools"] == ["propose"]  # after the budget only the proposal tool is offered
    errs = [json.loads(b["content"]) for c in prov.calls for m in c["messages"] if isinstance(m["content"], list)
            for b in m["content"] if b.get("type") == "tool_result" and b["is_error"]]
    assert any("tool budget" in e["error"] for e in errs)


def test_agent_budget_guard_makes_no_call(tmp_path):
    prov = get_provider(mock_cfg("qc_worker", [{"text": "never"}]), "qc_worker")
    led = Ledger(tmp_path / "l.json", run_id="t", budget={"per_run_limit": 0.0000001}, total_path=tmp_path / "t.json")
    ag = Toy(prov, led, log_path=tmp_path / "log.jsonl")
    with pytest.raises(BudgetExceeded, match="no call was made"):
        ag.run({})
    assert prov.calls == [] and not (tmp_path / "log.jsonl").exists()


def test_agent_replay_reuses_logged_output(tmp_path):
    script = [{"tool": "add", "arguments": {"a": 2, "b": 2}}, {"tool": "propose", "arguments": {"value": 4, "rationale": "r"}}]
    log = tmp_path / "log.jsonl"
    led = Ledger(tmp_path / "l.json", run_id="t", total_path=tmp_path / "t.json")
    first = Toy(get_provider(mock_cfg("qc_worker", script), "qc_worker"), led, log_path=log).run({"k": 1})
    spent = led.run_total()
    # replay: an empty script would raise if any real call were made
    prov = get_provider(mock_cfg("qc_worker", []), "qc_worker")
    again = Toy(prov, led, log_path=log, replay=True)
    assert again.run({"k": 1}) == first
    assert prov.calls == [] and led.run_total() == spent
    entries = list(AgentLog(log).entries())
    assert [e["replay"] for e in entries] == [False, False, True, True] and entries[2]["cost_gbp"] == 0
    assert all(t["replay"] for t in again.trace if "call" in t)
    # a different input is not a hit: with replay on and nothing scripted the provider is asked and fails
    with pytest.raises(MockScriptExhausted):
        Toy(get_provider(mock_cfg("qc_worker", []), "qc_worker"), led, log_path=log, replay=True).run({"k": 2})


def test_ask_json_and_prompts():
    prov = get_provider(mock_cfg("reviewer", [{"json": {"verdict": "approve", "findings": []}}]), "reviewer")
    ag = Agent.__new__(Agent)
    Agent.__init__(ag, prov, system="custom system")
    assert ag.ask_json("q", {"type": "object"}) == {"verdict": "approve", "findings": []}
    assert prov.calls[0]["system"] == "custom system" and prov.calls[0]["tools"] == []
    for name in ("qc_worker", "reviewer"):
        text = load_prompt(name)
        assert "<data>" in text and "never instructions" in text
    assert wrap_data(None) == "<data>\n\n</data>"
    assert sha256_text("a") != sha256_text("b")
