"""Named LLM providers (anthropic, openai, nvidia), `auto` key detection and the run's recorded choice."""

from __future__ import annotations

import json

import pytest
import yaml

from cellwarden import config as C
from cellwarden import runner as R
from cellwarden.llm import provider as P

KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NVIDIA_API_KEY")


@pytest.fixture
def no_keys(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


@pytest.mark.parametrize("env,expected", [
    ({}, None),
    ({"NVIDIA_API_KEY": "n"}, "nvidia"),
    ({"OPENAI_API_KEY": "o", "NVIDIA_API_KEY": "n"}, "openai"),
    ({"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o", "NVIDIA_API_KEY": "n"}, "anthropic"),
])
def test_auto_order_claude_openai_nvidia(env, expected):
    assert P.auto_provider(env) == expected


def test_presets_fill_only_what_is_unset(no_keys):
    s = P.role_settings({"default": {"provider": "nvidia"}}, "qc_worker")
    assert s["base_url"] == "https://integrate.api.nvidia.com/v1" and s["api_key_env"] == "NVIDIA_API_KEY"
    assert s["model"] == "nvidia/nemotron-3-ultra-550b-a55b" and s["json_mode"] == "object"
    s = P.role_settings({"default": {"provider": "openai", "model": "gpt-4.1"}}, "qc_worker")
    assert s["model"] == "gpt-4.1" and s["base_url"] == "https://api.openai.com/v1"
    s = P.role_settings({"default": {"provider": "openai_compatible", "base_url": "http://localhost:8000/v1",
                                     "model": "qwen"}}, "x")
    assert s.get("api_key_env") is None  # a local server without a key stays keyless


def test_role_switching_provider_takes_that_providers_defaults(no_keys):
    llm = {"default": {"provider": "anthropic"}, "roles": {"reviewer": {"provider": "nvidia"}}}
    worker, reviewer = P.role_settings(llm, "qc_worker"), P.role_settings(llm, "reviewer")
    assert worker["model"] == "claude-haiku-4-5" and worker["api_key_env"] == "ANTHROPIC_API_KEY"
    assert reviewer["model"] == "nvidia/nemotron-3-ultra-550b-a55b" and reviewer["api_key_env"] == "NVIDIA_API_KEY"


def test_get_provider_builds_openai_adapter_for_nvidia(no_keys):
    no_keys.setenv("NVIDIA_API_KEY", "secret")
    prov = P.get_provider({"default": {"provider": "nvidia"}}, "qc_worker")
    assert type(prov).__name__ == "OpenAICompatibleProvider"
    assert prov.base_url == "https://integrate.api.nvidia.com/v1" and prov.model == "nvidia/nemotron-3-ultra-550b-a55b"


def test_nvidia_request_payload(no_keys, monkeypatch):
    no_keys.setenv("NVIDIA_API_KEY", "secret")
    sent = {}

    class Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2}}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, headers=headers, body=json)
        return Resp()

    monkeypatch.setattr("requests.post", fake_post)
    prov = P.get_provider({"default": {"provider": "nvidia"}}, "reviewer")
    r = prov.complete(system="s", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=64,
                      json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}})
    assert sent["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert sent["headers"]["authorization"] == "Bearer secret"
    assert sent["body"]["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert sent["body"]["response_format"]["type"] == "json_object"
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": False}  # Nemotron answers without reasoning
    assert r.json == {"ok": True}


def test_thinking_switch_only_for_nemotron_and_overridable(no_keys):
    s = P.role_settings({"default": {"provider": "nvidia", "model": "minimaxai/minimax-m3.1-flash"}}, "reviewer")
    assert "extra_body" not in s
    s = P.role_settings({"default": {"provider": "openai_compatible", "model": "nemotron-local"}}, "reviewer")
    assert "extra_body" not in s  # only on NVIDIA's endpoint, where the template switch is known
    s = P.role_settings({"default": {"provider": "nvidia"},
                         "roles": {"reviewer": {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}}},
                        "reviewer")
    assert s["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_retired_model_error_says_why(no_keys, monkeypatch):
    no_keys.setenv("NVIDIA_API_KEY", "secret")

    class Gone:
        status_code = 410
        text = ""

        def json(self):
            return {"status": 410, "title": "Gone", "detail": "The model 'x' has reached its end of life"}

    monkeypatch.setattr("requests.post", lambda *a, **k: Gone())
    prov = P.get_provider({"default": {"provider": "nvidia", "model": "x"}}, "reviewer")
    with pytest.raises(P.ProviderError, match="HTTP 410: The model 'x' has reached its end of life"):
        prov.complete(system="s", messages=[{"role": "user", "content": "hi"}], tools=[], max_tokens=8)


def test_role_max_tokens_reaches_the_agent(no_keys):
    from cellwarden.agents.reviewer import Reviewer

    llm = {"default": {"provider": "mock", "model": "mock-model"}, "roles": {"reviewer": {"max_tokens": 4096}}}
    assert Reviewer(P.get_provider(llm, "reviewer")).max_tokens == 4096
    assert Reviewer(P.get_provider(llm, "qc_worker")).max_tokens == 1024  # class default without a setting
    assert Reviewer(P.get_provider(llm, "reviewer"), max_tokens=512).max_tokens == 512  # explicit argument wins


def _project(tmp_path, llm=None):
    data = tmp_path / "d"
    (data).mkdir()
    (data / "matrix.mtx.gz").write_text("x")
    (tmp_path / "samples.csv").write_text(f"sample_id,path\nA,{data}\n")
    cfg = {"run_dir": str(tmp_path / "run"), "samples": "samples.csv"}
    if llm is not None:
        cfg["llm"] = llm
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_no_llm_section_and_no_key_means_rules_only(tmp_path, no_keys):
    cfg = C.load_config(_project(tmp_path))
    assert cfg.get("llm") is None and "rules-only" in cfg["llm_choice"]


def test_no_llm_section_with_a_key_picks_it(tmp_path, no_keys):
    no_keys.setenv("OPENAI_API_KEY", "o")
    no_keys.setenv("NVIDIA_API_KEY", "n")
    cfg = C.load_config(_project(tmp_path))
    assert cfg["llm"]["default"]["provider"] == "openai" and "OPENAI_API_KEY" in cfg["llm_choice"]
    assert C.llm_settings(cfg)["budget"]["per_run_limit"] == 5


def test_explicit_provider_wins_over_keys_and_cli_overrides(tmp_path, no_keys):
    no_keys.setenv("ANTHROPIC_API_KEY", "a")
    p = _project(tmp_path, llm={"default": {"provider": "nvidia"}})
    assert C.load_config(p)["llm"]["default"]["provider"] == "nvidia"
    cfg = C.load_config(p, llm_override={"provider": "openai", "model": "gpt-4.1"})
    assert cfg["llm"]["default"] == {"provider": "openai", "model": "gpt-4.1"}
    assert C.load_config(p, llm_override={"provider": "none"})["llm"] is None


def test_role_max_tokens_and_extra_body_are_validated(tmp_path, no_keys):
    ok = {"default": {"provider": "nvidia", "extra_body": {"top_p": 0.9}},
          "roles": {"reviewer": {"max_tokens": 4096}}}
    assert C.load_config(_project(tmp_path, llm=ok))["llm"]["roles"]["reviewer"]["max_tokens"] == 4096
    bad = {"default": {"provider": "nvidia", "extra_body": "thinking off"}, "roles": {"reviewer": {"max_tokens": 0}}}
    (tmp_path / "b").mkdir()
    with pytest.raises(C.ConfigError) as e:
        C.load_config(_project(tmp_path / "b", llm=bad))
    assert "llm.default.extra_body must be a mapping" in str(e.value)
    assert "llm.roles.reviewer.max_tokens must be a positive number" in str(e.value)


def test_auto_without_key_warns(tmp_path, no_keys):
    cfg = C.load_config(_project(tmp_path, llm={"default": {"provider": "auto"}}))
    assert cfg["llm"] is None and any("found no API key" in w for w in cfg["_warnings"])


def test_choice_is_recorded_in_resolved_config(tmp_path, no_keys):
    no_keys.setenv("NVIDIA_API_KEY", "n")
    cfg = C.load_config(_project(tmp_path))
    res = C.prepare_run_dir(cfg)
    text = (tmp_path / "run" / "config" / "config.resolved.yaml").read_text()
    assert res["llm"]["default"]["provider"] == "nvidia" and "nvidia" in text
    assert "n\n" not in text.split("llm_choice")[0][-5:]  # the key itself is never written


def test_cli_llm_flags(tmp_path, no_keys):
    a = R._parser().parse_args(["run", "--config", str(_project(tmp_path)), "--llm", "nvidia", "--llm-model", "m"])
    assert a.llm == "nvidia" and a.llm_model == "m"


def test_doctor_reports_choice(tmp_path, no_keys):
    no_keys.setenv("NVIDIA_API_KEY", "n")
    rep = R.doctor(str(_project(tmp_path)), None)
    rows = {c["name"]: c for c in rep["checks"] if c.get("group") == "llm" or c.get("section") == "llm"}
    text = json.dumps(rep)
    assert "nvidia" in text and "nvidia/nemotron-3-ultra-550b-a55b" in text and "NVIDIA_API_KEY" in text
