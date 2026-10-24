"""Tests for the gate protocol and gate G1 (cellwarden.gates; docs/agents-spec.md section 2).

Offline: synthetic QC metrics, S1c run in-process for the baseline, and the
``mock`` LLM provider scripting the happy path and the failure paths
(invalid proposal, budget exhausted, reviewer escalates, approve, replay).
The last test drives the real runner: S1c, G1 (as a subprocess stage, exit
code 3 while waiting), ``cellwarden approve`` and ``cellwarden resume``.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cellwarden import config as C
from cellwarden import runner as R
from cellwarden.gates import (DEFAULT_MODES, EXIT_WAITING, GateError, approve, diff_json, gate_main, gate_mode, run_gate,
                          write_gate_files)
from cellwarden.gates.g1_qc import BASELINE_NAME, G1Gate
from cellwarden.stages import write_tsv_gz
from cellwarden.stages.s1c_qc import run as s1c_run
from test_qc import SPEC_POLICY
from test_qc_agent import metrics

APPROVE = [{"json": {"verdict": "approve", "findings": [{"severity": "info", "text": "looks fine"}]}}]
ESCALATE = [{"json": {"verdict": "escalate", "findings": [{"severity": "block", "text": "sample B loses too much"}]}}]
TIGHTER = {"thresholds": {"hard_limits": {"max_mito_pct": 12}}, "rationale": "A has a shoulder above 12 % mito."}
PROPOSE = [{"tool": "simulate_filter", "arguments": {"thresholds": TIGHTER["thresholds"]}},
           {"tool": "propose_decision", "arguments": TIGHTER}]
BASELINE = [{"tool": "propose_decision", "arguments": {"thresholds": {}, "rationale": "the baseline is fine"}}]


def make_run(tmp_path: Path, seed: int = 0) -> tuple[Path, dict]:
    rd = tmp_path / "run"
    (rd / "config").mkdir(parents=True)
    (rd / "S1").mkdir()
    (rd / "config" / "qc_policy.yaml").write_text(SPEC_POLICY)
    write_tsv_gz(metrics(seed), rd / "S1" / "qc_metrics.tsv.gz")
    cfg = {"run_id": "test-run", "qc_policy": str(rd / "config" / "qc_policy.yaml")}
    s1c_run(rd, cfg)
    return rd, cfg


def llm(worker, reviewer=APPROVE, **more):
    d = {"default": {"provider": "mock", "model": "mock-model"},
         "roles": {"qc_worker": {"script": worker}, "reviewer": {"script": reviewer}}}
    d.update(more)
    return d


def with_llm(cfg, rd, worker, reviewer=APPROVE, mode="auto", **more):
    more.setdefault("budget", {"ledger_path": str(rd / "total_ledger.json")})
    return dict(cfg, llm=llm(worker, reviewer, **more), gates={"G1": mode})


def gate_files(rd, gate="G1"):
    return sorted(p.name for p in (rd / "gates" / gate).iterdir())


def decisions(rd):
    return pd.read_csv(rd / "S1" / "qc_decisions.tsv.gz", sep="\t", dtype=str, keep_default_na=False)


def log_entries(rd):
    p = rd / "llm" / "agent_log.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


# ---------------------------------------------------------------------------
# protocol helpers
# ---------------------------------------------------------------------------


def test_gate_mode_defaults_and_validation():
    assert DEFAULT_MODES == {"G1": "auto", "G2": "review", "G3": "auto", "G4": "review"}
    assert gate_mode({}, "G1") == "auto" and gate_mode({}, "G2") == "review"
    assert gate_mode({"gates": {"G1": "review"}}, "G1") == "review"
    with pytest.raises(GateError, match="gates.G1 must be one of"):
        gate_mode({"gates": {"G1": "maybe"}}, "G1")
    assert EXIT_WAITING == 3


def test_diff_json_and_write_gate_files(tmp_path):
    old = {"thresholds": {"hard_limits": {"max_mito_pct": 25}, "n_mads": 3}, "review_cells": ["a"], "summary": {}}
    new = {"thresholds": {"hard_limits": {"max_mito_pct": 15}, "n_mads": 3, "per_sample": {}}, "review_cells": ["a", "b"]}
    d = diff_json(old, new)
    assert "thresholds.hard_limits.max_mito_pct: 25 -> 15" in d
    assert any(x.startswith("thresholds.per_sample: added") for x in d) and "summary: removed" in d
    assert "review_cells: 1 -> 2 items (+1, -0)" in d
    assert diff_json(old, old) == []
    files = write_gate_files(tmp_path / "G9", {"b": 1}, {"p": 1}, None, {"d": 1}, "<html></html>")
    assert set(files) == {"bundle", "proposal", "decision", "report"} and not (tmp_path / "G9" / "review.json").exists()
    assert json.loads((tmp_path / "G9" / "proposal.json").read_text()) == {"p": 1}


# ---------------------------------------------------------------------------
# G1 without an llm section: rules only, never stops
# ---------------------------------------------------------------------------


def test_g1_rules_only_without_llm(tmp_path):
    rd, cfg = make_run(tmp_path)
    before = (rd / "S1" / "qc_decisions.tsv.gz").read_bytes()
    res = run_gate("G1", rd, cfg)
    assert res.status == "continue" and res.mode == "auto" and not res.waiting and res["continue"] is True
    assert res.decision["decided_by"] == "rules" and res.decision["mode"] == "auto" and res.decision["fallback"] is True
    assert res.proposal["fallback"] and res.proposal["worker"] == {"model": None, "role": "qc_worker"}
    assert res.review["verdict"] == "approve" and res.review["reviewer"] == {"model": "rules"}
    assert gate_files(rd) == ["bundle.json", "decision.json", "proposal.json", "qc_summary.json", "report.html", "review.json"]
    assert not (rd / "llm").exists()
    # the baseline copy is kept and the decisions file is untouched (byte-identical)
    assert (rd / "S1" / BASELINE_NAME).read_bytes() == before
    assert (rd / "S1" / "qc_decisions.tsv.gz").read_bytes() == before
    dec = json.loads((rd / "gates" / "G1" / "decision.json").read_text())
    assert set(dec) >= {"gate", "mode", "decided_by", "decided_utc", "decision", "changes_vs_proposal", "bundle_sha256"}
    assert set(dec["decision"]) == {"thresholds", "review_cells", "summary"} and dec["decision"]["review_cells"] == []
    bundle = json.loads((rd / "gates" / "G1" / "bundle.json").read_text())
    assert "cell_id" not in json.dumps(bundle) and bundle["bundle_sha256"] == dec["bundle_sha256"]
    html = (rd / "gates" / "G1" / "report.html").read_text()
    assert html.lower().startswith("<!doctype html>") and "decided by rules" in html and "QC decisions per sample" in html
    assert "no LLM calls" in html and "http" not in html.split("<body>")[0]  # self-contained: no CDN
    summary = json.loads((rd / "gates" / "G1" / "qc_summary.json").read_text())
    assert summary["decided_by"] == "rules" and summary["counts"]["exclude"] > 0
    # review mode without an llm waits (the baseline needs a human); auto never does
    res = run_gate("G1", rd, dict(cfg, gates={"G1": "review"}))
    assert res.waiting and (rd / "gates" / "G1" / "pending.json").exists() and not (rd / "gates" / "G1" / "decision.json").exists()


def test_g1_rerun_is_byte_identical(tmp_path):
    rd, cfg = make_run(tmp_path)
    run_gate("G1", rd, cfg)
    d1 = (rd / "S1" / "qc_decisions.tsv.gz").read_bytes()
    b1 = (rd / "gates" / "G1" / "bundle.json").read_text()
    run_gate("G1", rd, cfg)
    assert (rd / "S1" / "qc_decisions.tsv.gz").read_bytes() == d1
    assert (rd / "gates" / "G1" / "bundle.json").read_text() == b1


def test_g1_needs_s1c(tmp_path):
    rd = tmp_path / "run"
    (rd / "S1").mkdir(parents=True)
    with pytest.raises(GateError, match="S1a-S1c must run first"):
        G1Gate(rd, {})


# ---------------------------------------------------------------------------
# G1 with the mock provider
# ---------------------------------------------------------------------------


def test_g1_agent_happy_path_auto(tmp_path):
    rd, cfg = make_run(tmp_path)
    baseline_bytes = (rd / "S1" / "qc_decisions.tsv.gz").read_bytes()
    res = run_gate("G1", rd, with_llm(cfg, rd, PROPOSE))
    assert res.status == "continue" and res.decision["decided_by"] == "agent" and res.decision["mode"] == "auto"
    assert res.proposal["fallback"] is False and res.proposal["policy_check"]["ok"]
    assert res.proposal["worker"] == {"model": "mock-model", "role": "qc_worker"} and res.proposal["n_tool_calls"] == 2
    assert res.review["verdict"] == "approve" and res.review["reviewer"]["model"] == "mock-model"
    assert res.decision["decision"]["thresholds"]["hard_limits"]["max_mito_pct"] == 12
    assert res.decision["changes_vs_proposal"] == [] and res.decision["proposal_sha256"]
    # qc_decisions.tsv.gz rewritten through the rules engine, baseline kept
    t = decisions(rd)
    assert (t["decided_by"] == "agent").all() and (t["policy_hash"] == t["policy_hash"].iloc[0]).all()
    assert (rd / "S1" / BASELINE_NAME).read_bytes() == baseline_bytes
    base = pd.read_csv(io.BytesIO(baseline_bytes), sep="\t", dtype=str, keep_default_na=False, compression="gzip")
    assert list(t["cell_id"]) == list(base["cell_id"]) and (t["decision"] != base["decision"]).sum() == 0
    assert (t["reasons"] != base["reasons"]).sum() == 10  # the planted 14 % cells: HIGH_MT instead of ADAPT_HIGH_MT
    # log and ledger
    es = log_entries(rd)
    assert [e["role"] for e in es] == ["qc_worker", "qc_worker", "reviewer"]  # the proposal turn ends the loop
    assert all(e["gate"] == "G1" and e["model"] == "mock-model" and e["cost_gbp"] > 0 for e in es)
    assert es[1]["tool_calls"][0]["name"] == "propose_decision" and "usage" in es[0]
    led = json.loads((rd / "llm" / "ledger.json").read_text())
    assert led["run_id"] == "test-run" and len(led["entries"]) == 3 and led["total_gbp"] == pytest.approx(sum(e["cost_gbp"] for e in es))
    assert json.loads((rd / "total_ledger.json").read_text())["runs"] == {"test-run": pytest.approx(led["total_gbp"])}
    html = (rd / "gates" / "G1" / "report.html").read_text()
    assert "decided by agent" in html and "A has a shoulder" in html and "looks fine" in html and "mock-model" in html
    summary = json.loads((rd / "gates" / "G1" / "qc_summary.json").read_text())
    assert summary["decided_by"] == "agent" and summary["thresholds"]["hard_limits"]["max_mito_pct"] == 12


def test_g1_review_mode_waits_even_when_approved(tmp_path):
    rd, cfg = make_run(tmp_path)
    before = (rd / "S1" / "qc_decisions.tsv.gz").read_bytes()
    res = run_gate("G1", rd, with_llm(cfg, rd, PROPOSE, mode="review"))
    assert res.waiting and res.decision is None and res.review["verdict"] == "approve"
    assert gate_files(rd) == ["bundle.json", "pending.json", "proposal.json", "report.html", "review.json"]
    pending = json.loads((rd / "gates" / "G1" / "pending.json").read_text())
    assert pending["approve"] == f"cellwarden approve --run-dir {rd} --stage qc-review" and pending["mode"] == "review"
    assert (rd / "S1" / "qc_decisions.tsv.gz").read_bytes() == before  # nothing applied while waiting
    assert "waiting for a human" in (rd / "gates" / "G1" / "report.html").read_text()


def test_g1_invalid_proposals_fall_back_to_rules_and_review(tmp_path):
    rd, cfg = make_run(tmp_path)
    bad = {"thresholds": {"hard_limits": {"min_genes": 10}}, "rationale": "loosen"}
    script = [{"tool": "propose_decision", "arguments": bad}, {"tool": "propose_decision", "arguments": dict(bad, rationale="again")}]
    res = run_gate("G1", rd, with_llm(cfg, rd, script))
    assert res.waiting and res.mode == "review" and res.decision is None
    p = json.loads((rd / "gates" / "G1" / "proposal.json").read_text())
    assert p["fallback"] is True and "no valid proposal" in p["fallback_reason"] and "min_genes 10" in p["fallback_reason"]
    assert p["proposal"]["thresholds"]["hard_limits"]["min_genes"] == 200  # the baseline content
    assert res.review["verdict"] == "escalate" and res.review["reviewer"]["model"] == "rules"
    assert any("fallback" in n for n in res.notes)
    assert len(log_entries(rd)) == 2  # two worker turns, the reviewer was never asked


def test_g1_budget_exhausted_makes_no_call(tmp_path):
    rd, cfg = make_run(tmp_path)
    cfg2 = with_llm(cfg, rd, PROPOSE, budget={"per_run_limit": 0.000001, "ledger_path": str(rd / "total.json")})
    res = run_gate("G1", rd, cfg2)
    assert res.waiting and res.mode == "review"
    p = json.loads((rd / "gates" / "G1" / "proposal.json").read_text())
    assert p["fallback"] and p["fallback_reason"].startswith("budget guard") and "no call was made" in p["fallback_reason"]
    assert log_entries(rd) == [] and not (rd / "llm" / "ledger.json").exists()
    assert any("budget" in n for n in res.notes)
    # the total limit across runs is honoured too (an earlier run already spent it)
    rd2, cfg3 = make_run(tmp_path / "second")
    total = rd2 / "total.json"
    total.write_text(json.dumps({"schema": 1, "runs": {"earlier": 29.99}}))
    res = run_gate("G1", rd2, with_llm(cfg3, rd2, PROPOSE, budget={"ledger_path": str(total)}))
    assert res.waiting and "budget guard" in json.loads((rd2 / "gates" / "G1" / "proposal.json").read_text())["fallback_reason"]


def test_g1_budget_warning_at_80_pct(tmp_path):
    rd, cfg = make_run(tmp_path)
    total = rd / "total.json"
    total.write_text(json.dumps({"schema": 1, "runs": {"earlier": 25.0}}))  # 83 % of the 30 GBP default
    res = run_gate("G1", rd, with_llm(cfg, rd, BASELINE, budget={"ledger_path": str(total)}))
    assert res.status == "continue" and any("total budget" in n and "%" in n for n in res.notes)
    assert "total budget" in (rd / "gates" / "G1" / "report.html").read_text()


def test_g1_provider_unavailable_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rd, cfg = make_run(tmp_path)
    cfg2 = dict(cfg, llm={"default": {"provider": "anthropic"}, "budget": {"ledger_path": str(rd / "t.json")}})
    res = run_gate("G1", rd, cfg2)
    assert res.waiting and "provider unavailable" in res.proposal["fallback_reason"] and "ANTHROPIC_API_KEY" in res.proposal["fallback_reason"]
    cfg2["llm"]["default"]["provider"] = "bedrock"
    assert "not implemented" in run_gate("G1", rd, cfg2).proposal["fallback_reason"]


def test_g1_reviewer_request_changes_gives_one_revision(tmp_path):
    rd, cfg = make_run(tmp_path)
    reviewer = [{"json": {"verdict": "request_changes", "findings": [{"severity": "warn", "text": "12 is arbitrary; keep the baseline"}]}},
                APPROVE[0]]
    worker = PROPOSE + BASELINE
    res = run_gate("G1", rd, with_llm(cfg, rd, worker, reviewer))
    assert res.status == "continue" and res.decision["decided_by"] == "agent"
    assert res.proposal["proposal"]["thresholds"]["hard_limits"]["max_mito_pct"] == 25 and res.proposal["rationale"] == "the baseline is fine"
    assert res.review["verdict"] == "approve" and res.review["rounds"][0]["verdict"] == "request_changes"
    es = log_entries(rd)
    assert [e["role"] for e in es] == ["qc_worker", "qc_worker", "reviewer", "qc_worker", "reviewer"]
    assert es[3]["input_sha256"] != es[1]["input_sha256"]  # the revision round is a new turn in the same conversation
    # a second request_changes escalates
    rd2, cfg2 = make_run(tmp_path / "two")
    res = run_gate("G1", rd2, with_llm(cfg2, rd2, PROPOSE + BASELINE, [reviewer[0], reviewer[0]]))
    assert res.waiting and res.review["verdict"] == "escalate" and "requested twice" in res.review["findings"][-1]["text"]


def test_g1_reviewer_escalates_then_human_approves(tmp_path):
    rd, cfg = make_run(tmp_path)
    cfg2 = with_llm(cfg, rd, PROPOSE, ESCALATE)
    res = run_gate("G1", rd, cfg2)
    assert res.waiting and res.decision is None and res.review["verdict"] == "escalate"
    assert any("escalate" in n for n in res.notes)
    # approve as proposed
    dec = approve("G1", rd, cfg2, by="alice")
    assert dec["decided_by"] == "human:alice" and dec["changes_vs_proposal"] == [] and dec["gate"] == "G1"
    assert dec["decision"]["thresholds"]["hard_limits"]["max_mito_pct"] == 12 and dec["mode"] == "auto"
    assert not (rd / "gates" / "G1" / "pending.json").exists()
    # resume: the gate reruns, honours the human decision for this bundle and applies it
    res = run_gate("G1", rd, cfg2)
    assert res.status == "continue" and res.decision["decided_by"] == "human:alice"
    assert len(log_entries(rd)) == 3  # no new LLM calls
    t = decisions(rd)
    assert (t["decided_by"] == "human").all()
    assert json.loads((rd / "gates" / "G1" / "qc_summary.json").read_text())["decided_by"] == "human:alice"
    assert "decided by human:alice" in (rd / "gates" / "G1" / "report.html").read_text()
    # a changed bundle (different metrics) invalidates the human decision: the agents run again
    m = metrics(seed=5)
    write_tsv_gz(m, rd / "S1" / "qc_metrics.tsv.gz")
    s1c_run(rd, cfg)
    res = run_gate("G1", rd, with_llm(cfg, rd, BASELINE, ESCALATE))
    assert res.waiting and res.decision is None and len(log_entries(rd)) == 5


def test_approve_from_file_validates_against_policy(tmp_path):
    rd, cfg = make_run(tmp_path)
    cfg2 = with_llm(cfg, rd, PROPOSE, ESCALATE)
    with pytest.raises(GateError, match="has not run yet"):
        approve("G1", rd, cfg2)
    run_gate("G1", rd, cfg2)
    edited = json.loads((rd / "gates" / "G1" / "proposal.json").read_text())
    edited["proposal"]["thresholds"]["hard_limits"]["max_mito_pct"] = 10
    edited["proposal"]["review_cells"] = ["A_c0000"]
    f = tmp_path / "edited.json"
    f.write_text(json.dumps(edited))
    dec = approve("G1", rd, cfg2, from_file=f, by="bob")
    assert dec["decided_by"] == "human:bob" and dec["from_file"] == str(f)
    assert "thresholds.hard_limits.max_mito_pct: 12.0 -> 10.0" in dec["changes_vs_proposal"]
    assert any(c.startswith("review_cells: 0 -> 1 items") for c in dec["changes_vs_proposal"])
    assert dec["decision"]["summary"]["total"]["review"] == 1  # derived fields recomputed
    res = run_gate("G1", rd, cfg2)
    assert res.status == "continue" and decisions(rd).set_index("cell_id").loc["A_c0000", "decision"] == "review"
    # bare decision content and {"decision": {...}} are accepted too; policy violations are refused
    f.write_text(json.dumps({"decision": {"thresholds": {"hard_limits": {"min_genes": 100}}, "review_cells": []}}))
    with pytest.raises(GateError, match="min_genes 100 is below"):
        approve("G1", rd, cfg2, from_file=f)
    f.write_text("not json")
    with pytest.raises(GateError, match="cannot read"):
        approve("G1", rd, cfg2, from_file=f)
    f.write_text(json.dumps({"thresholds": {"n_mads": 2.5}, "review_cells": []}))
    assert approve("G1", rd, cfg2, from_file=f)["decision"]["thresholds"]["n_mads"] == 2.5


def test_g1_replay_reuses_the_log_without_calls(tmp_path):
    rd, cfg = make_run(tmp_path)
    first = run_gate("G1", rd, with_llm(cfg, rd, PROPOSE))
    spent = json.loads((rd / "llm" / "ledger.json").read_text())["total_gbp"]
    n = len(log_entries(rd))
    # replay with empty scripts: any real call would raise MockScriptExhausted and fall back
    again = run_gate("G1", rd, with_llm(cfg, rd, [], [], replay=True))
    assert again.status == "continue" and again.decision["decided_by"] == "agent"
    assert again.decision["decision"] == first.decision["decision"] and again.review == first.review
    es = log_entries(rd)
    assert len(es) == 2 * n and all(e["replay"] for e in es[n:]) and all(e["cost_gbp"] == 0 for e in es[n:])
    assert json.loads((rd / "llm" / "ledger.json").read_text())["total_gbp"] == spent
    # without replay the empty script is a provider failure: fallback + review
    res = run_gate("G1", rd, with_llm(cfg, rd, [], []))
    assert res.waiting and "provider failure" in res.proposal["fallback_reason"]


def test_g1_escalate_mode_goes_straight_to_a_human(tmp_path):
    rd, cfg = make_run(tmp_path)
    res = run_gate("G1", rd, with_llm(cfg, rd, PROPOSE, mode="escalate"))
    assert res.waiting and res.mode == "escalate" and res.proposal["fallback"] and log_entries(rd) == []
    assert res.review["verdict"] == "escalate"
    dec = approve("G1", rd, with_llm(cfg, rd, PROPOSE, mode="escalate"), by="carol")
    assert dec["decided_by"] == "human:carol" and dec["decision"]["thresholds"]["hard_limits"]["max_mito_pct"] == 25


def test_gate_main_exit_codes(tmp_path):
    rd, cfg = make_run(tmp_path)
    (rd / "config" / "config.resolved.yaml").write_text(yaml.safe_dump(with_llm(cfg, rd, PROPOSE, ESCALATE)))
    assert gate_main("G1", ["--run-dir", str(rd)]) == EXIT_WAITING
    (rd / "config" / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    assert gate_main("G1", ["--run-dir", str(rd)]) == 0
    (rd / "S1" / "qc_metrics.tsv.gz").unlink()
    assert gate_main("G1", ["--run-dir", str(rd)]) == 1


# ---------------------------------------------------------------------------
# runner integration: exit 3, cellwarden approve, cellwarden resume, fingerprints
# ---------------------------------------------------------------------------


def test_runner_stops_at_g1_and_resumes_after_approve(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    for s in ("A", "B"):
        (proj / f"{s}.h5ad").write_text("placeholder")
    (proj / "samples.csv").write_text("sample_id,path\nA,A.h5ad\nB,B.h5ad\n")
    (proj / "qc_policy.yaml").write_text(SPEC_POLICY)
    raw = {"project": "gates", "run_dir": "run", "samples": "samples.csv", "qc_policy": "qc_policy.yaml",
           "resources": {"threads": 2, "max_memory_gb": 8, "gpu_mem_cap_gb": 4, "nice": 0},
           "llm": llm(PROPOSE, ESCALATE, budget={"ledger_path": str(tmp_path / "total.json")})}
    (proj / "config.yaml").write_text(yaml.safe_dump(raw))
    res = C.prepare_run_dir(C.load_config(proj / "config.yaml"))
    rd = Path(res["run_dir"])
    (rd / "S1").mkdir()
    write_tsv_gz(metrics(), rd / "S1" / "qc_metrics.tsv.gz")

    def dummy(sid, outputs):
        code = "import os,pathlib\nrd=pathlib.Path(os.environ['CELLWARDEN_RUN_DIR'])\n"
        for o in outputs:
            code += f"p=rd/{o!r}; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('x')\n"
        return lambda ctx: [sys.executable, "-c", code]

    real = {s.id: s for s in R.default_stages()}
    stages = [real["S1c"], real["G1"],
              R.Stage("S2", "R", dummy("S2", ["S2/object.rds", "S2/summary.json"]), real["S2"].config_keys,
                      ("S1/qc_decisions.tsv.gz",), real["S2"].outputs)]
    out = io.StringIO()
    assert R.main(["run", "--config", str(proj / "config.yaml")], stages=stages, out=out) == 3, out.getvalue()
    text = out.getvalue()
    assert "G1   QC review — WAITING" in text and f"cellwarden approve --run-dir {rd} --stage qc-review" in text
    st = json.loads((rd / "state.json").read_text())
    assert st["stages"]["S1c"]["status"] == "done" and st["stages"]["G1"]["status"] == "waiting" and st["stages"]["G1"]["exit_code"] == 3
    assert st["stages"]["S2"]["status"] == "pending" and not (rd / "S2").exists()
    assert (rd / "gates" / "G1" / "pending.json").exists() and "G1: waiting for human approval" in (rd / "logs" / "G1.log").read_text()
    out = io.StringIO()
    assert R.main(["status", "--run-dir", str(rd)], out=out) == 0 and "waiting for a human decision" in out.getvalue()
    # resume without approving: still waiting
    out = io.StringIO()
    assert R.main(["resume", "--run-dir", str(rd)], stages=stages, out=out) == 3
    # approve, then resume
    out = io.StringIO()
    assert R.main(["approve", "--run-dir", str(rd), "--stage", "G1", "--by", "dana"], out=out) == 0, out.getvalue()
    assert "approved by human:dana" in out.getvalue() and "cellwarden resume" in out.getvalue()
    out = io.StringIO()
    assert R.main(["resume", "--run-dir", str(rd)], stages=stages, out=out) == 0, out.getvalue()
    st = json.loads((rd / "state.json").read_text())
    assert st["stages"]["G1"]["status"] == "done" and st["stages"]["S2"]["status"] == "done"
    assert json.loads((rd / "gates" / "G1" / "decision.json").read_text())["decided_by"] == "human:dana"
    assert (decisions(rd)["decided_by"] == "human").all()
    # S2's recorded input is the G1-rewritten file; a rerun skips everything
    assert st["stages"]["S2"]["fingerprint_parts"]["inputs"]["S1/qc_decisions.tsv.gz"] == st["stages"]["G1"]["outputs"]["S1/qc_decisions.tsv.gz"]
    out = io.StringIO()
    assert R.main(["run", "--config", str(proj / "config.yaml")], stages=stages, out=out) == 0
    assert out.getvalue().count("— skipped (") == 3 and "nothing to do" in out.getvalue()
    # a bad approve is refused with exit 2
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"thresholds": {"hard_limits": {"max_mito_pct": 99}}, "review_cells": []}))
    out = io.StringIO()
    assert R.main(["approve", "--run-dir", str(rd), "--stage", "G1", "--from-file", str(f)], out=out) == 2
    assert "violates the policy" in out.getvalue()
    assert R.main(["approve", "--run-dir", str(rd), "--stage", "G7"], out=io.StringIO()) == 2
