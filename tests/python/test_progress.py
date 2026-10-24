"""Progress output: plain lines, step parsing, time-left estimates, the live display."""

import io
import json
from pathlib import Path

import pytest

from cellwarden import progress as P


def test_parse_step_from_r_and_python_logs():
    assert P.parse_step("2026-10-23T13:54:29Z [INFO] start: harmony") == "harmony"
    assert P.parse_step("2026-10-23T13:54:29Z [INFO] done:  harmony (0.24 s)") is None
    assert P.parse_step("[16:40:01] G2: bundle built (sha d887b58), mode auto") == "G2: bundle built (sha d887b58), mode auto"
    assert P.parse_step("[16:40:01] S1b: start (run dir /x)") is None
    assert P.parse_step("[16:40:09] S1b: done in 4.1s") is None
    assert P.parse_step("# cellwarden: started 2026-10-23T16:06:35Z") is None
    assert P.parse_step("Warning message: something") is None
    assert len(P.parse_step("[10:00:00] " + "x" * 200)) == 90


def test_log_follower_reads_only_new_complete_lines(tmp_path):
    log = tmp_path / "S2.log"
    log.write_text("# cellwarden: started\n")
    f = P.LogFollower(log)
    assert f.latest_step() is None
    with open(log, "a") as fh:
        fh.write("t [INFO] start: pca\nt [INFO] start: harm")
    assert f.latest_step() == "pca"  # the half-written line waits for its end
    with open(log, "a") as fh:
        fh.write("ony\n")
    assert f.latest_step() == "harmony" and f.latest_step() is None


def _run_dir(tmp_path, cells=2700):
    (tmp_path / "S0").mkdir()
    (tmp_path / "S0" / "ingest_report.json").write_text(json.dumps({"samples": [{"cells": cells}]}))
    return tmp_path


def test_estimator_scales_with_cells_and_learns(tmp_path):
    hist = tmp_path / "hist.json"
    e = P.Estimator(tmp_path / "none", {}, history_path=hist)
    assert e.predict("S2") is None and e.predict("G1") == P.GATE_SECONDS_RULES  # cells unknown before ingest
    assert e.remaining(["S2"]) is None
    rd = _run_dir(tmp_path)
    e = P.Estimator(rd, {"llm": {"default": {}}}, history_path=hist)
    fixed, per = P.STAGE_MODEL["S2"]
    assert e.predict("S2") == pytest.approx(fixed + per * 2.7)
    assert e.predict("G2") == P.GATE_SECONDS_LLM["G2"]
    assert e.remaining(["S3"], current="S2", elapsed=5) == pytest.approx(e.predict("S2") - 5 + e.predict("S3"))
    assert e.predict("X1") is None and e.remaining(["X1"]) is None
    e.record("S2", 2 * e.predict("S2"))  # this machine is twice as slow
    e2 = P.Estimator(rd, {"llm": {"default": {}}}, history_path=hist)
    assert e2.predict("S2") == pytest.approx(2 * (fixed + per * 2.7))
    assert json.loads(hist.read_text())["S2"] == [2.0]


def test_estimator_keeps_a_short_history(tmp_path):
    hist = tmp_path / "h.json"
    e = P.Estimator(_run_dir(tmp_path), {}, history_path=hist)
    for _ in range(P.HISTORY_KEEP + 5):
        e.record("S0", 1.0)
    assert len(json.loads(hist.read_text())["S0"]) == P.HISTORY_KEEP
    hist.write_text("not json")
    assert P.Estimator(tmp_path, {}, history_path=hist).history == {}


def test_plain_reporter_lines():
    out = io.StringIO()
    r = P.make_reporter("auto", out, 12)  # a StringIO is not a terminal
    assert type(r) is P.PlainReporter
    r.run_start("run1", Path("/r"), ["S0"])
    r.skip(0, "S0", "done, inputs unchanged")
    r.start(5, "S2", "not run yet")
    r.step("harmony")
    r.done(5, "S2", 32.4, 600 * 1024**2)
    r.waiting(7, "G2", 97, Path("/r/gates/G2/report.html"), "cellwarden approve --run-dir /r --stage context-review",
              "cellwarden resume --run-dir /r")
    r.failed(8, "G3", 1, 2.0, Path("/r/logs/G3.log"), ["boom"], "cellwarden resume --run-dir /r")
    text = out.getvalue()
    assert "  [ 1/12] S0   Ingest — skipped (done, inputs unchanged)" in text
    assert "  [ 6/12] S2   Normalise, PCA, batch correction, clustering — running (not run yet)" in text
    assert "— done in 32.4s, peak RSS 600.00 MB" in text and "harmony" not in text
    assert "[ 8/12] G2   Tissue and disease context review — WAITING for your decision after 1m37s" in text
    assert "approve:  cellwarden approve --run-dir /r --stage context-review" in text
    assert "— FAILED (exit code 1) after 2.0s" in text and "         | boom" in text
    assert "\x1b" not in text  # no terminal control codes in logs


def test_live_reporter_renders(tmp_path):
    pytest.importorskip("rich")
    out = io.StringIO()
    r = P.make_reporter("live", out, 3, P.Estimator(_run_dir(tmp_path), {}, history_path=tmp_path / "h.json"))
    assert type(r) is P.LiveReporter
    r.run_start("run1", tmp_path, ["S0", "S1a", "S2"])
    r.done(0, "S0", 9.8, 1024**2)
    r.start(1, "S1a", "upstream S0 was rerun")
    r.step("metrics")
    from rich.console import Console

    c = Console(file=io.StringIO(), width=100)
    c.print(r._render())
    shown = c.file.getvalue()
    assert "QC metrics" in shown and "metrics" in shown and "1/3 stages" in shown and "left" in shown
    r.done(1, "S1a", 30, None)
    r.close()
    text = out.getvalue()
    assert "Ingest" in text and "QC metrics" in text


def test_make_reporter_rejects_unknown_mode():
    with pytest.raises(ValueError):
        P.make_reporter("fancy", io.StringIO(), 1)


def test_fmt_eta():
    assert P.fmt_eta(30) == "under a minute" and P.fmt_eta(185) == "about 3 min" and P.fmt_eta(3 * 3600 + 600) == "about 3 h 10 min"
