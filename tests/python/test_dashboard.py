"""cellwarden.dashboard: renders from state.json + summaries, never fails on partial runs."""
from __future__ import annotations

import json
from pathlib import Path

from cellwarden import dashboard


def _state(run: Path, stages: dict) -> None:
    (run / "state.json").write_text(json.dumps({
        "schema": 1, "run_id": "r1", "stage_order": dashboard.DEFAULT_ORDER, "stages": stages,
        "updated_utc": "2026-10-17T10:00:10Z"}))


def test_empty_run_dir(tmp_path):
    p = dashboard.build(tmp_path)
    html = p.read_text()
    assert p.name == "dashboard.html" and "PENDING" in html and "No stage has started" in html


def test_partial_run_with_waiting_gate_and_failed_stage(tmp_path):
    run = tmp_path
    (run / "logs").mkdir()
    (run / "logs" / "S2.log").write_text("line1\nError: boom\n")
    (run / "gates" / "G1").mkdir(parents=True)
    (run / "gates" / "G1" / "pending.json").write_text("{}")
    (run / "gates" / "G1" / "report.html").write_text("<html></html>")
    (run / "S2").mkdir()
    (run / "S2" / "summary.json").write_text(json.dumps({
        "path": "cpu_scfast", "engine": "scfast", "scfast_version": "0.1.0", "device": "cpu", "plan": "cpu",
        "cells_in": 100, "cells_out": 90, "n_hvg": 50, "pcs": 10, "clusters": 3,
        "integration": {"method": "none"}, "timings_s": {"pca": 1.0, "umap": 3.0}}))
    _state(run, {
        "S0": {"status": "done", "started_utc": "2026-10-17T10:00:00Z", "ended_utc": "2026-10-17T10:00:05Z", "wall_s": 5, "peak_rss_bytes": 2e9},
        "S1a": {"status": "done", "started_utc": "2026-10-17T10:00:05Z", "ended_utc": "2026-10-17T10:00:06Z", "wall_s": 1},
        "G1": {"status": "waiting", "started_utc": "2026-10-17T10:00:06Z", "ended_utc": "2026-10-17T10:00:07Z", "wall_s": 1},
        "S2": {"status": "failed", "started_utc": "2026-10-17T10:00:07Z", "ended_utc": "2026-10-17T10:00:09Z", "wall_s": 2, "exit_code": 1},
    })
    d = dashboard.collect(run)
    assert d["overall"] == "failed" and d["path"] == "cpu_scfast" and d["max_rss"] == 2e9
    html = dashboard.build(run, watch=10).read_text()
    assert "FAILED" in html and "Error: boom" in html            # log tail for the failed stage
    assert "waiting for" in html and "gates/G1/report.html" in html
    assert "http-equiv" not in html                               # finished/failed runs do not auto-refresh
    assert "cpu_scfast" in html and "umap 3.0 s" in html


def test_running_run_refreshes(tmp_path):
    _state(tmp_path, {"S0": {"status": "running", "started_utc": "2026-10-17T10:00:00Z"}})
    html = dashboard.build(tmp_path, watch=30).read_text()
    assert 'http-equiv="refresh" content="30"' in html and "RUNNING" in html


def test_cli(tmp_path, capsys):
    _state(tmp_path, {"S0": {"status": "done", "wall_s": 1}})
    assert dashboard.main(["--run-dir", str(tmp_path), "--out", str(tmp_path / "d.html")]) == 0
    assert (tmp_path / "d.html").exists()
