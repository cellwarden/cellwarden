"""bench/hpc (docs/scfast-integration.md Phase 8): the input-size guard and the summariser.

The scale runs themselves need the HPC; only the pieces that run anywhere are tested.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HPC = REPO / "bench" / "hpc"
sys.path.insert(0, str(HPC))
import make_scale_inputs as msi  # noqa: E402
import summarise  # noqa: E402


def test_make_scale_inputs_refuses_large_sizes_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("CELLWARDEN_ALLOW_LARGE", raising=False)
    with pytest.raises(msi.RefusedError, match="CELLWARDEN_ALLOW_LARGE=1"):
        msi.check_allowed([2000, 50_000])
    msi.check_allowed([2000, 5000])  # at the limit is fine
    msi.check_allowed([50_000], env={"CELLWARDEN_ALLOW_LARGE": "1"})
    with pytest.raises(msi.RefusedError):
        msi.check_allowed([50_000], env={"CELLWARDEN_ALLOW_LARGE": "0"})
    # through main(): exit 2, nothing written
    assert msi.main(["--sizes", "50000", "--out", str(tmp_path / "x")]) == 2
    assert not (tmp_path / "x").exists()
    # and as a script, the documented sizes are refused too
    proc = subprocess.run([sys.executable, str(HPC / "make_scale_inputs.py"), "--out", str(tmp_path / "y")],
                          capture_output=True, text=True, env={**os.environ, "CELLWARDEN_ALLOW_LARGE": ""})
    assert proc.returncode == 2 and "CELLWARDEN_ALLOW_LARGE=1" in proc.stderr
    assert not (tmp_path / "y").exists()
    with pytest.raises(ValueError):
        msi.parse_sizes("10")
    assert msi.parse_sizes("2000, 500,2000") == [500, 2000]


def test_make_scale_inputs_small_size_writes_a_complete_input(tmp_path):
    pytest.importorskip("h5py")
    out = tmp_path / "data"
    assert msi.main(["--sizes", "300", "--genes", "400", "--types", "4", "--out", str(out)]) == 0
    d = out / "n300"
    assert (d / "sample.h5").exists() and (d / "samples.csv").exists() and (d / "truth.tsv").exists()
    truth = (d / "truth.tsv").read_text().splitlines()
    assert truth[0] == "cell_id\tcell_type" and len(truth) == 301
    assert all(l.startswith("syn_") for l in truth[1:])
    assert "syn,sample.h5,10x_h5" in (d / "samples.csv").read_text()
    man = json.loads((out / "manifest.json").read_text())
    assert man["datasets"][0]["cells"] == 300 and set(man["datasets"][0]["files"]) == {"sample.h5", "truth.tsv", "samples.csv"}
    # the sample sheet validates as a cellwarden config input
    from cellwarden import config as C

    (d / "config.yaml").write_text(f"project: t\nrun_dir: {tmp_path / 'run'}\nsamples: samples.csv\n")
    cfg = C.load_config(d / "config.yaml")
    assert cfg["samples_resolved"][0]["sample_id"] == "syn"


def _fake_runs(tmp_path):
    rows = [
        {"size": 2000, "engine": "seurat-cpu", "status": "ok", "path": "cpu_seurat", "s2_s": 10.0, "wall_s": 30.0,
         "peak_rss_gb": 4.0, "threads": 8, "node": "fake", "commit": "abc1234"},
        {"size": 2000, "engine": "scfast-cpu", "status": "ok", "path": "cpu_scfast", "s2_s": 4.0, "wall_s": 20.0,
         "peak_rss_gb": 1.5, "ari_vs_seurat": 0.95, "threads": 8, "node": "fake", "commit": "abc1234"},
        {"size": 200000, "engine": "seurat-cpu", "status": "ok", "path": "cpu_seurat", "s2_s": 600.0, "wall_s": 900.0,
         "peak_rss_gb": 40.0, "threads": 8, "node": "fake", "commit": "abc1234"},
        {"size": 200000, "engine": "scfast-cpu", "status": "ok", "path": "cpu_scfast", "s2_s": 500.0, "wall_s": 800.0,
         "peak_rss_gb": 10.0, "ari_vs_seurat": 0.93, "ari_vs_truth": 0.9, "threads": 8, "node": "fake", "commit": "abc1234"},
        {"size": 200000, "engine": "scfast-cuda", "status": "ok", "path": "gpu_scfast", "s2_s": 120.0, "wall_s": 400.0,
         "peak_rss_gb": 12.0, "peak_gpu_gb": 6.5, "ari_vs_seurat": 0.92, "threads": 8, "node": "fake", "commit": "abc1234"},
        {"size": 200000, "engine": "rapids", "status": "ok", "path": "gpu_exchange", "s2_s": 150.0, "wall_s": 420.0,
         "peak_rss_gb": 30.0, "peak_gpu_gb": 9.0, "ari_vs_seurat": 0.80, "threads": 8, "node": "fake", "commit": "abc1234"},
        {"size": 500000, "engine": "seurat-cpu", "status": "failed", "exit": 1, "wall_s": 3000.0},
        {"size": 500000, "engine": "scfast-cuda", "status": "ok", "path": "gpu_scfast", "s2_s": 300.0, "wall_s": 900.0,
         "peak_rss_gb": 20.0, "peak_gpu_gb": 14.0},
        {"size": 500000, "engine": "rapids", "status": "timeout", "exit": 124, "wall_s": 14400.0},
    ]
    p = tmp_path / "runs.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_summarise_fake_results(tmp_path):
    runs = _fake_runs(tmp_path)
    out = tmp_path / "out"
    assert summarise.main(["--runs", str(runs), "--out", str(out)]) == 0
    rows = json.loads((out / "summary.json").read_text())
    by = {(r["size"], r["engine"]): r for r in rows}
    assert by[(2000, "seurat-cpu")]["verdict"] == "baseline"
    assert by[(2000, "scfast-cpu")]["verdict"].startswith("parity ok; too small")
    assert by[(2000, "scfast-cpu")]["speedup_s2"] == pytest.approx(2.5)
    r = by[(200000, "scfast-cpu")]
    assert r["speedup_s2"] == pytest.approx(1.2) and r["mem_ratio"] == pytest.approx(4.0) and r["verdict"] == "adopt"
    r = by[(200000, "scfast-cuda")]
    assert r["speedup_s2"] == pytest.approx(5.0) and r["verdict"] == "adopt" and r["peak_gpu_gb"] == 6.5
    assert by[(200000, "rapids")]["verdict"].startswith("fails parity")
    assert by[(500000, "seurat-cpu")]["verdict"] == "baseline"
    assert by[(500000, "scfast-cuda")]["verdict"] == "no Seurat baseline"
    assert by[(500000, "rapids")]["verdict"] == "timeout"
    # ordering: by size, then the engine order
    assert [(r["size"], r["engine"]) for r in rows][:2] == [(2000, "seurat-cpu"), (2000, "scfast-cpu")]
    md = (out / "summary.md").read_text()
    assert "## 200,000 cells" in md and "| scfast-cuda | ok | gpu_scfast |" in md and "adopt" in md
    csv_lines = (out / "summary.csv").read_text().splitlines()
    assert csv_lines[0].startswith("size,engine,status,path,s2_s") and len(csv_lines) == 10


def test_summarise_reads_side_files_and_clusters(tmp_path):
    # a run with no numbers in the record: S2/summary.json, time.log, gpu_mem.log and clusters supply them
    def run(engine, s2_s, rss_kb, clusters, gpu=None):
        d = tmp_path / engine
        (d / "run" / "S2").mkdir(parents=True)
        (d / "run" / "S2" / "summary.json").write_text(json.dumps(
            {"total_s": s2_s, "path": "x", "cells": 4, "clusters": 2, "peak_memory": {"r_peak_rss_gb": 0.5}}))
        (d / "time.log").write_text(f"\tMaximum resident set size (kbytes): {rss_kb}\n")
        if gpu is not None:
            (d / "gpu_mem.log").write_text("".join(f"2026/10/17 10:00:0{i}.000, {v}\n" for i, v in enumerate(gpu)))
        (d / "clusters.tsv").write_text("cell_id\tcluster\n" + "".join(f"syn_c{i}\t{c}\n" for i, c in enumerate(clusters)))
        return {"size": 4, "engine": engine, "run_dir": str(d / "run"), "time_log": str(d / "time.log"),
                "gpu_mem_log": str(d / "gpu_mem.log"), "clusters": str(d / "clusters.tsv"), "truth": str(tmp_path / "truth.tsv")}

    (tmp_path / "truth.tsv").write_text("cell_id\tcell_type\n" + "".join(f"syn_c{i}\t{t}\n" for i, t in enumerate("AABB")))
    recs = [run("seurat-cpu", 8.0, 2 * 1024**2, "0011"), run("scfast-cuda", 2.0, 1024**2, "1100", gpu=[100, 2048, 512])]
    runs = tmp_path / "runs.jsonl"
    runs.write_text("".join(json.dumps(r) + "\n" for r in recs))
    rows = summarise.build_rows(summarise.read_runs(runs))
    by = {r["engine"]: r for r in rows}
    assert by["seurat-cpu"]["s2_s"] == 8.0 and by["seurat-cpu"]["peak_rss_gb"] == pytest.approx(2.0)
    assert by["scfast-cuda"]["peak_rss_gb"] == pytest.approx(1.0) and by["scfast-cuda"]["peak_gpu_gb"] == pytest.approx(2.0)
    assert by["scfast-cuda"]["ari_vs_seurat"] == pytest.approx(1.0) and by["scfast-cuda"]["ari_vs_truth"] == pytest.approx(1.0)
    assert by["scfast-cuda"]["speedup_s2"] == pytest.approx(4.0) and by["scfast-cuda"]["mem_ratio"] == pytest.approx(2.0)
    assert by["scfast-cuda"]["verdict"].startswith("parity ok; too small")
    assert "ari_vs_seurat" not in by["seurat-cpu"] and by["seurat-cpu"]["ari_vs_truth"] == pytest.approx(1.0)


def test_summarise_bad_input(tmp_path):
    p = tmp_path / "runs.jsonl"
    p.write_text("")
    assert summarise.main(["--runs", str(p), "--out", str(tmp_path / "o")]) == 2
    p.write_text('{"size": 1}\n')
    assert summarise.main(["--runs", str(p), "--out", str(tmp_path / "o")]) == 2
    assert summarise.main(["--runs", str(tmp_path / "missing.jsonl"), "--out", str(tmp_path / "o")]) == 2
