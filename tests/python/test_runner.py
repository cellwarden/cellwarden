"""Tests for the cellwarden runner (docs/stages-spec.md section 1) with dummy stage commands."""

import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from cellwarden import config as C
from cellwarden import runner as R

REPO = Path(__file__).resolve().parents[2]

# A dummy stage: records its call and environment, checks its inputs, writes its
# outputs, deletes files (like S1c deleting the exchange file), and fails when
# <run-dir>/FAIL_<stage> exists.
SCRIPT = r"""
import json, os, sys, pathlib
sid, outs, ins, dels = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]), json.loads(sys.argv[4])
rd = pathlib.Path(os.environ["CELLWARDEN_RUN_DIR"])
with open(rd / "calls.log", "a") as f:
    f.write(sid + "\n")
pref = ("CELLWARDEN_", "OMP_", "OPENBLAS_", "MKL_", "VECLIB_", "NUMBA_", "PYTHONPATH", "R_MAX_VSIZE")
env = {k: v for k, v in os.environ.items() if k.startswith(pref)}
env["__nice__"] = os.nice(0)
env["__cwd__"] = os.getcwd()
(rd / f"env_{sid}.json").write_text(json.dumps(env))
for i in ins:
    if not any((rd / a).exists() for a in i.split("|")):
        print(f"{sid}: missing input {i}", file=sys.stderr); sys.exit(4)
print(f"{sid}: working")
if (rd / f"FAIL_{sid}").exists():
    for i in range(30):
        print(f"{sid} boom line {i}", file=sys.stderr if i % 2 else sys.stdout)
    sys.exit(3)
if (rd / f"NOOUT_{sid}").exists():
    sys.exit(0)
for o in outs:
    p = rd / o.split("|")[0]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{sid} output\n")
for d in dels:
    (rd / d).unlink(missing_ok=True)
"""


def dummy(stage_id, inputs=(), outputs=(), deletes=(), transient=(), external=None):
    def cmd(ctx):
        return [sys.executable, "-c", SCRIPT, stage_id, json.dumps(list(outputs)), json.dumps(list(inputs)),
                json.dumps(list(deletes))]

    return R.Stage(stage_id, "python", cmd, R.STAGE_CONFIG_KEYS[stage_id], tuple(inputs), tuple(outputs),
                   tuple(transient), external)


def dummy_stages():
    """Same shape as the real registry (inputs, outputs, deletion of the S1 exchange file)."""
    return [
        dummy("S0", ["config/samples.csv"], ["S0/object.rds|S0/object.qs2", "S0/ingest_report.json"],
              external=R._sample_paths),
        dummy("S1a", ["S0/object.rds|S0/object.qs2"], ["S1/metrics_counts.h5ad", "S1/qc_metrics.tsv.gz"],
              transient=["S1/metrics_counts.h5ad"]),
        dummy("S1b", ["S1/metrics_counts.h5ad", "S1/qc_metrics.tsv.gz"], ["S1/qc_metrics.tsv.gz"]),
        dummy("S1c", ["S1/qc_metrics.tsv.gz", "config/qc_policy.yaml"], ["S1/qc_decisions.tsv.gz", "S1/qc_summary.json"],
              deletes=["S1/metrics_counts.h5ad"]),
        dummy("S2", ["S0/object.rds|S0/object.qs2", "S1/qc_decisions.tsv.gz"], ["S2/object.rds|S2/object.qs2", "S2/summary.json"]),
    ]


ALL = ["S0", "S1a", "S1b", "S1c", "S2"]


@pytest.fixture
def project(tmp_path):
    proj = tmp_path / "proj"
    (proj / "data").mkdir(parents=True)
    (proj / "data" / "a.h5ad").write_text("sample a")
    (proj / "data" / "b.h5ad").write_text("sample b")
    (proj / "samples.csv").write_text("sample_id,path,donor\nA,data/a.h5ad,D1\nB,data/b.h5ad,D2\n")
    (proj / "qc_policy.yaml").write_text("hard_limits:\n  min_genes: 200\n")
    cfg = {"project": "t", "run_dir": str(tmp_path / "run"), "samples": "samples.csv", "qc_policy": "qc_policy.yaml",
           "resources": {"threads": 3, "max_memory_gb": 12, "gpu_mem_cap_gb": 7.5, "nice": 5}}
    (proj / "config.yaml").write_text(yaml.safe_dump(cfg))
    return proj


def edit_config(proj, **kv):
    cfg = yaml.safe_load((proj / "config.yaml").read_text())
    for k, v in kv.items():
        parts = k.split("__")
        cur = cfg
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    (proj / "config.yaml").write_text(yaml.safe_dump(cfg))


def run(proj, *extra, stages=None):
    out = io.StringIO()
    code = R.main(["run", "--config", str(proj / "config.yaml"), *extra], stages=stages or dummy_stages(), out=out)
    return code, out.getvalue()


def calls(rd):
    p = Path(rd) / "calls.log"
    lines = p.read_text().split() if p.exists() else []
    p.unlink(missing_ok=True)
    return lines


def state(rd):
    return json.loads((Path(rd) / "state.json").read_text())


def rd_of(proj):
    return Path(yaml.safe_load((proj / "config.yaml").read_text())["run_dir"])


# --------------------------------------------------------------------------


def test_first_run_then_skip(project):
    rd = rd_of(project)
    code, out = run(project)
    assert code == 0, out
    assert calls(rd) == ALL
    st = state(rd)
    assert st["run_id"] == C.load_resolved(rd)["run_id"]
    assert st["stage_order"] == ALL
    for sid in ALL:
        r = st["stages"][sid]
        assert r["status"] == "done" and r["exit_code"] == 0
        assert r["wall_s"] >= 0 and r["peak_rss_bytes"] > 0
        assert r["started_utc"].endswith("Z") and r["ended_utc"].endswith("Z")
        assert len(r["fingerprint"]) == 64 and r["log"] == f"logs/{sid}.log"
        log = (rd / r["log"]).read_text()
        assert f"{sid}: working" in log and log.startswith("# cellwarden: started") and "exit code 0" in log
    assert set(st["stages"]["S2"]["fingerprint_parts"]["config"]) == set(R.STAGE_CONFIG_KEYS["S2"])
    assert not (rd / "S1" / "metrics_counts.h5ad").exists()  # deleted by "S1c"

    code, out = run(project)
    assert code == 0 and calls(rd) == []
    assert "nothing to do" in out and out.count("— skipped (") == 5


def test_change_resolution_reruns_only_s2(project):
    rd = rd_of(project)
    run(project)
    calls(rd)
    fp_before = {s: state(rd)["stages"][s]["fingerprint"] for s in ALL}
    edit_config(project, clustering__resolution=0.9)
    code, out = run(project)
    assert code == 0, out
    assert calls(rd) == ["S2"]
    assert "config clustering" in state(rd)["stages"]["S2"]["reason"]
    assert all(state(rd)["stages"][s]["fingerprint"] == fp_before[s] for s in ALL[:-1])


def test_change_policy_reruns_s1c_and_later(project):
    rd = rd_of(project)
    run(project)
    calls(rd)
    (project / "qc_policy.yaml").write_text("hard_limits:\n  min_genes: 300\n")
    code, out = run(project)
    assert code == 0, out
    assert calls(rd) == ["S1c", "S2"]
    # Protected populations feed S1a: changing them reruns S1a onwards.
    (project / "qc_policy.yaml").write_text(
        "hard_limits:\n  min_genes: 300\nprotected_populations:\n  - name: Platelet\n    markers: [PPBP]\n")
    run(project)
    assert calls(rd) == ["S1a", "S1b", "S1c", "S2"]


def test_change_seed_reruns_from_s1a_because_exchange_was_deleted(project):
    rd = rd_of(project)
    run(project)
    calls(rd)
    edit_config(project, seed=7)
    code, out = run(project)
    assert code == 0, out
    # S1b uses the seed, but needs S1/metrics_counts.h5ad, which S1c deleted.
    assert calls(rd) == ["S1a", "S1b", "S1c", "S2"]
    assert "was removed" in state(rd)["stages"]["S1a"]["reason"]


def test_change_sample_file_reruns_everything(project):
    rd = rd_of(project)
    run(project)
    calls(rd)
    (project / "data" / "b.h5ad").write_text("sample b, new version")
    run(project)
    assert calls(rd) == ALL
    assert "input" in state(rd)["stages"]["S0"]["reason"]


def test_unrelated_setting_reruns_nothing(project):
    rd = rd_of(project)
    run(project)
    calls(rd)
    edit_config(project, resources__threads=1, resources__nice=0, project="renamed")
    run(project)
    assert calls(rd) == []


def test_failure_then_resume(project):
    rd = rd_of(project)
    rd.mkdir(parents=True)
    (rd / "FAIL_S1b").write_text("")
    code, out = run(project)
    assert code == 1
    assert calls(rd) == ["S0", "S1a", "S1b"]
    st = state(rd)
    assert st["stages"]["S1b"]["status"] == "failed" and st["stages"]["S1b"]["exit_code"] == 3
    assert st["stages"]["S1c"]["status"] == "pending" and st["stages"]["S2"]["status"] == "pending"
    assert "FAILED (exit code 3)" in out
    # The last 20 log lines are shown (stdout and stderr both captured), not more.
    assert "S1b boom line 29" in out and "S1b boom line 12" in out and "S1b boom line 8" not in out
    assert "cellwarden resume --run-dir" in out
    log = (rd / "logs" / "S1b.log").read_text()
    assert "boom line 1" in log and "boom line 2" in log

    (rd / "FAIL_S1b").unlink()
    out = io.StringIO()
    code = R.main(["resume", "--run-dir", str(rd)], stages=dummy_stages(), out=out)
    assert code == 0, out.getvalue()
    assert calls(rd) == ["S1b", "S1c", "S2"]
    assert all(state(rd)["stages"][s]["status"] == "done" for s in ALL)


def test_failure_downstream_invalidated(project):
    rd = rd_of(project)
    run(project)
    calls(rd)
    edit_config(project, n_hvg=1000)  # S2 only
    (rd / "FAIL_S2").write_text("")
    assert run(project)[0] == 1
    assert state(rd)["stages"]["S2"]["status"] == "failed"
    (rd / "FAIL_S2").unlink()
    edit_config(project, clustering__resolution=0.1)
    assert run(project)[0] == 0
    assert calls(rd) == ["S2", "S2"]


def test_exit_zero_without_outputs_is_failure(project):
    rd = rd_of(project)
    rd.mkdir(parents=True)
    (rd / "NOOUT_S0").write_text("")
    code, out = run(project)
    assert code == 1
    assert state(rd)["stages"]["S0"]["status"] == "failed"
    assert "did not write" in (rd / "logs" / "S0.log").read_text()


def test_missing_output_reruns(project):
    rd = rd_of(project)
    run(project)
    calls(rd)
    (rd / "S1" / "qc_summary.json").unlink()
    run(project)
    assert calls(rd) == ["S1c", "S2"]
    assert "output missing" in state(rd)["stages"]["S1c"]["reason"]
    (rd / "S0" / "object.rds").unlink()
    (rd / "S0" / "object.qs2").write_text("either format satisfies the output")
    run(project)
    assert calls(rd) == []


def test_until_and_continue(project):
    rd = rd_of(project)
    code, out = run(project, "--until", "s1a")
    assert code == 0
    assert calls(rd) == ["S0", "S1a"]
    st = state(rd)
    assert st["stages"]["S1b"]["status"] == "pending"
    code, out = run(project, "--until", "S1a")
    assert calls(rd) == [] and "nothing to do up to S1a" in out
    run(project)
    assert calls(rd) == ["S1b", "S1c", "S2"]
    code, out = run(project, "--until", "S9")
    assert code == 2 and "unknown stage" in out


def test_interrupted_running_stage_is_rerun_on_resume(project):
    rd = rd_of(project)
    run(project)
    calls(rd)
    st = state(rd)
    st["stages"]["S1c"]["status"] = "running"
    (rd / "state.json").write_text(json.dumps(st))
    out = io.StringIO()
    R.main(["resume", "--run-dir", str(rd)], stages=dummy_stages(), out=out)
    assert calls(rd) == ["S1c", "S2"]
    assert "did not finish" in state(rd)["stages"]["S1c"]["reason"]
    st = state(rd)
    st["stages"]["S1b"]["status"] = "running"
    (rd / "state.json").write_text(json.dumps(st))
    R.main(["resume", "--run-dir", str(rd)], stages=dummy_stages(), out=out)
    # S1b needs S1/metrics_counts.h5ad, which S1c deleted: start again from S1a.
    assert calls(rd) == ["S1a", "S1b", "S1c", "S2"]


def test_env_caps_and_nice(project):
    rd = rd_of(project)
    run(project)
    env = json.loads((rd / "env_S1b.json").read_text())
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "NUMBA_NUM_THREADS", "CELLWARDEN_THREADS"):
        assert env[k] == "3", k
    assert env["CELLWARDEN_MAX_MEMORY_GB"] == "12" and env["R_MAX_VSIZE"] == "12Gb"
    assert env["CELLWARDEN_GPU_MEM_CAP_GB"] == "7.5"
    assert env["CELLWARDEN_PYTHON"] == sys.executable
    assert env["CELLWARDEN_RUN_DIR"] == str(rd)
    assert env["CELLWARDEN_REPO_ROOT"] == str(REPO)
    assert env["CELLWARDEN_CONFIG"] == str(rd / "config" / "config.resolved.yaml")
    assert env["CELLWARDEN_STAGE"] == "S1b"
    assert env["CELLWARDEN_RUN_ID"] == C.load_resolved(rd)["run_id"]
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(REPO / "python")
    assert env["__cwd__"] == str(rd)
    assert env["__nice__"] == min(19, os.nice(0) + 5)
    assert state(rd)["stages"]["S1b"]["command"][:3] == ["nice", "-n", "5"]


def test_stage_env_unit(tmp_path):
    cfg = {"resources": {"threads": 2, "max_memory_gb": 64, "gpu_mem_cap_gb": 20, "nice": 10}, "run_id": "r",
           "repo_root": "/repo"}
    env = R.stage_env(cfg, tmp_path, "S2", base={"PYTHONPATH": "/other:/repo/python", "CELLWARDEN_ALLOW_CPU_AS_GPU": "1",
                                                   "NUMBA_CACHE_DIR": "/keep"})
    assert env["PYTHONPATH"] == "/repo/python:/other"
    assert env["CELLWARDEN_ALLOW_CPU_AS_GPU"] == "1"  # inherited
    assert env["NUMBA_CACHE_DIR"] == "/keep"
    assert env["CELLWARDEN_MAX_MEMORY_GB"] == "64" and env["OMP_NUM_THREADS"] == "2"


def test_r_stage_command_and_rscript_override(tmp_path, monkeypatch):
    ctx = R.RunContext(tmp_path, {}, REPO)
    st = {s.id: s for s in R.default_stages()}
    monkeypatch.delenv("CELLWARDEN_RSCRIPT", raising=False)
    assert st["S0"].command(ctx) == ["Rscript", str(REPO / "R" / "stages" / "run_stage.R"), "--stage", "S0",
                                     "--run-dir", str(tmp_path)]
    monkeypatch.setenv("CELLWARDEN_RSCRIPT", "/opt/R/bin/Rscript")
    assert st["S2"].command(ctx)[0] == "/opt/R/bin/Rscript" and st["S2"].command(ctx)[3] == "S2"
    assert st["S1b"].command(ctx) == [sys.executable, "-m", "cellwarden.stages.s1b_doublets", "--run-dir", str(tmp_path)]
    assert st["S1c"].command(ctx)[2] == "cellwarden.stages.s1c_qc"
    # docs/agents-spec.md section 1: S0 S1a S1b S1c G1 S2 S3 G2 G3 S4 G4 S5
    real = [s.id for s in R.default_stages()]
    assert real[:6] == ["S0", "S1a", "S1b", "S1c", "G1", "S2"]
    assert [s.lang for s in R.default_stages()][:6] == ["R", "R", "python", "python", "python", "R"]
    assert st["G1"].command(ctx)[2] == "cellwarden.gates.g1_qc"
    assert "S1/qc_decisions.tsv.gz" in st["G1"].inputs and "S1/qc_decisions.tsv.gz" in st["G1"].outputs
    assert st["S4"].command(ctx)[2] == "cellwarden.stages.s4" and st["S4"].optional_inputs


def test_status_output(project):
    rd = rd_of(project)
    rd.mkdir(parents=True)
    (rd / "FAIL_S1c").write_text("")
    run(project)
    out = io.StringIO()
    assert R.main(["status", "--run-dir", str(rd)], stages=dummy_stages(), out=out) == 0
    text = out.getvalue()
    lines = text.splitlines()
    header = next(l for l in lines if l.startswith("STAGE"))
    for col in ("LANG", "STATUS", "STARTED (UTC)", "WALL", "PEAK RSS", "EXIT"):
        assert col in header
    # columns by header position (stage names contain spaces)
    cols = ["STAGE", "NAME", "LANG", "STATUS", "STARTED (UTC)", "WALL", "PEAK RSS", "EXIT"]
    starts = [header.index(c) for c in cols] + [None]
    rows = {}
    for l in lines:
        if l.split() and l.split()[0] in ALL:
            rows[l.split()[0]] = {c: l[starts[k]:starts[k + 1]].strip() for k, c in enumerate(cols)}
    assert rows["S0"]["STATUS"] == "done" and rows["S1c"]["STATUS"] == "failed" and rows["S2"]["STATUS"] == "pending"
    assert rows["S0"]["NAME"] == "Ingest" and rows["S2"]["NAME"].startswith("Normalise")
    assert rows["S1c"]["EXIT"] == "3"
    assert rows["S0"]["PEAK RSS"].split()[-1] in ("MB", "KB")
    peak = state(rd)["stages"]["S0"]["peak_rss_bytes"]
    assert 1024**2 < peak < 100 * 1024**2  # the dummy stage, not the runner's own memory
    assert "last lines of logs/S1c.log" in text and "boom line 29" in text
    out = io.StringIO()
    assert R.main(["status", "--run-dir", str(rd), "--json"], out=out) == 0
    assert json.loads(out.getvalue())["stages"]["S1c"]["status"] == "failed"
    out = io.StringIO()
    assert R.main(["status", "--run-dir", str(rd / "nope")], out=out) == 1
    assert "has not started" in out.getvalue()


def test_concurrent_runner_refused(project):
    rd = rd_of(project)
    run(project, "--until", "S0")
    cfg = C.load_resolved(rd)
    with R._Lock(rd):
        with pytest.raises(R.RunnerError, match="another cellwarden runner"):
            R.run_pipeline(rd, cfg, stages=dummy_stages(), out=io.StringIO())
        code, out = run(project)
        assert code == 2 and "another cellwarden runner" in out
        assert "a runner is active" in R.status_text(rd, dummy_stages())


def test_invalid_config_reports_all(project):
    edit_config(project, device="tpu", n_hvg=-1)
    code, out = run(project)
    assert code == 2 and "device" in out and "n_hvg" in out
    assert not rd_of(project).exists()


def test_file_checksum(tmp_path, monkeypatch):
    small = tmp_path / "s.txt"
    small.write_text("hello")
    assert R.file_checksum(small).startswith("sha256:")
    monkeypatch.setattr(R, "FULL_HASH_LIMIT", 10)
    big = tmp_path / "b.bin"
    big.write_bytes(b"x" * 5000)
    c1 = R.file_checksum(big)
    assert c1.startswith("sampled:")
    big.write_bytes(b"x" * 4999 + b"y")
    assert R.file_checksum(big) != c1
    d = tmp_path / "dir"
    d.mkdir()
    (d / "matrix.mtx").write_text("1")
    c = R.file_checksum(d)
    (d / "matrix.mtx").write_text("2")
    assert c.startswith("dir:") and R.file_checksum(d) != c


def test_doctor(project, tmp_path):
    rep = R.doctor(str(project / "config.yaml"))
    names = {(c["group"], c["name"]) for c in rep["checks"]}
    assert ("python", "anndata") in names and ("gpu", "gpu") in names and ("storage", "run_dir") in names
    assert ("python", "rapids_singlecell") in names  # optional ones are listed
    opt = [c for c in rep["checks"] if not c["required"]]
    assert opt and all("detail" in c for c in opt)
    if shutil.which("Rscript"):
        assert ("R", "Seurat") in names and ("R", "harmony") in names
    out = io.StringIO()
    R._print_doctor(rep, out)
    assert "python:" in out.getvalue() and "gpu:" in out.getvalue()
    edit_config(project, device="tpu")
    rep = R.doctor(str(project / "config.yaml"))
    assert rep["ok"] is False
    assert any(c["group"] == "config" and not c["ok"] and "device" in c["detail"] for c in rep["checks"])
    # Write access: a read-only directory fails.
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        if os.access(ro, os.W_OK):
            pytest.skip("running as a user who can write anywhere")
        ok, _ = R._writable(ro / "run")
        assert not ok
    finally:
        ro.chmod(0o700)


def test_doctor_cli_json(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"))
    r = subprocess.run([sys.executable, "-m", "cellwarden", "doctor", "--json", "--run-dir", str(tmp_path / "x")],
                       capture_output=True, text=True, env=env, timeout=300)
    rep = json.loads(r.stdout)
    assert r.returncode == (0 if rep["ok"] else 1)
    assert "gpu" in rep


def test_detach_commands(tmp_path):
    m, cmd, how = R.detach_command(["run", "--config", "c.yaml"], tmp_path, "my run/1", method="tmux")
    assert cmd[:4] == ["tmux", "new-session", "-d", "-s"] and cmd[4] == "cellwarden-my-run-1"
    assert "tmux attach -t cellwarden-my-run-1" in how and "logs/runner.log" in how
    assert "-m cellwarden run --config c.yaml" in cmd[5]
    m, cmd, how = R.detach_command(["run"], tmp_path, "r", method="tmux", env_file=tmp_path / "e.json")
    assert "--env-file" in cmd[5] and "CELLWARDEN_" not in cmd[5].split("--env-file")[0]
    m, cmd, how = R.detach_command(["resume", "--run-dir", str(tmp_path)], tmp_path, "r", method="nohup")
    assert cmd[0] == "nohup" and cmd[1:4] == [sys.executable, "-m", "cellwarden"]
    assert "tail -f" in how and "cellwarden status" in how


def _wait_done(rd, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = rd / "state.json"
        if p.exists():
            st = json.loads(p.read_text())
            if all(st["stages"].get(s, {}).get("status") == "done" for s in ALL) and not R._runner_alive(rd):
                return st
        time.sleep(0.3)
    raise AssertionError("detached run did not finish: " + (rd / "logs" / "runner.log").read_text()
                         if (rd / "logs" / "runner.log").exists() else "no runner.log")


@pytest.mark.parametrize("method", ["nohup", "tmux"])
def test_detach_end_to_end(project, method):
    if shutil.which(method) is None:
        pytest.skip(f"{method} not installed")
    rd = rd_of(project)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "python"), str(Path(__file__).parent)]),
               CELLWARDEN_STAGES="test_runner:dummy_stages", CELLWARDEN_DETACH_WITH=method)
    r = subprocess.run([sys.executable, "-m", "cellwarden", "run", "--config", str(project / "config.yaml"), "--detach"],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert ("tmux attach -t" in r.stdout) if method == "tmux" else ("nohup" in r.stdout)
    try:
        _wait_done(rd)
    finally:
        if method == "tmux":
            subprocess.run(["tmux", "kill-session", "-t", "cellwarden-" + C.load_resolved(rd)["run_id"][:60]],
                           capture_output=True)
    assert calls(rd) == ALL
    assert "finished up to S2" in (rd / "logs" / "runner.log").read_text()
    assert not list(rd.glob(".cellwarden-env-*"))  # the env hand-over file is removed
