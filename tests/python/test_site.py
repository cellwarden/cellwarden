"""Tests for portability (docs/agents-spec.md section 8, DESIGN 7): cellwarden.site, the site files,
scheduler templates and submission with a fake sbatch/bsub on PATH, split mode, init-site and the
doctor extras. All offline (CELLWARDEN_DOCTOR_OFFLINE=1)."""

import io
import json
import os
import stat
import sys
from pathlib import Path

import pytest
import yaml

from cellwarden import config as C
from cellwarden import runner as R
from cellwarden import site as S

from test_runner import dummy_stages, project  # noqa: F401  (fixture)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("CELLWARDEN_DOCTOR_OFFLINE", "1")


def fake_scheduler(tmp_path, monkeypatch, name="sbatch", reply="Submitted batch job 4242", stdin=False):
    """A fake submit command on PATH that records its arguments (and stdin) and prints a job id."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / f"{name}.calls"
    body = f'#!/bin/sh\necho "$@" >> {log}\n' + (f"cat >> {log}\n" if stdin else "") + f'echo "{reply}"\n'
    exe = bindir / name
    exe.write_text(body)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    return log


def no_scheduler(tmp_path, monkeypatch):
    """A PATH with only the interpreter's directory: no sbatch/qsub/bsub anywhere."""
    bindir = tmp_path / "nobin"
    bindir.mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.setattr(S, "detect_scheduler", lambda: None)


def resolved(proj):
    cfg = C.load_config(str(proj / "config.yaml"))
    return C.prepare_run_dir(cfg)


# --------------------------------------------------------------------------
# Detection and site drafting
# --------------------------------------------------------------------------


def test_detect_scheduler(tmp_path, monkeypatch):
    no_scheduler(tmp_path, monkeypatch)
    monkeypatch.undo()
    monkeypatch.setenv("CELLWARDEN_DOCTOR_OFFLINE", "1")
    bindir = tmp_path / "nobin"
    monkeypatch.setenv("PATH", str(bindir))
    assert S.detect_scheduler() is None
    fake_scheduler(tmp_path, monkeypatch, "sbatch")
    assert S.detect_scheduler() == "slurm"


def test_detect_site_never_raises_and_is_offline(tmp_path, monkeypatch):
    no_scheduler(tmp_path, monkeypatch)
    det = S.detect_site(check_internet=True)
    assert det["scheduler"] is None and det["cpus"] >= 1 and det["container_runtime"] in ("none", "docker", "apptainer", "singularity")
    assert det["internet"]["checked"] and det["internet"]["llm"]["ok"] is False and "skipped" in det["internet"]["llm"]["detail"]
    det2 = S.detect_site(check_internet=False)
    assert det2["internet"]["llm"] is None


def test_draft_site_workstation_and_hpc():
    det = {"cpus": 16, "memory_gb": 128.0, "scheduler": None, "gpu": {"present": True, "total_gb": 24.0, "name": "L4"},
           "container_runtime": "none", "internet": {"checked": True, "llm": {"ok": True}}, "scratch_candidates": ["/tmp"]}
    ws = S.draft_site(det)
    assert ws["site"] == "workstation" and ws["execution"] == "direct"
    assert ws["resources"]["threads"] == 16 and ws["resources"]["nice"] == 0 and ws["resources"]["gpu_mem_cap_gb"] == 20
    assert ws["scheduler"]["cpus"] is None and ws["scheduler"]["gpus"] == 1
    det.update(scheduler="slurm", container_runtime="apptainer", scratch_candidates=["/scratch"])
    hpc = S.draft_site(det, "mycluster")
    assert hpc["site"] == "mycluster" and hpc["execution"] == "direct"
    assert hpc["resources"]["threads"] == 8 and hpc["resources"]["nice"] == 10 and hpc["resources"]["max_memory_gb"] == 64
    assert hpc["scratch"] == "/scratch/$USER/cellwarden" and hpc["container"] == {"runtime": "apptainer", "image": None, "options": ["--nv"]}
    # a scheduler but no route to the LLM endpoint from this node: suggest split
    det["internet"] = {"checked": True, "llm": {"ok": False}}
    assert S.draft_site(det)["execution"] == "split"
    # nothing detected at all
    bare = S.draft_site({})
    assert bare["execution"] == "direct" and bare["resources"]["threads"] == 1 and bare["container"]["runtime"] == "none"
    text = S.site_yaml_text(hpc, det)
    assert yaml.safe_load(text) == hpc and text.startswith("#")


@pytest.mark.parametrize("name", ["iridisx", "workstation", "docker"])
def test_shipped_site_files_are_valid(name):
    path = REPO / "conf" / "sites" / f"{name}.yaml"
    assert S.validate_site_file(path) == []
    data = S.load_site_file(path)
    assert data["site"] == name and data["execution"] in C.EXECUTION_MODES
    assert set(data) <= set(C.SITE_SECTIONS) | {"site"}


def test_validate_site_file_reports_problems(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("site: x\nexecution: cloud\nresources: {threads: 2, bogus: 1}\nfoo: 1\ncontainer: 3\n")
    probs = S.validate_site_file(p)
    assert any("execution" in x for x in probs) and any("bogus" in x for x in probs)
    assert any("unknown site key 'foo'" in x for x in probs) and any("container must be a mapping" in x for x in probs)
    assert S.validate_site_file(tmp_path / "missing.yaml")


def test_init_site_cli(tmp_path, monkeypatch):
    no_scheduler(tmp_path, monkeypatch)
    out = io.StringIO()
    dest = tmp_path / "sites" / "here.yaml"
    assert R.main(["init-site", "--write", str(dest), "--name", "here", "--no-internet-check"], out=out) == 0
    assert dest.exists() and f"site: {dest}" in out.getvalue()
    assert S.validate_site_file(dest) == []
    site = yaml.safe_load(dest.read_text())
    assert site["site"] == "here" and site["execution"] == "direct"
    assert "# internet: not checked" in dest.read_text()
    # printed instead of written; JSON detection
    out = io.StringIO()
    assert R.main(["init-site", "--no-internet-check"], out=out) == 0
    assert yaml.safe_load(out.getvalue())["site"] == "workstation"
    out = io.StringIO()
    assert R.main(["init-site", "--json"], out=out) == 0
    det = json.loads(out.getvalue())
    assert det["scheduler"] is None and "gpu" in det and det["internet"]["llm"]["ok"] is False
    # a site file drafted here loads through config.yaml (site: <path>)
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "samples.csv").write_text("sample_id,path\nA,a.h5ad\n")
    (proj / "a.h5ad").write_text("x")
    (proj / "config.yaml").write_text(yaml.safe_dump({"project": "p", "run_dir": str(tmp_path / "run"), "samples": "samples.csv",
                                                      "site": str(dest), "resources": {"threads": 1}}))
    cfg = C.load_config(str(proj / "config.yaml"))
    assert cfg["site"] == str(dest) and cfg["resources"]["threads"] == 1  # config.yaml wins per key
    assert cfg["resources"]["nice"] == site["resources"]["nice"] and cfg["execution"] == "direct"


def test_site_file_by_name_supplies_defaults(project):
    (project / "config.yaml").write_text((project / "config.yaml").read_text() + "site: iridisx\n")
    cfg = C.load_config(str(project / "config.yaml"))
    assert cfg["site"] == "iridisx" and cfg["site_source"].endswith("conf/sites/iridisx.yaml")
    assert cfg["container"]["runtime"] == "apptainer" and cfg["scheduler"]["partition"] == "gpu"
    assert cfg["resources"]["threads"] == 3  # from config.yaml, not the site file


# --------------------------------------------------------------------------
# Batch scripts and submission
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sched,marker", [("slurm", "#SBATCH --job-name="), ("pbs", "#PBS -N "), ("lsf", "#BSUB -J "), ("sge", "#$ -N ")])
def test_render_batch_script(project, sched, marker):
    cfg = resolved(project)
    cfg["scheduler"] = dict(cfg["scheduler"], partition="gpu", account="acc1", time="1-02:30:00", gpus=1, extra=["--qos=low"])
    rd = Path(cfg["run_dir"])
    txt = S.render_batch_script(cfg, rd, sched, until="S2", no_gates=True)
    assert txt.startswith("#!/usr/bin/env bash") and marker in txt
    assert "gpu" in txt and "acc1" in txt and "--qos=low" in txt
    assert f"-m cellwarden resume --run-dir {rd} --execution direct --until S2 --no-gates" in txt
    assert "set -euo pipefail" in txt and f"export PYTHONPATH={REPO / 'python'}" in txt
    assert "CELLWARDEN_REPO_ROOT=" in txt and "ANTHROPIC_API_KEY" not in txt
    if sched == "lsf":
        assert "#BSUB -W 26:30" in txt
    elif sched == "slurm":
        assert "--time=1-02:30:00" in txt and "--gres=gpu:1" in txt and "--cpus-per-task=3" in txt and "--mem=12G" in txt
    elif sched == "pbs":
        assert "walltime=1-02:30:00" in txt and ":ngpus=1" in txt
    plain = S.render_batch_script(cfg, rd, sched)
    assert "--until" not in plain and "--no-gates" not in plain
    with pytest.raises(C.ConfigError):
        S.render_batch_script(cfg, rd, "nqs")


def test_container_prefix(project):
    cfg = resolved(project)
    rd = Path(cfg["run_dir"])
    assert S.container_prefix(cfg, rd) == ""
    cfg["container"] = {"runtime": "apptainer", "image": "/img/cellwarden.sif", "options": ["--nv"]}
    pre = S.container_prefix(cfg, rd)
    assert pre.startswith("apptainer exec --nv --env CELLWARDEN_IN_CONTAINER=1 --bind") and pre.endswith("/img/cellwarden.sif")
    assert str(rd) in pre
    assert "apptainer exec --nv" in S.render_batch_script(cfg, rd, "slurm")
    cfg["container"] = {"runtime": "docker", "image": "ghcr.io/x/cellwarden:1", "options": ["--gpus", "all"]}
    assert S.container_prefix(cfg, rd).startswith("docker run --rm --gpus all -e CELLWARDEN_IN_CONTAINER=1 -v")


def test_stage_container_argv(project, monkeypatch):
    monkeypatch.delenv("CELLWARDEN_IN_CONTAINER", raising=False)
    cfg = resolved(project)
    rd = Path(cfg["run_dir"])
    assert S.stage_container_argv(cfg, rd) == []
    cfg["container"] = {"runtime": "apptainer", "image": "/img/cellwarden.sif", "options": ["--nv"]}
    argv = S.stage_container_argv(cfg, rd)
    assert argv[:3] == ["apptainer", "exec", "--nv"] and argv[-1] == "/img/cellwarden.sif"
    binds = [argv[i + 1] for i, a in enumerate(argv) if a == "--bind"]
    assert str(rd.resolve()) in binds and str(Path(cfg["config_dir"]).resolve()) in binds
    assert "CELLWARDEN_REPO_ROOT=/opt/cellwarden" in argv[argv.index("--env") + 1]
    # a stage command then uses the image's R scripts and python
    ctx = R.RunContext(run_dir=rd, cfg=cfg, repo_root=Path("/host/repo"))
    r_cmd = R._r_stage("S0")(ctx)
    assert r_cmd[: len(argv)] == argv and r_cmd[len(argv):len(argv) + 2] == ["Rscript", "/opt/cellwarden/R/stages/run_stage.R"]
    assert R._py_stage("cellwarden.stages.s3_celltypist")(ctx)[len(argv)] == "python"
    cfg["container"] = {"runtime": "docker", "image": "ghcr.io/x/cellwarden:1", "options": ["--gpus", "all"]}
    d = S.stage_container_argv(cfg, rd)
    assert d[:3] == ["docker", "run", "--rm"] and d[-3:] == ["--entrypoint", "", "ghcr.io/x/cellwarden:1"]
    monkeypatch.setenv("CELLWARDEN_IN_CONTAINER", "1")  # inside the image: no second wrap
    assert S.stage_container_argv(cfg, rd) == []


def test_submit_slurm_with_fake_sbatch(project, tmp_path, monkeypatch):
    log = fake_scheduler(tmp_path, monkeypatch, "sbatch")
    cfg = resolved(project)
    rd = Path(cfg["run_dir"])
    out = io.StringIO()
    rec = S.submit(cfg, rd, "slurm", out=out)
    script = rd / "submit" / "slurm.sh"
    assert script.exists() and rec["script"] == str(script) and rec["job_id"] == "4242"
    assert log.read_text().strip() == str(script)                     # sbatch <script>
    assert script.stat().st_mode & stat.S_IXUSR
    last = json.loads((rd / "submit" / "last_job.json").read_text())
    assert last["job_id"] == "4242" and last["scheduler"] == "slurm" and last["mode"] == "slurm"
    assert "submitted slurm job 4242" in out.getvalue() and "cellwarden status --run-dir" in out.getvalue()


def test_submit_lsf_reads_script_from_stdin(project, tmp_path, monkeypatch):
    log = fake_scheduler(tmp_path, monkeypatch, "bsub", reply="Job <77> is submitted to queue <normal>.", stdin=True)
    cfg = resolved(project)
    rec = S.submit(cfg, Path(cfg["run_dir"]), "lsf", out=io.StringIO())
    assert rec["job_id"] == "77"
    assert "#BSUB -J cellwarden-t" in log.read_text()                      # script content arrived on stdin


def test_submit_without_command_or_with_failure(project, tmp_path, monkeypatch):
    no_scheduler(tmp_path, monkeypatch)
    cfg = resolved(project)
    with pytest.raises(C.ConfigError, match="not on PATH"):
        S.submit(cfg, Path(cfg["run_dir"]), "slurm", out=io.StringIO())
    bindir = tmp_path / "fakebin"
    bindir.mkdir(exist_ok=True)
    (bindir / "qsub").write_text("#!/bin/sh\necho 'qsub: no partition' >&2\nexit 1\n")
    (bindir / "qsub").chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir))
    with pytest.raises(C.ConfigError, match="qsub failed"):
        S.submit(cfg, Path(cfg["run_dir"]), "pbs", out=io.StringIO())


def test_run_with_execution_slurm_submits_instead_of_running(project, tmp_path, monkeypatch):
    log = fake_scheduler(tmp_path, monkeypatch, "sbatch")
    out = io.StringIO()
    code = R.main(["run", "--config", str(project / "config.yaml"), "--execution", "slurm"], stages=dummy_stages(), out=out)
    assert code == 0, out.getvalue()
    rd = Path(yaml.safe_load((project / "config.yaml").read_text())["run_dir"])
    assert (rd / "submit" / "slurm.sh").exists() and "Submitted" not in out.getvalue()
    assert "submitted slurm job 4242" in out.getvalue()
    assert not (rd / "calls.log").exists()                            # no stage ran here
    assert (rd / "config" / "config.resolved.yaml").exists()         # the run dir is prepared for the job
    st = json.loads((rd / "state.json").read_text()) if (rd / "state.json").exists() else {"stages": {}}
    assert not any(v.get("status") == "done" for v in st.get("stages", {}).values())
    # the job's command line resumes the same run directory in direct mode
    txt = (rd / "submit" / "slurm.sh").read_text()
    assert f"cellwarden resume --run-dir {rd} --execution direct" in txt
    # ... which is what the job would do: run it here as the job would
    out = io.StringIO()
    code = R.main(["resume", "--run-dir", str(rd), "--execution", "direct"], stages=dummy_stages(), out=out)
    assert code == 0, out.getvalue()
    st = json.loads((rd / "state.json").read_text())
    assert all(st["stages"][s]["status"] == "done" for s in ("S0", "S1a", "S1b", "S1c", "S2"))
    # sbatch was called exactly once
    assert len(log.read_text().splitlines()) == 1


def test_execution_from_config_and_site_file(project, tmp_path, monkeypatch):
    fake_scheduler(tmp_path, monkeypatch, "sbatch")
    site = tmp_path / "site.yaml"
    site.write_text("site: t\nexecution: slurm\nscheduler: {partition: batch, time: '02:00:00'}\n")
    (project / "config.yaml").write_text((project / "config.yaml").read_text() + f"site: {site}\n")
    out = io.StringIO()
    code = R.main(["run", "--config", str(project / "config.yaml")], stages=dummy_stages(), out=out)
    assert code == 0, out.getvalue()
    rd = Path(yaml.safe_load((project / "config.yaml").read_text())["run_dir"])
    txt = (rd / "submit" / "slurm.sh").read_text()
    assert "#SBATCH --partition=batch" in txt and "#SBATCH --time=02:00:00" in txt
    # --execution direct overrides the site file and runs here
    out = io.StringIO()
    code = R.main(["run", "--config", str(project / "config.yaml"), "--execution", "direct"], stages=dummy_stages(), out=out)
    assert code == 0 and (rd / "calls.log").exists()


def test_split_mode_submits_heavy_stages_with_no_gates(project, tmp_path, monkeypatch):
    log = fake_scheduler(tmp_path, monkeypatch, "sbatch")
    out = io.StringIO()
    code = R.main(["run", "--config", str(project / "config.yaml"), "--execution", "split"], stages=dummy_stages(), out=out)
    assert code == 0, out.getvalue()
    rd = Path(yaml.safe_load((project / "config.yaml").read_text())["run_dir"])
    txt = (rd / "submit" / "slurm.sh").read_text()
    assert "--until S2 --no-gates" in txt                              # last heavy stage of the dummy registry
    last = json.loads((rd / "submit" / "last_job.json").read_text())
    assert last["mode"] == "split" and last["until"] == "S2" and last["no_gates"] is True
    assert "then, on this node" in out.getvalue() and "cellwarden resume --run-dir" in out.getvalue()
    assert len(log.read_text().splitlines()) == 1
    # the heavy stages before S5, skipping gates
    assert S.heavy_until(["S0", "S1a", "S1b", "S1c", "G1", "S2", "S3", "G2", "G3", "S4", "G4", "S5"]) == "S4"
    assert S.heavy_until(["S0", "G1", "S5"]) == "S0"


def test_split_mode_needs_a_scheduler(project, tmp_path, monkeypatch):
    no_scheduler(tmp_path, monkeypatch)
    out = io.StringIO()
    code = R.main(["run", "--config", str(project / "config.yaml"), "--execution", "split"], stages=dummy_stages(), out=out)
    assert code == 2 and "needs a scheduler" in out.getvalue()
    out = io.StringIO()
    code = R.main(["run", "--config", str(project / "config.yaml"), "--execution", "pbs"], stages=dummy_stages(), out=out)
    assert code == 2 and "not on PATH" in out.getvalue()


# --------------------------------------------------------------------------
# doctor extras
# --------------------------------------------------------------------------


def by_name(checks):
    return {(c["group"], c["name"]): c for c in checks}


def test_doctor_checks_without_llm(tmp_path, monkeypatch):
    no_scheduler(tmp_path, monkeypatch)
    ch = by_name(S.doctor_checks({}))
    assert ch[("site", "site file")]["ok"] and "init-site" in ch[("site", "site file")]["detail"]
    assert ch[("site", "execution")]["ok"] and ch[("site", "container")]["ok"]
    assert ch[("llm", "provider")]["ok"] and "not configured" in ch[("llm", "provider")]["detail"]
    assert ("llm", "tool-use test") not in ch and ("llm", "budget") not in ch
    assert ch[("opentargets", "api")]["ok"] is False and "skipped" in ch[("opentargets", "api")]["detail"]
    assert all(not c["required"] for k, c in ch.items() if k[0] in ("llm", "opentargets"))


def test_doctor_checks_llm_without_key_skips_tool_test(tmp_path, monkeypatch):
    no_scheduler(tmp_path, monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MY_KEY", raising=False)
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"runs": {"r1": 1.25, "r2": 0.5}}))
    cfg = {"llm": {"default": {"provider": "anthropic", "model": "claude-haiku-4-5", "api_key_env": "MY_KEY"},
                   "budget": {"total_limit": 30, "ledger_path": str(ledger)}},
           "opentargets": {"offline": True, "cache_dir": "/c"}}
    ch = by_name(S.doctor_checks(cfg))
    assert ch[("llm", "api key")]["ok"] is False and "$MY_KEY is not set" in ch[("llm", "api key")]["detail"]
    assert ch[("llm", "tool-use test")]["ok"] and ch[("llm", "tool-use test")]["detail"] == "skipped (no API key)"
    assert "api.anthropic.com" in ch[("llm", "endpoint")]["detail"]
    assert ch[("llm", "budget")]["ok"] and "GBP 1.75 of 30.00" in ch[("llm", "budget")]["detail"]
    assert ch[("opentargets", "api")]["ok"] and "offline" in ch[("opentargets", "api")]["detail"]
    # over budget
    ledger.write_text(json.dumps({"runs": {"r1": 31}}))
    assert by_name(S.doctor_checks(cfg))[("llm", "budget")]["ok"] is False
    # the older flat shape is accepted too
    ch = by_name(S.doctor_checks({"llm": {"provider": "openai_compatible", "base_url": "http://localhost:11434/v1"}}))
    assert "OPENAI_API_KEY" in ch[("llm", "api key")]["detail"] and "localhost:11434" in ch[("llm", "endpoint")]["detail"]


def test_doctor_checks_llm_mock_and_tool_use(tmp_path, monkeypatch):
    no_scheduler(tmp_path, monkeypatch)
    ch = by_name(S.doctor_checks({"llm": {"default": {"provider": "mock"}}}))
    assert ch[("llm", "provider")]["ok"] and "mock" in ch[("llm", "provider")]["detail"]
    # with a key, the tool-use test runs through cellwarden.llm.get_provider when it exists;
    # a provider that fails must give a failed, non-required check rather than an exception
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(S, "_llm_provider_module", lambda: None)
    ch = by_name(S.doctor_checks({"llm": {"default": {"provider": "anthropic"}}}))
    assert ch[("llm", "tool-use test")]["ok"] and "skipped" in ch[("llm", "tool-use test")]["detail"]

    class Boom:
        @staticmethod
        def get_provider(cfg_llm, role):
            raise RuntimeError("no network here")

    monkeypatch.setattr(S, "_llm_provider_module", lambda: Boom)
    ch = by_name(S.doctor_checks({"llm": {"default": {"provider": "anthropic"}}}))
    c = ch[("llm", "tool-use test")]
    assert c["ok"] is False and not c["required"] and "no network here" in c["detail"]

    class Fake:
        @staticmethod
        def get_provider(cfg_llm, role):
            class P:
                def complete(self, **kw):
                    assert kw["tools"] and kw["max_tokens"] <= 256
                    return type("R", (), {"tool_calls": [{"name": "ping"}], "text": "", "stop_reason": "tool_use"})()

            return P()

    monkeypatch.setattr(S, "_llm_provider_module", lambda: Fake)
    ch = by_name(S.doctor_checks({"llm": {"default": {"provider": "anthropic"}}}))
    assert ch[("llm", "tool-use test")]["ok"] and "tool_calls=1" in ch[("llm", "tool-use test")]["detail"]


def test_doctor_checks_execution_and_site(project, tmp_path, monkeypatch):
    no_scheduler(tmp_path, monkeypatch)
    cfg = resolved(project)
    cfg["execution"] = "slurm"
    ch = by_name(S.doctor_checks(cfg))
    assert ch[("site", "execution")]["ok"] is False and "sbatch" in ch[("site", "execution")]["detail"]
    fake_scheduler(tmp_path, monkeypatch, "sbatch")
    assert by_name(S.doctor_checks(cfg))[("site", "execution")]["ok"]
    cfg["execution"] = "split"
    monkeypatch.setattr(S, "detect_scheduler", lambda: "slurm")
    assert by_name(S.doctor_checks(cfg))[("site", "execution")]["ok"]
    cfg["container"] = {"runtime": "apptainer", "image": "/nope.sif"}
    c = by_name(S.doctor_checks(cfg))[("site", "container")]
    assert c["ok"] is False and ("not on PATH" in c["detail"] or "not found" in c["detail"])
    cfg["site"], cfg["site_source"] = "iridisx", str(REPO / "conf" / "sites" / "iridisx.yaml")
    assert by_name(S.doctor_checks(cfg))[("site", "site file")]["ok"]
    cfg["site_source"] = str(tmp_path / "missing.yaml")
    assert by_name(S.doctor_checks(cfg))[("site", "site file")]["ok"] is False


def test_doctor_cli_includes_portability_checks(project, tmp_path, monkeypatch):
    no_scheduler(tmp_path, monkeypatch)
    rep = R.doctor(str(project / "config.yaml"))
    groups = {c["group"] for c in rep["checks"]}
    assert {"site", "llm", "opentargets"} <= groups
    out = io.StringIO()
    R._print_doctor(rep, out)
    assert "site:" in out.getvalue() and "llm:" in out.getvalue()


# --------------------------------------------------------------------------
# scfast engine checks
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg,scfastr,gpu,expect",
    [
        ({}, "0.1.0", False, "scfast"),
        ({}, None, False, "seurat"),
        ({}, "0.1.0", True, "seurat"),
        ({"device": "gpu"}, "0.1.0", False, "seurat"),
        ({"device": "cpu"}, "0.1.0", True, "scfast"),
        ({"engine": "seurat"}, "0.1.0", False, "seurat"),
        ({"engine": "scfast", "device": "auto"}, "0.1.0", True, "scfast"),
    ],
)
def test_resolve_engine(cfg, scfastr, gpu, expect):
    assert S.resolve_engine(cfg, scfastr=scfastr, gpu_available=gpu)[0] == expect


def test_resolve_engine_scfast_errors():
    with pytest.raises(RuntimeError, match="R CMD INSTALL rust/scfast-r"):
        S.resolve_engine({"engine": "scfast"}, scfastr=None, gpu_available=False)
    with pytest.raises(RuntimeError, match="GPU"):
        S.resolve_engine({"engine": "scfast", "device": "gpu"}, scfastr="0.1.0", gpu_available=False)


def test_doctor_checks_engine(tmp_path, monkeypatch):
    no_scheduler(tmp_path, monkeypatch)
    monkeypatch.setattr(S, "scfastr_version", lambda: "0.1.0")
    monkeypatch.setattr(S, "scfast_py_version", lambda: None)
    ch = by_name(S.doctor_checks({"engine": "auto", "device": "cpu"}))
    assert ch[("engine", "scfastR")]["ok"] and ch[("engine", "scfastR")]["detail"] == "0.1.0"
    assert not ch[("engine", "scfast (python)")]["ok"] and not ch[("engine", "scfast (python)")]["required"]
    assert ch[("engine", "engine (resolved)")]["detail"].startswith("scfast")
    monkeypatch.setattr(S, "scfastr_version", lambda: None)
    ch = by_name(S.doctor_checks({"engine": "scfast", "device": "cpu"}))
    r = ch[("engine", "engine (resolved)")]
    assert not r["ok"] and r["required"] and "scfastR" in r["detail"]
    ch = by_name(S.doctor_checks({"engine": "auto", "device": "cpu"}))
    assert ch[("engine", "engine (resolved)")]["detail"].startswith("seurat")
    assert not any(c["required"] for k, c in ch.items() if k[0] == "engine")


# GPU-plan engine resolution: engine_gpu and the scfastR CUDA state (docs/scfast-integration.md "Pipeline wiring")


@pytest.mark.parametrize(
    "cfg,cuda,expect",
    [
        ({"device": "gpu"}, True, "scfast"),
        ({"device": "gpu"}, False, "seurat"),
        ({"device": "auto"}, True, "scfast"),
        ({"device": "gpu", "engine_gpu": "auto"}, True, "scfast"),
        ({"device": "gpu", "engine_gpu": "rapids"}, True, "seurat"),
        ({"device": "gpu", "engine_gpu": "scfast"}, True, "scfast"),
        ({"device": "gpu", "engine": "seurat", "engine_gpu": "auto"}, True, "seurat"),
        ({"device": "gpu", "engine": "scfast", "engine_gpu": "scfast"}, True, "scfast"),
        ({"device": "gpu", "engine": "scfast"}, True, "scfast"),
        ({"device": "cpu", "engine_gpu": "scfast"}, False, "scfast"),
        ({"device": "cpu", "engine_gpu": "rapids"}, True, "scfast"),
    ],
)
def test_resolve_engine_gpu_plan(cfg, cuda, expect):
    eng, why = S.resolve_engine(cfg, scfastr="0.1.0", gpu_available=True, scfastr_cuda_available=cuda)
    assert eng == expect
    if expect == "scfast" and cfg.get("device") != "cpu":
        assert "CUDA" in why


def test_resolve_engine_gpu_plan_errors_and_defaults():
    # default: a GPU plan without CUDA keeps the rapids exchange, with no CUDA info passed at all
    assert S.resolve_engine({"device": "gpu"}, scfastr="0.1.0", gpu_available=True)[0] == "seurat"
    assert S.resolve_engine({"device": "gpu"}, scfastr=None, gpu_available=True, scfastr_cuda_available=True)[0] == "seurat"
    with pytest.raises(RuntimeError, match="engine_gpu: scfast needs scfastR built with CUDA"):
        S.resolve_engine({"device": "gpu", "engine_gpu": "scfast"}, scfastr="0.1.0", gpu_available=True)
    with pytest.raises(RuntimeError, match="engine_gpu: scfast needs scfastR built with CUDA"):
        S.resolve_engine({"device": "gpu", "engine_gpu": "scfast"}, scfastr=None, gpu_available=True,
                         scfastr_cuda_available=True)
    with pytest.raises(RuntimeError, match="no GPU backend"):
        S.resolve_engine({"device": "gpu", "engine": "scfast"}, scfastr="0.1.0", gpu_available=True)
    with pytest.raises(RuntimeError, match="no GPU backend"):
        S.resolve_engine({"device": "gpu", "engine": "scfast", "engine_gpu": "rapids"}, scfastr="0.1.0",
                         gpu_available=True, scfastr_cuda_available=True)


def test_scfastr_cuda_parses_scfast_info(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    rs = bindir / "Rscript"

    def fake(out: str, code: int = 0):
        rs.write_text(f"#!/bin/sh\nprintf '%s' '{out}'\nexit {code}\n")
        rs.chmod(0o755)

    monkeypatch.setenv("CELLWARDEN_RSCRIPT", str(rs))
    fake("TRUE TRUE")
    assert S.scfastr_cuda() == {"built": True, "available": True}
    fake("TRUE FALSE")
    assert S.scfastr_cuda() == {"built": True, "available": False}
    fake("FALSE FALSE")
    assert S.scfastr_cuda() == {"built": False, "available": False}
    fake("FALSE TRUE")  # available without built is never reported
    assert S.scfastr_cuda() == {"built": False, "available": False}
    fake("")  # scfastR not installed: the expression prints nothing
    assert S.scfastr_cuda() == {"built": False, "available": False}
    fake("TRUE TRUE", code=1)
    assert S.scfastr_cuda() == {"built": False, "available": False}
    monkeypatch.setenv("CELLWARDEN_RSCRIPT", str(bindir / "missing"))
    assert S.scfastr_cuda() == {"built": False, "available": False}


def test_scfastr_cuda_real_rscript_reports_no_cuda_here():
    # this machine has no CUDA: whatever is installed, the answer is "not available" and never raises
    c = S.scfastr_cuda()
    assert set(c) == {"built", "available"} and c["available"] is False


def test_doctor_checks_engine_cuda_lines(tmp_path, monkeypatch):
    no_scheduler(tmp_path, monkeypatch)
    monkeypatch.setattr(S, "scfastr_version", lambda: "0.1.0")
    monkeypatch.setattr(S, "scfast_py_version", lambda: None)
    monkeypatch.setattr(S, "scfastr_cuda", lambda: {"built": True, "available": True})
    ch = by_name(S.doctor_checks({"engine": "auto", "device": "gpu"}))
    assert ch[("engine", "scfastR cuda built")]["ok"] and ch[("engine", "CUDA device visible (scfastR)")]["ok"]
    assert ch[("engine", "engine (resolved)")]["detail"].startswith("scfast (device plan is GPU")
    assert "engine_gpu: auto" in ch[("engine", "engine (resolved)")]["detail"]
    monkeypatch.setattr(S, "scfastr_cuda", lambda: {"built": True, "available": False})
    ch = by_name(S.doctor_checks({"engine": "auto", "device": "gpu"}))
    assert ch[("engine", "scfastR cuda built")]["ok"] and not ch[("engine", "CUDA device visible (scfastR)")]["ok"]
    assert ch[("engine", "engine (resolved)")]["detail"].startswith("seurat")
    ch = by_name(S.doctor_checks({"engine": "auto", "device": "gpu", "engine_gpu": "scfast"}))
    r = ch[("engine", "engine (resolved)")]
    assert not r["ok"] and r["required"] and "engine_gpu: scfast" in r["detail"]
    monkeypatch.setattr(S, "scfastr_cuda", lambda: {"built": False, "available": False})
    ch = by_name(S.doctor_checks({"engine": "auto", "device": "cpu"}))
    assert not ch[("engine", "scfastR cuda built")]["ok"] and not ch[("engine", "scfastR cuda built")]["required"]
    assert not ch[("engine", "CUDA device visible (scfastR)")]["required"]
    assert ch[("engine", "engine (resolved)")]["detail"].startswith("scfast")
    # without scfastR the CUDA lines are off and Rscript is never asked
    monkeypatch.setattr(S, "scfastr_version", lambda: None)
    monkeypatch.setattr(S, "scfastr_cuda", lambda: (_ for _ in ()).throw(AssertionError("not called")))
    ch = by_name(S.doctor_checks({"engine": "auto", "device": "cpu"}))
    assert ch[("engine", "scfastR cuda built")]["detail"] == "scfastR is not installed"
