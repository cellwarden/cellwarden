"""Portability: site detection, site files, scheduler submission, doctor extras.

docs/agents-spec.md section 8 and docs/DESIGN.md section 7.

* :func:`detect_site` looks at the current node: scheduler (``sbatch`` /
  ``qsub`` / ``bsub``), GPU (``nvidia-smi``), container runtime (``apptainer``
  / ``singularity`` / ``docker``), CPU count and RAM, and internet access
  (HTTPS to the LLM endpoint and Open Targets). Nothing detected is fine: the
  draft then holds workstation defaults.
* :func:`draft_site` turns the detection into the site YAML of DESIGN 7.2;
  ``cellwarden init-site [--write conf/sites/<name>.yaml]`` prints or writes it.
* :func:`render_batch_script` fills ``conf/templates/<scheduler>.sh.j2`` and
  :func:`submit` writes ``<run-dir>/submit/<scheduler>.sh`` and calls
  ``sbatch`` / ``qsub`` / ``bsub``. The job runs ``cellwarden resume --run-dir D
  --execution direct``, so the batch job and the login node share one
  ``state.json`` and resume from wherever the other stopped.
* ``execution: split`` (:func:`submit_split`) submits the heavy stages up to
  the last non-gate stage before S5 as one job with ``--no-gates``; gates,
  Open Targets and the report then run on the submitting node with
  ``cellwarden resume`` once the job has finished.
* :func:`doctor_checks` adds the portability and LLM checks that
  ``cellwarden doctor`` prints.

Tests use a fake ``sbatch`` on ``PATH`` (``CELLWARDEN_SCHEDULER_PATH`` is not
needed: :func:`submit` looks the submit command up with ``shutil.which``).
Network checks honour ``CELLWARDEN_DOCTOR_OFFLINE=1`` (skip) and use short
timeouts.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import config as cfgmod

__all__ = [
    "SCHEDULERS",
    "detect_site",
    "draft_site",
    "init_site_main",
    "render_batch_script",
    "submit",
    "submit_split",
    "doctor_checks",
    "load_site_file",
    "validate_site_file",
]

# scheduler -> (submit command, job-id regex on stdout, how the script is passed)
SCHEDULERS: dict[str, dict[str, Any]] = {
    "slurm": {"submit": "sbatch", "probe": ("sbatch", "squeue"), "jobid": r"Submitted batch job (\d+)", "stdin": False},
    "pbs": {"submit": "qsub", "probe": ("qsub", "qstat"), "jobid": r"^(\S+)", "stdin": False},
    "lsf": {"submit": "bsub", "probe": ("bsub", "bjobs"), "jobid": r"Job <(\d+)>", "stdin": True},
    "sge": {"submit": "qsub", "probe": ("qsub", "qstat"), "jobid": r"job (\d+)", "stdin": False},
}

LLM_ENDPOINTS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openai_compatible": None,  # base_url from config
}
OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"
DEFAULT_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "nvidia": "NVIDIA_API_KEY",
                   "openai_compatible": "OPENAI_API_KEY"}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def detect_scheduler() -> str | None:
    """``slurm`` / ``pbs`` / ``lsf`` / ``sge`` / None, from the commands on PATH."""
    if shutil.which("sbatch"):
        return "slurm"
    if shutil.which("bsub"):
        return "lsf"
    if shutil.which("qsub"):
        # PBS and SGE both have qsub; SGE ships qconf, PBS ships pbsnodes.
        if shutil.which("qconf") and not shutil.which("pbsnodes"):
            return "sge"
        return "pbs"
    return None


def detect_gpu() -> dict[str, Any]:
    """``{"present": bool, "name": str|None, "total_gb": float|None, "driver": str|None}`` via nvidia-smi."""
    out: dict[str, Any] = {"present": False, "name": None, "total_gb": None, "driver": None, "count": 0}
    smi = shutil.which("nvidia-smi")
    if not smi:
        return out
    try:
        r = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return out
    rows = [ln.split(",") for ln in r.stdout.splitlines() if ln.strip()]
    if r.returncode != 0 or not rows:
        return out
    name, mem, drv = (x.strip() for x in rows[0][:3])
    out.update(present=True, name=name, driver=drv, count=len(rows))
    try:
        out["total_gb"] = round(float(mem) / 1024, 1)
    except ValueError:
        pass
    return out


def detect_container_runtime() -> str:
    for rt in ("apptainer", "singularity", "docker"):
        if shutil.which(rt):
            return rt
    return "none"


def detect_memory_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * size / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def https_reachable(url: str, timeout: float = 5.0) -> tuple[bool, str]:
    """HEAD/GET ``url``; any HTTP response (even 4xx) means the host is reachable."""
    if os.environ.get("CELLWARDEN_DOCTOR_OFFLINE") == "1":
        return False, "skipped (CELLWARDEN_DOCTOR_OFFLINE=1)"
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "cellwarden-doctor"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return True, f"HTTP {e.code} (host reachable)"
    except Exception as e:  # URLError, socket.timeout, ssl errors
        return False, f"unreachable: {str(e)[:120]}"


def detect_site(check_internet: bool = True, cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What this node offers. Pure detection; never raises."""
    llm_cfg = (_llm_section(cfg or {}) or {}).get("default") or {}
    provider = llm_cfg.get("provider") or "anthropic"
    llm_url = llm_cfg.get("base_url") or LLM_ENDPOINTS.get(provider) or LLM_ENDPOINTS["anthropic"]
    d: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "cpus": os.cpu_count() or 1,
        "memory_gb": detect_memory_gb(),
        "scheduler": detect_scheduler(),
        "gpu": detect_gpu(),
        "container_runtime": detect_container_runtime(),
        "internet": {"llm": None, "opentargets": None, "checked": bool(check_internet)},
        "scratch_candidates": [p for p in (os.environ.get("SCRATCH"), "/scratch", "/tmp") if p and Path(p).is_dir()],
        "detected_utc": _utc(),
    }
    if check_internet:
        ok, note = https_reachable(llm_url)
        d["internet"]["llm"] = {"ok": ok, "url": llm_url, "detail": note}
        ok, note = https_reachable(OPENTARGETS_URL)
        d["internet"]["opentargets"] = {"ok": ok, "url": OPENTARGETS_URL, "detail": note}
    return d


# --------------------------------------------------------------------------
# Site YAML
# --------------------------------------------------------------------------


def draft_site(det: Mapping[str, Any], name: str | None = None) -> dict[str, Any]:
    """Site YAML (DESIGN 7.2) from a detection. Workstation defaults when nothing is detected."""
    cpus = int(det.get("cpus") or 1)
    mem = det.get("memory_gb")
    gpu = det.get("gpu") or {}
    sched = det.get("scheduler")
    internet = det.get("internet") or {}
    llm_ok = (internet.get("llm") or {}).get("ok")
    name = name or ("hpc" if sched else "workstation")
    # Leave half the node for others when it looks shared (a scheduler is present).
    shared = sched is not None
    threads = max(1, min(cpus, 8) if shared else cpus)
    mem_cap = 64 if mem is None else (min(64, max(4, int(mem * 0.5))) if shared else max(4, int(mem * 0.8)))
    gpu_total = gpu.get("total_gb")
    gpu_cap = 20 if gpu_total is None else max(1, int(gpu_total * 0.85))
    scratch = (det.get("scratch_candidates") or ["~/cellwarden"])[0]
    if scratch == "/scratch":
        scratch = "/scratch/$USER/cellwarden"
    elif scratch == "/tmp":
        scratch = "~/cellwarden"
    execution = "direct"
    if sched and internet.get("checked") and llm_ok is False:
        execution = "split"  # compute nodes may have internet; the login node does not reach the LLM
    site = {
        "site": name,
        "execution": execution,
        "resources": {"threads": threads, "max_memory_gb": mem_cap, "gpu_mem_cap_gb": gpu_cap, "nice": 10 if shared else 0},
        "scratch": scratch,
        "container": {"runtime": det.get("container_runtime") or "none", "image": None,
                      "options": ["--nv"] if det.get("container_runtime") in ("apptainer", "singularity") and gpu.get("present") else []},
        "scheduler": {"partition": None, "account": None, "queue": None, "time": "24:00:00",
                      "cpus": threads if sched else None, "mem_gb": mem_cap if sched else None,
                      "gpus": 1 if gpu.get("present") else 0, "extra": []},
    }
    return site


def site_yaml_text(site: Mapping[str, Any], det: Mapping[str, Any] | None = None) -> str:
    lines = ["# cellwarden site file drafted by `cellwarden init-site` (docs/DESIGN.md 7.2)."]
    if det:
        gpu = det.get("gpu") or {}
        lines += [
            f"# host {det.get('hostname')} ({det.get('os')}), {det.get('cpus')} CPUs, "
            f"{det.get('memory_gb') or '?'} GB RAM, scheduler {det.get('scheduler') or 'none'}, "
            f"container {det.get('container_runtime')}",
            f"# GPU: {gpu.get('name') or 'none'}" + (f" ({gpu.get('total_gb')} GB, driver {gpu.get('driver')})" if gpu.get("present") else ""),
        ]
        inet = det.get("internet") or {}
        if inet.get("checked"):
            for k in ("llm", "opentargets"):
                v = inet.get(k) or {}
                lines.append(f"# internet {k}: {'ok' if v.get('ok') else 'NO'} ({v.get('detail')})")
        else:
            lines.append("# internet: not checked")
    rt = (site.get("container") or {}).get("runtime", "none")
    if rt != "none" and not (site.get("container") or {}).get("image"):
        ref = default_image() if rt == "docker" else f"docker://{default_image()}"
        lines += [f"# {rt} found: to run every stage in the prebuilt cellwarden image (R, Seurat, scfast, RAPIDS;",
                  f"# nothing else to install), set container.image to {ref}",
                  f"# or, faster to start, to a local copy: {rt} pull cellwarden.sif {ref}" if rt != "docker" else
                  f"# (docker pull {ref} once)."]
    lines += ["# Review the values, set scheduler.partition/account if you submit jobs, then",
              "# point config.yaml at this file with `site: <name>` or `site: path/to/file.yaml`.", ""]
    return "\n".join(lines) + yaml.safe_dump(dict(site), sort_keys=False, allow_unicode=True)


def load_site_file(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise cfgmod.ConfigError(f"site file {path} must hold a mapping")
    return data


def validate_site_file(path: str | Path) -> list[str]:
    """Problems with a site file (empty list = valid)."""
    problems: list[str] = []
    try:
        data = load_site_file(path)
    except (OSError, yaml.YAMLError, cfgmod.ConfigError) as e:
        return [str(e)]
    for k in data:
        if k not in cfgmod.SITE_SECTIONS and k != "site":
            problems.append(f"unknown site key {k!r} (allowed: site, {', '.join(cfgmod.SITE_SECTIONS)})")
    ex = data.get("execution", "direct")
    if ex not in cfgmod.EXECUTION_MODES:
        problems.append(f"execution must be one of {list(cfgmod.EXECUTION_MODES)}, got {ex!r}")
    for sec in ("resources", "container", "scheduler"):
        v = data.get(sec)
        if v is not None and not isinstance(v, dict):
            problems.append(f"{sec} must be a mapping")
        elif isinstance(v, dict):
            for kk in v:
                if kk not in cfgmod.DEFAULTS[sec]:
                    problems.append(f"unknown key {sec}.{kk}")
    return problems


def init_site_main(args: list[str] | None = None, out=None) -> int:
    import argparse

    out = out or sys.stdout
    p = argparse.ArgumentParser(prog="cellwarden init-site", description="detect this node and draft a site file")
    p.add_argument("--write", default=None, metavar="PATH", help="write the YAML here (default: print)")
    p.add_argument("--name", default=None, help="site name (default: hpc or workstation)")
    p.add_argument("--no-internet-check", action="store_true", help="skip the HTTPS checks")
    p.add_argument("--config", default=None, help="config.yaml (for the LLM provider endpoint)")
    p.add_argument("--json", action="store_true", help="print the detection as JSON instead")
    a = p.parse_args(args)
    cfg = None
    if a.config:
        try:
            cfg = yaml.safe_load(Path(a.config).read_text()) or {}
        except Exception:
            cfg = None
    det = detect_site(check_internet=not a.no_internet_check, cfg=cfg)
    if a.json:
        print(json.dumps(det, indent=2), file=out)
        return 0
    site = draft_site(det, a.name)
    text = site_yaml_text(site, det)
    if a.write:
        dest = Path(a.write).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        print(f"cellwarden: site file written to {dest}\n  use it with:  site: {dest}  in config.yaml", file=out)
    else:
        print(text, file=out)
    return 0


# --------------------------------------------------------------------------
# Scheduler submission
# --------------------------------------------------------------------------


def _template_env():
    import jinja2

    tdir = cfgmod.REPO_ROOT / "conf" / "templates"
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(tdir)), keep_trailing_newline=True,
                             undefined=jinja2.StrictUndefined, autoescape=False)
    env.filters["sh"] = shlex.quote
    return env


def _hhmm(t: str) -> str:
    """``D-HH:MM:SS`` / ``HH:MM:SS`` -> ``HH:MM`` for LSF."""
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d)
    bits = t.split(":")
    h = int(bits[0]) + 24 * days
    m = int(bits[1]) if len(bits) > 1 else 0
    return f"{h}:{m:02d}"


def container_prefix(cfg: Mapping[str, Any], run_dir: Path) -> str:
    """Command prefix that runs the stage inside the configured container image ('' when none)."""
    c = cfg.get("container") or {}
    rt, image = c.get("runtime", "none"), c.get("image")
    if rt == "none" or not image:
        return ""
    opts = " ".join(shlex.quote(o) for o in c.get("options") or [])
    repo = shlex.quote(str(cfg.get("repo_root") or cfgmod.REPO_ROOT))
    rd = shlex.quote(str(run_dir))
    # CELLWARDEN_IN_CONTAINER=1: the runner inside does not wrap its stages in the image again.
    if rt in ("apptainer", "singularity"):
        return f"{rt} exec {opts} --env CELLWARDEN_IN_CONTAINER=1 --bind {repo},{rd} {shlex.quote(str(image))}"
    if rt == "docker":
        return (f"docker run --rm {opts} -e CELLWARDEN_IN_CONTAINER=1 -v {repo}:{repo} -v {rd}:{rd} -w {repo} "
                f"--entrypoint '' {shlex.quote(str(image))}")
    return ""


IMAGE_REPO_ROOT = "/opt/cellwarden"  # where containers/Dockerfile installs the repository files
# Stage environment variables handed to a docker stage container (apptainer inherits the environment).
_STAGE_ENV_PREFIXES = ("CELLWARDEN_", "OMP_", "OPENBLAS_", "MKL_", "VECLIB_", "NUMBA_", "R_MAX_VSIZE",
                       "ANTHROPIC_", "OPENAI_", "CUDA_VISIBLE_DEVICES")


def stage_container_argv(cfg: Mapping[str, Any], run_dir: Path) -> list[str]:
    """Prefix that runs one stage inside the configured image in ``execution: direct`` ([] when
    no image is set, or when already inside the image, CELLWARDEN_IN_CONTAINER=1). The run
    directory, the config directory and every sample's folder are bound at the same paths, and
    the stage uses the image's cellwarden (IMAGE_REPO_ROOT), not the host's."""
    c = cfg.get("container") or {}
    rt, image = c.get("runtime", "none"), c.get("image")
    if rt == "none" or not image or os.environ.get("CELLWARDEN_IN_CONTAINER") == "1":
        return []
    binds = {str(Path(run_dir).resolve())}
    if cfg.get("config_dir"):
        binds.add(str(Path(cfg["config_dir"]).resolve()))
    for s in cfg.get("samples_resolved") or []:
        p = Path(s["path"]).resolve()
        binds.add(str(p if p.is_dir() else p.parent))
    opts = [str(o) for o in c.get("options") or []]
    env = {"CELLWARDEN_IN_CONTAINER": "1", "CELLWARDEN_REPO_ROOT": IMAGE_REPO_ROOT, "CELLWARDEN_PYTHON": "python",
           "PYTHONPATH": f"{IMAGE_REPO_ROOT}/python"}
    if rt in ("apptainer", "singularity"):
        argv = [rt, "exec", *opts]
        for b in sorted(binds):
            argv += ["--bind", b]
        argv += ["--env", ",".join(f"{k}={v}" for k, v in env.items())]
        return argv + [str(image)]
    if rt == "docker":
        argv = ["docker", "run", "--rm", "-i", "--user", f"{os.getuid()}:{os.getgid()}", *opts]
        for b in sorted(binds):
            argv += ["-v", f"{b}:{b}"]
        for k, v in env.items():
            argv += ["-e", f"{k}={v}"]
        for k in sorted(os.environ):
            if k.startswith(_STAGE_ENV_PREFIXES) and k not in env:
                argv += ["-e", k]
        return argv + ["--entrypoint", "", str(image)]
    return []


def default_image() -> str:
    """The published image matching this cellwarden version (.github/workflows/release.yml);
    CELLWARDEN_IMAGE overrides it (e.g. ad2n15/cellwarden:<tag>, the Docker Hub copy, or a fork's image)."""
    from . import __version__

    return os.environ.get("CELLWARDEN_IMAGE") or f"ghcr.io/cellwarden/cellwarden:{__version__}"


def render_batch_script(cfg: Mapping[str, Any], run_dir: Path, scheduler: str, *, until: str | None = None,
                        no_gates: bool = False) -> str:
    if scheduler not in SCHEDULERS:
        raise cfgmod.ConfigError(f"unknown scheduler {scheduler!r} (one of {list(SCHEDULERS)})")
    s = dict(cfgmod.DEFAULTS["scheduler"])
    s.update(cfg.get("scheduler") or {})
    res = cfg.get("resources") or {}
    run_dir = Path(run_dir)
    job_name = "cellwarden-" + re.sub(r"[^A-Za-z0-9_.-]+", "-", str(cfg.get("project") or "run"))[:40]
    log_path = run_dir / "logs" / f"{scheduler}-%j.log" if scheduler == "slurm" else run_dir / "logs" / f"{scheduler}.log"
    # Pass-through environment, by name only: secrets stay in the scheduler's environment propagation.
    env: dict[str, str] = {"CELLWARDEN_REPO_ROOT": str(cfg.get("repo_root") or cfgmod.REPO_ROOT)}
    for k in ("CELLWARDEN_RSCRIPT", "CELLWARDEN_ALLOW_CPU_AS_GPU", "CELLWARDEN_DOCTOR_OFFLINE"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    ctx = {
        "job_name": job_name,
        "log_path": str(log_path),
        "time": s.get("time") or "24:00:00",
        "time_hhmm": _hhmm(s.get("time") or "24:00:00"),
        "partition": s.get("partition"),
        "queue": s.get("queue"),
        "account": s.get("account"),
        "cpus": s.get("cpus") or res.get("threads"),
        "mem_gb": s.get("mem_gb") or res.get("max_memory_gb"),
        "gpus": s.get("gpus") or 0,
        "extra": list(s.get("extra") or []),
        "written_utc": _utc(),
        "run_id": cfg.get("run_id"),
        "repo_root": str(cfg.get("repo_root") or cfgmod.REPO_ROOT),
        "pythonpath": str(Path(cfg.get("repo_root") or cfgmod.REPO_ROOT) / "python"),
        "env": env,
        "python": sys.executable,
        "run_dir": str(run_dir),
        "until": until,
        "no_gates": no_gates,
        "container_cmd": container_prefix(cfg, run_dir),
    }
    return _template_env().get_template(f"{scheduler}.sh.j2").render(**ctx)


def submit(cfg: Mapping[str, Any], run_dir: Path, scheduler: str, *, until: str | None = None,
           no_gates: bool = False, out=None) -> dict[str, Any]:
    """Write ``<run-dir>/submit/<scheduler>.sh`` and submit it. Returns ``{job_id, script, command, stdout}``."""
    out = out or sys.stdout
    run_dir = Path(run_dir)
    spec = SCHEDULERS[scheduler]
    cmd = shutil.which(spec["submit"])
    if cmd is None:
        raise cfgmod.ConfigError(f"execution: {scheduler} but {spec['submit']!r} is not on PATH "
                                 f"(run `cellwarden init-site` on a node that has it, or use execution: direct)")
    sdir = run_dir / "submit"
    sdir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    script = sdir / f"{scheduler}.sh"
    script.write_text(render_batch_script(cfg, run_dir, scheduler, until=until, no_gates=no_gates), encoding="utf-8")
    script.chmod(0o750)
    if spec["stdin"]:
        argv = [cmd]
        r = subprocess.run(argv, stdin=open(script), capture_output=True, text=True)
    else:
        argv = [cmd, str(script)]
        r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        raise cfgmod.ConfigError(f"{spec['submit']} failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")
    m = re.search(spec["jobid"], r.stdout.strip(), re.M)
    job_id = m.group(1) if m else r.stdout.strip().split()[0] if r.stdout.strip() else None
    rec = {"scheduler": scheduler, "job_id": job_id, "script": str(script), "command": argv,
           "stdout": r.stdout.strip(), "submitted_utc": _utc(), "until": until, "no_gates": no_gates,
           "mode": "split" if no_gates else scheduler}
    (sdir / "last_job.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"cellwarden: submitted {scheduler} job {job_id or '?'} ({script})\n"
          f"  follow:   cellwarden status --run-dir {run_dir}\n"
          f"  job log:  {run_dir / 'logs'}/{scheduler}*.log", file=out)
    return rec


def heavy_until(stage_ids: list[str]) -> str | None:
    """Last non-gate stage before S5 (the stages a split job runs)."""
    last = None
    for sid in stage_ids:
        if sid.upper() == "S5":
            break
        if not sid.upper().startswith("G"):
            last = sid
    return last


def submit_split(cfg: Mapping[str, Any], run_dir: Path, stage_ids: list[str], *, until: str | None = None,
                 out=None) -> dict[str, Any]:
    """``execution: split``: one offline job for the heavy stages, gates and S5 on this node.

    The job runs ``cellwarden resume --until <last heavy stage> --no-gates``. When it
    has finished, ``cellwarden resume --run-dir D`` here runs the gate stages (LLM,
    Open Targets) and S5. TODO(coordinator): once G1-G4 are registered, decide
    whether a gate placed before a heavy stage (G1 before S2) should instead
    split the job in two; with ``--no-gates`` S2 uses the rules baseline
    (``qc.review_action``) as it does today.
    """
    out = out or sys.stdout
    sched = detect_scheduler()
    if sched is None:
        raise cfgmod.ConfigError("execution: split needs a scheduler (sbatch/qsub/bsub) on this node")
    stop = heavy_until(stage_ids)
    if until is not None and until.upper() != "S5":
        stop = until
    rec = submit(cfg, run_dir, sched, until=stop, no_gates=True, out=out)
    print(f"  then, on this node (gates, Open Targets, report):  cellwarden resume --run-dir {run_dir}", file=out)
    return rec


# --------------------------------------------------------------------------
# Doctor extras
# --------------------------------------------------------------------------


def _llm_section(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    """``llm:`` as the gates see it (``config.llm_settings``), or None when the gates run rules-only.

    ``default`` has the provider resolved and its preset (model, key variable, endpoint) filled in.
    """
    raw = cfg.get("llm")
    if not raw or not isinstance(raw, Mapping):
        return None
    out = cfgmod.llm_settings(cfg) or {}
    from .llm.provider import role_settings

    out["default"] = role_settings(out, "default")
    return out


def _llm_provider_module():
    try:
        from . import llm  # noqa: F401  (written by another developer; may not exist yet)
    except Exception:
        return None
    return llm


def scfastr_version() -> str | None:
    """Version of the R package scfastR, or None (no Rscript, or package not installed)."""
    rs = os.environ.get("CELLWARDEN_RSCRIPT") or "Rscript"
    if shutil.which(rs) is None:
        return None
    expr = 'if (nzchar(system.file(package = "scfastR"))) cat(as.character(utils::packageVersion("scfastR")))'
    try:
        r = subprocess.run([rs, "--vanilla", "-e", expr], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = r.stdout.strip()
    return out if r.returncode == 0 and out else None


def scfast_py_version() -> str | None:
    """Version of the Python module scfast (rust/scfast-py), or None if it cannot be imported."""
    try:
        import scfast
    except Exception:
        return None
    return str(getattr(scfast, "__version__", "unknown"))


def scfastr_cuda() -> dict[str, bool]:
    """CUDA state of the R package scfastR: ``{"built": bool, "available": bool}``.

    Read from ``scfastR::scfast_info()`` fields ``cuda_built`` / ``cuda_available`` (a missing field,
    a build without the CUDA feature, no Rscript, or no scfastR all give ``False``). ``available``
    implies ``built``.
    """
    off = {"built": False, "available": False}
    rs = os.environ.get("CELLWARDEN_RSCRIPT") or "Rscript"
    if shutil.which(rs) is None:
        return off
    expr = ('if (nzchar(system.file(package = "scfastR"))) { i <- as.list(scfastR::scfast_info()); '
            'f <- function(x) if (isTRUE(x)) "TRUE" else "FALSE"; '
            'b <- if (is.null(i$cuda_built)) i$cuda else i$cuda_built; '
            'cat(f(b), f(i$cuda_available), sep = " ") }')
    try:
        r = subprocess.run([rs, "--vanilla", "-e", expr], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return off
    parts = r.stdout.strip().split()
    if r.returncode != 0 or len(parts) != 2:
        return off
    built = parts[0] == "TRUE"
    return {"built": built, "available": built and parts[1] == "TRUE"}


def resolve_engine(cfg: Mapping[str, Any] | None, *, scfastr: str | None, gpu_available: bool,
                   scfastr_cuda_available: bool = False) -> tuple[str, str]:
    """The engine a run would use, as ``(engine, reason)``; engine is ``scfast`` or ``seurat``.

    CPU plan (``device: cpu``, or ``device: auto`` with no usable GPU): ``auto`` picks scfast when
    scfastR is installed, else seurat; ``engine: scfast`` without scfastR is an error (``RuntimeError``
    with an install hint).

    GPU plan (docs/scfast-integration.md "Pipeline wiring"): ``engine_gpu: rapids`` keeps the rapids
    exchange (seurat); ``engine_gpu: scfast`` runs scfast in process on CUDA and is an error unless
    scfastR was built with CUDA and a device is visible (``scfastr_cuda_available``); ``engine_gpu: auto``
    picks scfast when that holds, else seurat. ``engine: seurat`` is always seurat; ``engine: scfast``
    on a GPU plan that cannot run scfast on CUDA is an error.
    """
    cfg = cfg or {}
    engine = cfg.get("engine", "auto")
    engine_gpu = cfg.get("engine_gpu", "auto")
    device = cfg.get("device", "auto")
    install = ("engine: scfast needs the R package scfastR; install it with "
               "`R CMD INSTALL rust/scfast-r` (needs cargo >= 1.80)")
    if engine == "seurat":
        return "seurat", "engine: seurat"
    gpu_plan = device == "gpu" or (device == "auto" and gpu_available)
    cuda_ok = scfastr is not None and scfastr_cuda_available
    if engine == "scfast" and device != "gpu":
        # As before: an explicit scfast with device: auto is scfast (S2's own plan decides the device;
        # on a GPU plan it then needs scfastR with CUDA, which the R side checks).
        if scfastr is None:
            raise RuntimeError(install)
        if gpu_plan and cuda_ok and engine_gpu != "rapids":
            return "scfast", "engine: scfast; device plan is GPU and scfastR has a usable CUDA device"
        return "scfast", "engine: scfast"
    if not gpu_plan:
        if scfastr is None:
            return "seurat", "scfastR is not installed"
        return "scfast", "scfastR is installed and the device plan is CPU"
    if engine_gpu == "rapids":
        if engine == "scfast":
            raise RuntimeError("engine: scfast has no GPU backend with engine_gpu: rapids; "
                               "use device: cpu, engine_gpu: auto/scfast, or engine: auto/seurat")
        return "seurat", "device plan is GPU and engine_gpu: rapids (exchange)"
    if cuda_ok:
        return "scfast", ("device plan is GPU and scfastR has a usable CUDA device"
                          + ("; engine_gpu: scfast" if engine_gpu == "scfast" else " (engine_gpu: auto)"))
    why = ("scfastR is not installed" if scfastr is None
           else "scfastR was built without CUDA or no CUDA device is visible")
    if engine_gpu == "scfast":
        raise RuntimeError(f"engine_gpu: scfast needs scfastR built with CUDA and a visible device ({why}); "
                           "use engine_gpu: auto/rapids or device: cpu")
    if engine == "scfast":
        raise RuntimeError(f"engine: scfast has no GPU backend here ({why}); it cannot be combined with device: gpu")
    return "seurat", f"device plan is GPU and {why} (rapids exchange)"


def doctor_checks(cfg: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Checks appended by ``cellwarden doctor``: site, execution, container, LLM, Open Targets, budget."""
    checks: list[dict[str, Any]] = []

    def add(group: str, name: str, ok: bool, detail: str, required: bool = True) -> None:
        checks.append({"group": group, "name": name, "ok": bool(ok), "required": required, "detail": detail})

    cfg = cfg or {}
    # Site file
    src = cfg.get("site_source")
    if cfg.get("site"):
        probs = validate_site_file(src) if src else [f"site {cfg.get('site')!r} could not be resolved"]
        add("site", "site file", not probs, src if not probs else "; ".join(probs))
    else:
        add("site", "site file", True, "none (settings from config.yaml; `cellwarden init-site` drafts one)", required=False)
    # Execution mode
    ex = cfg.get("execution") or "direct"
    if ex == "direct":
        add("site", "execution", True, "direct (runs on this node)")
    else:
        sched = detect_scheduler() if ex == "split" else ex
        spec = SCHEDULERS.get(sched or "")
        have = spec is not None and shutil.which(spec["submit"]) is not None
        add("site", "execution", have, f"{ex}: " + (f"{spec['submit']} found" if have else
                                                   (f"{spec['submit']} (for {sched}) is not on PATH" if spec else "no scheduler (sbatch/qsub/bsub) on PATH")))
    # Container
    c = cfg.get("container") or {}
    rt = c.get("runtime", "none")
    if rt == "none":
        add("site", "container", True, "none (host environment)", required=False)
    else:
        w = shutil.which(rt)
        img = c.get("image")
        img_ok = img is None or str(img).startswith(("docker://", "oras://", "library://")) or Path(str(img)).exists() or rt == "docker"
        add("site", "container", w is not None and img_ok,
            f"{rt}: {w or 'not on PATH'}; image {img or 'none'}{'' if img_ok else ' (not found)'}")
    # LLM (config shape of DESIGN 4.8: llm.default.{provider,model,api_key_env,base_url}, llm.roles, llm.budget)
    llm_cfg = _llm_section(cfg)
    if llm_cfg is None:
        why = (cfg or {}).get("llm_choice") or "llm: not configured"
        add("llm", "provider", True, f"{why} (gates run rules-only with a human review)" if "rules-only" not in why else why,
            required=False)
        llm_cfg = {}
        provider = None
    else:
        provider = (llm_cfg.get("default") or {}).get("provider") or "anthropic"
    if provider is None:
        pass
    elif provider == "mock":
        add("llm", "provider", True, "mock (tests; no network)", required=False)
    elif provider in ("bedrock", "vertex"):
        add("llm", "provider", False, f"{provider}: not implemented yet (use anthropic or openai_compatible)", required=False)
    else:
        dflt = llm_cfg.get("default") or {}
        mod_ = _llm_provider_module()
        model = dflt.get("model") or ((mod_.role_settings(llm_cfg, "doctor").get("model"))
                                      if mod_ is not None and hasattr(mod_, "role_settings") else None)
        add("llm", "provider", True, f"{provider}, model {model}"
            + (f" ({cfg['llm_choice']})" if (cfg or {}).get("llm_choice") else ""), required=False)
        key_env = dflt.get("api_key_env") or DEFAULT_KEY_ENV.get(provider, "ANTHROPIC_API_KEY")
        have_key = bool(os.environ.get(key_env))
        add("llm", "api key", have_key, f"${key_env} is {'set' if have_key else 'not set'} (provider {provider})", required=False)
        base = dflt.get("base_url") or LLM_ENDPOINTS.get(provider)
        if base:
            ok, note = https_reachable(base)
            add("llm", "endpoint", ok, f"{base}: {note}", required=False)
        if not have_key:
            add("llm", "tool-use test", True, "skipped (no API key)", required=False)
        else:
            mod = _llm_provider_module()
            if mod is None or not hasattr(mod, "get_provider"):
                add("llm", "tool-use test", True, "skipped (cellwarden.llm not available yet)", required=False)
            else:
                try:
                    prov = mod.get_provider(llm_cfg, "doctor")
                    resp = prov.complete(system="Reply by calling the tool.", messages=[{"role": "user", "content": "ping"}],
                                         tools=[{"name": "ping", "description": "reply", "input_schema": {"type": "object", "properties": {}}}],
                                         max_tokens=256)
                    used = getattr(resp, "tool_calls", None)
                    add("llm", "tool-use test", bool(used) or bool(getattr(resp, "text", "")),
                        f"ok: stop_reason={getattr(resp, 'stop_reason', '?')}, tool_calls={len(used or [])}", required=False)
                except Exception as e:
                    add("llm", "tool-use test", False, f"failed: {str(e)[:200]}", required=False)
    if provider is not None:
        # Budget: the total ledger shared across runs (cellwarden.llm.ledger: {"runs": {run_id: gbp}})
        budget = llm_cfg.get("budget") or {}
        ledger_path = os.path.expanduser(str(budget.get("ledger_path") or llm_cfg.get("ledger_path") or "~/.cellwarden/ledger.json"))
        limit = budget.get("total_limit", 30.0)
        spent = None
        try:
            if Path(ledger_path).exists():
                data = json.loads(Path(ledger_path).read_text())
                runs = data.get("runs") or {}
                spent = float(sum(float(v) for v in runs.values())) if isinstance(runs, dict) else float(data.get("total_gbp") or 0)
            else:
                spent = 0.0
        except Exception as e:
            add("llm", "budget", True, f"ledger {ledger_path} unreadable: {e}", required=False)
        if spent is not None:
            try:
                lim = float(limit)
                add("llm", "budget", spent < lim, f"GBP {spent:.2f} of {lim:.2f} spent across runs ({ledger_path})", required=False)
            except (TypeError, ValueError):
                add("llm", "budget", True, f"GBP {spent:.2f} spent; total_limit {limit!r} is not a number", required=False)
    # scfast engine (docs/scfast-integration.md): all optional, the Seurat path needs neither.
    r_ver = scfastr_version()
    py_ver = scfast_py_version()
    add("engine", "scfastR", r_ver is not None,
        r_ver or "not installed (optional; `R CMD INSTALL rust/scfast-r`)", required=False)
    add("engine", "scfast (python)", py_ver is not None,
        py_ver or "not importable (optional; `maturin develop --release -m rust/scfast-py/Cargo.toml`)",
        required=False)
    cuda = scfastr_cuda() if r_ver is not None else {"built": False, "available": False}
    add("engine", "scfastR cuda built", cuda["built"],
        "yes" if cuda["built"] else ("scfastR is not installed" if r_ver is None else
                                     "no (optional; build scfastR with the cuda feature for engine_gpu: scfast)"),
        required=False)
    add("engine", "CUDA device visible (scfastR)", cuda["available"],
        "yes" if cuda["available"] else ("n/a: scfastR not built with cuda" if not cuda["built"] else
                                         "no CUDA device visible to scfastR"),
        required=False)
    try:
        from .gpu_core import gpu_available as _gpu_ok

        gpu_ok = bool(_gpu_ok()[0])
    except Exception:
        gpu_ok = False
    try:
        eng, why = resolve_engine(cfg, scfastr=r_ver, gpu_available=gpu_ok, scfastr_cuda_available=cuda["available"])
        add("engine", "engine (resolved)", True,
            f"{eng} ({why}; engine: {cfg.get('engine', 'auto')}, engine_gpu: {cfg.get('engine_gpu', 'auto')})",
            required=False)
    except RuntimeError as e:
        add("engine", "engine (resolved)", False, str(e),
            required=cfg.get("engine") == "scfast" or cfg.get("engine_gpu") == "scfast")
    # Open Targets
    ot = cfg.get("opentargets") or {}
    if ot.get("enabled", True) is False:
        add("opentargets", "api", True, "disabled in config", required=False)
    elif ot.get("offline"):
        add("opentargets", "api", True, f"offline mode; cache {ot.get('cache_dir')}", required=False)
    else:
        ok, note = https_reachable(OPENTARGETS_URL)
        add("opentargets", "api", ok, f"{OPENTARGETS_URL}: {note}", required=False)
    return checks


if __name__ == "__main__":
    raise SystemExit(init_site_main())
