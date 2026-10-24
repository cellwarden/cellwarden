"""Tests for cellwarden.resources.choose_plan and its CLI (docs/stages-spec.md section 5)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cellwarden import gpu_core, resources as R

REPO = Path(__file__).resolve().parents[2]
GPU = {"available": True, "kind": "cuda", "reason": "1 CUDA device(s)", "total_gb": 24.0, "free_gb": 23.0}
NO_GPU = {"available": False, "kind": "none", "reason": "no CUDA device found"}


def cells_for(gb, n_hvg=2000):
    """Cell count whose estimate is exactly ``gb`` GiB."""
    return int(gb * 1024**3 / (n_hvg * 4 * 3))


def test_estimate():
    assert R.estimate_gpu_bytes(1000, 2000) == 1000 * 2000 * 4 * 3


def test_cpu_setting():
    p = R.choose_plan(10_000_000, 2000, "cpu", GPU, gpu_mem_cap_gb=20)
    assert (p["device"], p["plan"]) == ("cpu", "cpu")
    assert "device: cpu" in p["reason"]


def test_auto_fits():
    p = R.choose_plan(100_000, 2000, "auto", GPU, gpu_mem_cap_gb=20)
    assert (p["device"], p["plan"]) == ("gpu", "in_memory")
    assert "fits" in p["reason"] and p["estimated_gpu_gb"] == pytest.approx(2.235, abs=1e-3)


def test_auto_too_big_falls_back_to_cpu():
    p = R.choose_plan(cells_for(21), 2000, "auto", GPU, gpu_mem_cap_gb=20)
    assert (p["device"], p["plan"]) == ("cpu", "cpu")
    assert "does not fit" in p["reason"]


def test_auto_no_gpu():
    p = R.choose_plan(1000, 2000, "auto", NO_GPU, gpu_mem_cap_gb=20)
    assert (p["device"], p["plan"]) == ("cpu", "cpu")
    assert "no usable GPU" in p["reason"]


def test_gpu_no_gpu_fails():
    with pytest.raises(R.PlanError, match="no usable GPU"):
        R.choose_plan(1000, 2000, "gpu", NO_GPU, gpu_mem_cap_gb=20)


def test_gpu_managed_memory_and_too_big():
    p = R.choose_plan(cells_for(40), 2000, "gpu", GPU, gpu_mem_cap_gb=20)
    assert (p["device"], p["plan"]) == ("gpu", "managed_memory")
    assert "managed memory" in p["reason"]
    p = R.choose_plan(cells_for(10), 2000, "gpu", GPU, gpu_mem_cap_gb=20)
    assert p["plan"] == "in_memory"
    with pytest.raises(R.PlanError, match="too large"):
        R.choose_plan(cells_for(61), 2000, "gpu", GPU, gpu_mem_cap_gb=20)


def test_boundary_inclusive():
    n = cells_for(20)
    assert R.choose_plan(n, 2000, "auto", GPU, gpu_mem_cap_gb=20)["plan"] == "in_memory"
    assert R.choose_plan(n + 100, 2000, "auto", GPU, gpu_mem_cap_gb=20)["plan"] == "cpu"


def test_free_memory_lowers_budget():
    busy = dict(GPU, free_gb=5.0)
    p = R.choose_plan(cells_for(8), 2000, "auto", busy, gpu_mem_cap_gb=20)
    assert p["plan"] == "cpu" and "currently free" in p["reason"] and p["gpu_budget_gb"] == 5.0


def test_cap_defaults(monkeypatch):
    monkeypatch.delenv("CELLWARDEN_GPU_MEM_CAP_GB", raising=False)
    p = R.choose_plan(1000, 2000, "auto", GPU)  # no cap -> GPU total, lowered to free
    assert p["gpu_mem_cap_gb"] == 24.0 and p["gpu_budget_gb"] == 23.0
    monkeypatch.setenv("CELLWARDEN_GPU_MEM_CAP_GB", "4")
    assert R.choose_plan(1000, 2000, "auto", GPU)["gpu_mem_cap_gb"] == 4.0


def test_bad_device():
    with pytest.raises(ValueError):
        R.choose_plan(1, 1, "tpu", GPU)


def test_real_gpu_info_on_this_machine(monkeypatch):
    monkeypatch.delenv("CELLWARDEN_ALLOW_CPU_AS_GPU", raising=False)
    ok, _ = gpu_core.gpu_available()
    info = R.gpu_info(20)
    assert info["available"] == ok
    if not ok:
        assert R.choose_plan(1000, 2000, "auto", gpu_mem_cap_gb=20)["device"] == "cpu"


def test_cpu_as_gpu_override(monkeypatch):
    if gpu_core.gpu_available()[0]:
        pytest.skip("a real GPU is present")
    monkeypatch.setenv("CELLWARDEN_ALLOW_CPU_AS_GPU", "1")
    info = R.gpu_info(20)
    assert info["available"] and info["kind"] == "cpu-as-gpu" and info["total_gb"] == 20 and info["free_gb"] == 20
    p = R.choose_plan(1000, 2000, "gpu", gpu_mem_cap_gb=20)
    assert (p["device"], p["plan"]) == ("gpu", "in_memory") and "cpu-as-gpu" in p["reason"]
    p = R.choose_plan(cells_for(30), 2000, "gpu", gpu_mem_cap_gb=20)
    assert p["plan"] == "managed_memory"
    p = R.choose_plan(cells_for(30), 2000, "auto", gpu_mem_cap_gb=20)
    assert p["plan"] == "cpu"
    assert gpu_core._resolve_device("gpu") == "cpu-as-gpu"
    monkeypatch.setenv("CELLWARDEN_ALLOW_CPU_AS_GPU", "0")
    assert R.gpu_info(20)["available"] is False
    with pytest.raises(gpu_core.DeviceError):
        gpu_core._resolve_device("gpu")


def _cli(*args, env_extra=None):
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"))
    env.pop("CELLWARDEN_ALLOW_CPU_AS_GPU", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "cellwarden.resources", *args], capture_output=True, text=True, env=env)


def test_cli_choose_plan():
    r = _cli("choose-plan", "--n-cells", "5000", "--n-hvg", "2000", "--device", "cpu", "--gpu-mem-cap-gb", "20")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert {"device", "plan", "reason"} <= set(out) and out["device"] == "cpu"
    if not gpu_core.gpu_available()[0]:
        r = _cli("choose-plan", "--n-cells", "5000", "--n-hvg", "2000", "--device", "gpu", "--gpu-mem-cap-gb", "20")
        assert r.returncode == 2 and json.loads(r.stderr)["error"] == "PlanError"
        r = _cli("choose-plan", "--n-cells", "5000", "--n-hvg", "2000", "--device", "gpu",
                 env_extra={"CELLWARDEN_ALLOW_CPU_AS_GPU": "1", "CELLWARDEN_GPU_MEM_CAP_GB": "20"})
        assert r.returncode == 0, r.stderr
        out = json.loads(r.stdout)
        assert out["device"] == "gpu" and out["plan"] == "in_memory" and out["gpu_mem_cap_gb"] == 20
    r = _cli("gpu-info", "--gpu-mem-cap-gb", "8")
    assert r.returncode == 0 and "available" in json.loads(r.stdout)
