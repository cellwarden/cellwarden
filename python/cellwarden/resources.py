"""Device and memory plan (docs/stages-spec.md section 5; DESIGN 5.1).

``choose_plan(n_cells, n_hvg, device_setting, gpu_info)`` returns
``{device, plan, reason, ...}``:

* ``device`` is ``"cpu"`` or ``"gpu"``; ``plan`` is ``in_memory``,
  ``managed_memory`` or ``cpu``.
* Estimated GPU memory = cells x n_hvg x 4 bytes x 3 (working copies of the
  scaled HVG matrix).
* The usable budget is the GPU memory cap (``resources.gpu_mem_cap_gb``),
  lowered to the GPU's currently free memory when that is known (shared
  login nodes: another user may hold part of the GPU).
* ``device: cpu`` -> CPU path, plan ``cpu``.
* ``device: auto`` -> GPU ``in_memory`` if a GPU is available and the
  estimate fits the budget; otherwise CPU (plan ``cpu``). ``auto`` never
  chooses managed memory (spec: "otherwise CPU path").
* ``device: gpu`` -> GPU. Fails (:class:`PlanError`) if no usable GPU is
  found. ``in_memory`` if the estimate fits the budget, ``managed_memory``
  (RAPIDS managed memory, spills to host RAM) if it fits 3x the budget, and
  a :class:`PlanError` beyond that, since forcing the GPU there would thrash
  for hours; use ``device: auto`` or ``cpu``.

Test-only override: ``CELLWARDEN_ALLOW_CPU_AS_GPU=1`` makes a machine without a
GPU look like one GPU whose total and free memory equal the cap
(``gpu_info["kind"] == "cpu-as-gpu"``). ``cellwarden.gpu_core`` then runs its CPU
branch when asked for ``device="gpu"``.

CLI (used by the R side of S2)::

    python -m cellwarden.resources choose-plan --n-cells N --n-hvg H \
        --device auto|cpu|gpu --gpu-mem-cap-gb G
    python -m cellwarden.resources gpu-info [--gpu-mem-cap-gb G]

``choose-plan`` prints one JSON object on stdout and exits 0; on failure it
prints ``{"error": ..., "message": ...}`` on stderr and exits 2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

__all__ = ["PlanError", "BYTES_PER_VALUE", "WORKING_COPIES", "MANAGED_FACTOR", "estimate_gpu_bytes", "gpu_info", "choose_plan", "main"]

BYTES_PER_VALUE = 4
WORKING_COPIES = 3
MANAGED_FACTOR = 3
GB = 1024**3


class PlanError(RuntimeError):
    """No acceptable device/memory plan for the requested setting."""


def estimate_gpu_bytes(n_cells: int, n_hvg: int) -> int:
    return int(n_cells) * int(n_hvg) * BYTES_PER_VALUE * WORKING_COPIES


def cpu_as_gpu_enabled() -> bool:
    return os.environ.get("CELLWARDEN_ALLOW_CPU_AS_GPU") == "1"


def gpu_info(gpu_mem_cap_gb: float | None = None) -> dict[str, Any]:
    """Describe the GPU: ``{available, kind, reason, n_devices, total_gb, free_gb}``.

    ``kind`` is ``"cuda"``, ``"cpu-as-gpu"`` (test override) or ``"none"``.
    Memory figures are for device 0 and ``None`` when unknown.
    """
    from .gpu_core import gpu_available

    ok, reason = gpu_available()
    info: dict[str, Any] = {"available": ok, "kind": "cuda" if ok else "none", "reason": reason,
                            "n_devices": 0, "total_gb": None, "free_gb": None, "name": None}
    if ok:
        try:
            import cupy as cp

            info["n_devices"] = int(cp.cuda.runtime.getDeviceCount())
            free, total = cp.cuda.runtime.memGetInfo()
            info["free_gb"] = round(free / GB, 3)
            info["total_gb"] = round(total / GB, 3)
            try:
                props = cp.cuda.runtime.getDeviceProperties(0)
                name = props.get("name")
                info["name"] = name.decode() if isinstance(name, bytes) else name
            except Exception:
                pass
        except Exception as e:  # pragma: no cover - needs a GPU
            info["reason"] += f"; memory query failed ({type(e).__name__}: {e})"
    elif cpu_as_gpu_enabled():
        cap = float(gpu_mem_cap_gb) if gpu_mem_cap_gb is not None else None
        info.update(available=True, kind="cpu-as-gpu", n_devices=1, total_gb=cap, free_gb=cap,
                    name="cpu-as-gpu (CELLWARDEN_ALLOW_CPU_AS_GPU=1)",
                    reason=f"CELLWARDEN_ALLOW_CPU_AS_GPU=1: CPU treated as a GPU ({reason})")
    return info


query_gpu_info = gpu_info  # alias: choose_plan's parameter shadows the function name


def choose_plan(
    n_cells: int,
    n_hvg: int,
    device_setting: str,
    gpu_info: dict[str, Any] | None = None,
    *,
    gpu_mem_cap_gb: float | None = None,
) -> dict[str, Any]:
    """Pick the device and memory plan. See the module docstring for the rules.

    ``gpu_info`` is :func:`gpu_info` output (queried when ``None``). The cap
    comes from ``gpu_mem_cap_gb``, else ``gpu_info["mem_cap_gb"]``, else
    ``CELLWARDEN_GPU_MEM_CAP_GB``, else the GPU's total memory.
    """
    if device_setting not in ("auto", "cpu", "gpu"):
        raise ValueError(f"device must be 'auto', 'cpu' or 'gpu', got {device_setting!r}")
    if int(n_cells) < 0 or int(n_hvg) < 0:
        raise ValueError("n_cells and n_hvg must be non-negative")
    if gpu_mem_cap_gb is None and gpu_info is not None:
        gpu_mem_cap_gb = gpu_info.get("mem_cap_gb")
    if gpu_mem_cap_gb is None and os.environ.get("CELLWARDEN_GPU_MEM_CAP_GB"):
        gpu_mem_cap_gb = float(os.environ["CELLWARDEN_GPU_MEM_CAP_GB"])
    est = estimate_gpu_bytes(n_cells, n_hvg)
    base = {"n_cells": int(n_cells), "n_hvg": int(n_hvg), "device_setting": device_setting,
            "estimated_gpu_gb": round(est / GB, 3)}

    if device_setting == "cpu":
        return {"device": "cpu", "plan": "cpu", "reason": "device: cpu in config", **base}

    if gpu_info is None:
        gpu_info = query_gpu_info(gpu_mem_cap_gb)
    available = bool(gpu_info.get("available"))
    kind = gpu_info.get("kind", "cuda" if available else "none")
    if not available:
        why = gpu_info.get("reason", "no GPU")
        if device_setting == "gpu":
            raise PlanError(f"device: gpu requested but no usable GPU was found ({why})")
        return {"device": "cpu", "plan": "cpu", "reason": f"device: auto and no usable GPU ({why})",
                "gpu_kind": kind, **base}

    cap = gpu_mem_cap_gb if gpu_mem_cap_gb is not None else gpu_info.get("total_gb")
    if cap is None:
        raise PlanError("GPU memory cap unknown: pass gpu_mem_cap_gb (resources.gpu_mem_cap_gb)")
    budget = float(cap)
    budget_note = f"cap {float(cap):g} GB"
    free = gpu_info.get("free_gb")
    if free is not None and float(free) < budget:
        budget = float(free)
        budget_note = f"cap {float(cap):g} GB, lowered to {budget:g} GB currently free"
    base.update(gpu_kind=kind, gpu_budget_gb=round(budget, 3), gpu_mem_cap_gb=float(cap))
    est_gb = est / GB
    desc = f"estimated {est_gb:.2f} GB ({n_cells} cells x {n_hvg} HVGs x 4 B x 3) vs {budget_note}"
    tag = " [cpu-as-gpu test override]" if kind == "cpu-as-gpu" else ""

    if est_gb <= budget:
        return {"device": "gpu", "plan": "in_memory", "reason": f"device: {device_setting}; fits: {desc}{tag}", **base}
    if device_setting == "auto":
        return {"device": "cpu", "plan": "cpu",
                "reason": f"device: auto; does not fit the GPU budget: {desc}; using the CPU path{tag}", **base}
    if est_gb <= MANAGED_FACTOR * budget:
        return {"device": "gpu", "plan": "managed_memory",
                "reason": f"device: gpu; above the budget but within {MANAGED_FACTOR}x: {desc}; "
                          f"using RAPIDS managed memory{tag}", **base}
    raise PlanError(
        f"device: gpu requested but the data are too large even for managed memory: {desc} "
        f"(limit {MANAGED_FACTOR}x the budget); set device: auto or cpu"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m cellwarden.resources", description="Device and memory plan")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("choose-plan", help="print {device, plan, reason} as JSON")
    c.add_argument("--n-cells", type=int, required=True)
    c.add_argument("--n-hvg", type=int, required=True)
    c.add_argument("--device", choices=("auto", "cpu", "gpu"), required=True)
    c.add_argument("--gpu-mem-cap-gb", type=float, default=None,
                   help="default: $CELLWARDEN_GPU_MEM_CAP_GB, else the GPU's total memory")
    g = sub.add_parser("gpu-info", help="print GPU availability and memory as JSON")
    g.add_argument("--gpu-mem-cap-gb", type=float, default=None)
    a = p.parse_args(argv)
    cap = a.gpu_mem_cap_gb
    if cap is None and os.environ.get("CELLWARDEN_GPU_MEM_CAP_GB"):
        cap = float(os.environ["CELLWARDEN_GPU_MEM_CAP_GB"])
    try:
        if a.cmd == "gpu-info":
            out = query_gpu_info(cap)
        else:
            info = query_gpu_info(cap) if a.device != "cpu" else None
            out = choose_plan(a.n_cells, a.n_hvg, a.device, info, gpu_mem_cap_gb=cap)
    except (PlanError, ValueError) as e:
        print(json.dumps({"error": type(e).__name__, "message": str(e)}), file=sys.stderr)
        return 2
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
