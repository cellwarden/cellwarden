"""Shared helpers for the benchmark harness (docs/RUST_ENGINE.md section 8).

Kept free of heavy imports at module level, so runners can set thread
environment variables before numpy / scanpy are loaded.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = "1.0"
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
)

# Parity thresholds (docs/RUST_ENGINE.md section 7). Shared by parity.py and report.py.
PARITY_THRESHOLDS = {"pc_similarity": 0.99, "knn_overlap": 0.90, "ari": 0.90}
# Adoption rule (section 8.2).
ADOPT_MIN_CELLS = 100_000
ADOPT_MIN_SPEEDUP = 1.5
ADOPT_MIN_MEM_RATIO = 2.0


def set_threads(n: int | None) -> None:
    """Pin BLAS/OpenMP/numba/rayon threads. Must run before numpy is imported."""
    if n:
        for v in THREAD_ENV_VARS:
            os.environ[v] = str(n)


def repo_commit() -> str:
    try:
        root = Path(__file__).resolve().parent.parent
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- memory

_CLEAR_REFS = Path("/proc/self/clear_refs")
_STATUS = Path("/proc/self/status")


def _vm_hwm_bytes() -> int | None:
    try:
        for line in _STATUS.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def reset_peak_rss() -> bool:
    """Reset the kernel's peak-RSS counter (Linux >= 4.0). Returns False if unsupported."""
    try:
        _CLEAR_REFS.write_text("5")
        return True
    except OSError:
        return False


def peak_rss_bytes() -> tuple[int, str]:
    """(peak RSS in bytes, method). 'vmhwm' is per step when reset works; 'ru_maxrss' is process-wide."""
    hwm = _vm_hwm_bytes()
    if hwm is not None:
        return hwm, "vmhwm"
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024), "ru_maxrss"


# ---------------------------------------------------------------- environment

def _cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        # Sandboxes (Modal) can hide the model name; vendor, family and model still tell CPU
        # generations apart (AMD family 25 model 1 Milan, model 17 Genoa).
        info = {}
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, value = line.partition(":")
            info.setdefault(key.strip(), value.strip())
        if "model name" in info:
            return f"{info['model name']} ({info.get('vendor_id')} family {info.get('cpu family')} model {info.get('model')})"
    except Exception:
        pass
    return platform.processor() or "unknown"


def gpu_snapshot() -> list[dict[str, Any]]:
    """nvidia-smi state of every GPU (name, memory used/total, utilisation). Empty if no GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, name, used, total, util = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "name": name, "mem_used_mib": int(used),
                     "mem_total_mib": int(total), "util_pct": int(util)})
    return gpus


def environment(packages: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        load = os.getloadavg()
    except OSError:
        load = (-1.0, -1.0, -1.0)
    mem_total = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1]) * 1024
    except OSError:
        pass
    return {
        "node": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "mem_total_bytes": mem_total,
        "loadavg": list(load),
        "threads_env": {v: os.environ.get(v) for v in THREAD_ENV_VARS},
        "gpus": gpu_snapshot(),
        "commit": repo_commit(),
        "packages": packages or {},
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------- recording

class Recorder:
    """Times steps and appends one JSON line per step to results.jsonl."""

    def __init__(self, path: str | Path, base: dict[str, Any],
                 sync: Callable[[], None] | None = None,
                 gpu_mem: Callable[[], int | None] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.base = dict(base)
        self.sync = sync or (lambda: None)
        self.gpu_mem = gpu_mem or (lambda: None)
        self.rep = 0
        self.warmup = False

    def write(self, **fields: Any) -> None:
        rec = {"schema": SCHEMA_VERSION, **self.base, "rep": self.rep, "warmup": self.warmup, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    @contextmanager
    def step(self, name: str, kind: str = "compute"):
        """kind: 'compute', 'transfer' (CPU<->GPU copy), 'io' or 'startup'."""
        reset_ok = reset_peak_rss()
        self.sync()
        t0 = time.perf_counter()
        status, err = "ok", None
        try:
            yield
            self.sync()  # GPU work is asynchronous: wait before stopping the clock
        except BaseException as e:  # record the failure, then re-raise
            status, err = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            wall = time.perf_counter() - t0
            rss, method = peak_rss_bytes()
            gpu = self.gpu_mem()
            self.write(
                step=name, kind=kind, status=status, error=err,
                wall_s=round(wall, 6),
                transfer_s=round(wall, 6) if kind == "transfer" else 0.0,
                peak_rss_gb=round(rss / 1e9, 4),
                # 'step' = peak within this step; 'process' = peak since process start
                rss_scope="step" if (reset_ok and method == "vmhwm") else "process",
                peak_gpu_gb=None if gpu is None else round(gpu / 1e9, 4),
            )


# ---------------------------------------------------------------- outputs for parity

def write_outputs(outdir: str | Path, barcodes, pcs, knn, labels) -> None:
    """Write the parity outputs in the format bench/run_seurat.R also writes.

    pcs: cells x n_pcs float32 (row-major); knn: cells x k int32, 0-based, self
    excluded, -1 padding; labels: per-cell cluster labels (strings).
    """
    import numpy as np

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pcs = np.ascontiguousarray(pcs, dtype="<f4")
    knn = np.ascontiguousarray(knn, dtype="<i4")
    pcs.tofile(out / "pcs.f32")
    knn.tofile(out / "knn.i32")
    (out / "barcodes.txt").write_text("\n".join(map(str, barcodes)) + "\n")
    (out / "labels.txt").write_text("\n".join(map(str, labels)) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "n_cells": int(pcs.shape[0]), "n_pcs": int(pcs.shape[1]), "k": int(knn.shape[1]),
        "order": "row-major", "knn_base": 0, "knn_self": "excluded",
    }))


def read_outputs(outdir: str | Path):
    import numpy as np

    out = Path(outdir)
    meta = json.loads((out / "meta.json").read_text())
    n, p, k = meta["n_cells"], meta["n_pcs"], meta["k"]
    pcs = np.fromfile(out / "pcs.f32", dtype="<f4").reshape(n, p)
    knn = np.fromfile(out / "knn.i32", dtype="<i4").reshape(n, k)
    barcodes = (out / "barcodes.txt").read_text().split("\n")[:n]
    labels = (out / "labels.txt").read_text().split("\n")[:n]
    return barcodes, pcs, knn, labels


def knn_from_distances(dist_csr, k: int):
    """Neighbour indices (self excluded, -1 padded) from a scanpy-style distances CSR."""
    import numpy as np

    d = dist_csr.tocsr()
    n = d.shape[0]
    out = np.full((n, k), -1, dtype=np.int32)
    for i in range(n):
        s, e = d.indptr[i], d.indptr[i + 1]
        idx, val = d.indices[s:e], d.data[s:e]
        keep = idx != i
        idx, val = idx[keep], val[keep]
        order = np.argsort(val, kind="stable")[:k]
        out[i, : len(order)] = idx[order]
    return out
