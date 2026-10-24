"""GPU core step: scale -> PCA -> (Harmony) -> neighbours -> Leiden -> UMAP.

Reads a ``slim`` exchange file and writes a ``results`` exchange file
(docs/exchange-spec.md). Runs on the GPU with rapids-singlecell when it is
installed and a CUDA device is visible; otherwise on CPU with scanpy. Both
branches perform the same steps with the same parameters.

Test-only override: with the environment variable ``CELLWARDEN_ALLOW_CPU_AS_GPU=1``,
``device="gpu"`` on a machine without a usable GPU runs the CPU branch instead
of failing, and the summary (and ``uns["gpu_core"]["device"]``) reports
``device: "cpu-as-gpu"``. It exists so the GPU path of the pipeline (exchange
write -> gpu_core -> merge) can be exercised end to end on CPU-only machines;
never set it in production runs.

CLI::

    python -m cellwarden.gpu_core --in slim.h5ad --out results.h5ad \
        [--n-pcs 30 --n-neighbors 15 --resolution 0.5 --batch-key batch \
         --seed 0 --device auto --engine {rapids,scanpy,rust} --max-value 10 --run-id RUN]

Engines: ``rapids`` (GPU, rapids-singlecell) and ``scanpy`` (CPU) are the two classic branches; with no
``--engine`` the device picks one as before (gpu -> rapids, cpu -> scanpy). ``rust`` uses the Python module
``scfast`` (rust/scfast-py) for PCA, kNN, SNN and Leiden; UMAP still comes from umap-learn and Harmony
from harmonypy (both CPU). PCA and kNN run on CUDA (``device='cuda'``, reported as device ``gpu`` and
``scfast_device: cuda``) when ``scfast.device_info()`` reports ``cuda_built`` and ``cuda_available``;
otherwise ``--device auto`` stays on the CPU and ``--device gpu`` is a clear error. It does not apply
``--max-value`` clipping. ``distances`` in the
results are the kNN distances and ``connectivities`` the SNN (Jaccard) graph, not Scanpy's fuzzy weights.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .exchange import ExchangeError, read_exchange, write_exchange

__all__ = ["run", "gpu_available", "main", "ENGINES"]

ENGINES = ("rapids", "scanpy", "rust")

STAGE = "S2_gpu_core"
CLUSTER_KEY = "leiden"


class DeviceError(RuntimeError):
    """The requested device is not available."""


def gpu_available() -> tuple[bool, str]:
    """Return (available, reason). Available = rapids_singlecell imports and a CUDA device exists."""
    try:
        import rapids_singlecell  # noqa: F401
        import cupy as cp
    except Exception as e:  # ImportError, or CUDA library load failures
        return False, f"rapids_singlecell/cupy not importable ({type(e).__name__}: {e})"
    try:
        n = cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        return False, f"no usable CUDA device ({type(e).__name__}: {e})"
    if n < 1:
        return False, "no CUDA device found"
    return True, f"{n} CUDA device(s)"


def _resolve_device(device: str) -> str:
    if device not in ("auto", "gpu", "cpu"):
        raise ValueError(f"device must be 'auto', 'gpu' or 'cpu', got {device!r}")
    if device == "cpu":
        return "cpu"
    ok, reason = gpu_available()
    if device == "gpu" and not ok and os.environ.get("CELLWARDEN_ALLOW_CPU_AS_GPU") == "1":
        return "cpu-as-gpu"  # test-only override, see module docstring
    if device == "gpu" and not ok:
        raise DeviceError(f"device='gpu' requested but the GPU path is unavailable: {reason}")
    return "gpu" if ok else "cpu"


def _import_scfast():
    try:
        import scfast
    except ImportError as e:
        raise RuntimeError(
            "--engine rust needs the Python module scfast, which is not importable "
            f"({type(e).__name__}: {e}); install it with "
            "`pip install maturin && maturin develop --release -m rust/scfast-py/Cargo.toml` "
            "(needs cargo >= 1.80), or use --engine scanpy / rapids"
        ) from e
    return scfast


def scfast_cuda_available(scfast) -> tuple[bool, str]:
    """Whether the Python module scfast can run on CUDA, as ``(ok, reason)``.

    Asks ``scfast.device_info()`` (added with the CUDA backend; a build without it has no such
    function, which counts as unavailable) for ``cuda_built`` / ``cuda_available``; a missing
    field is False.
    """
    info_fn = getattr(scfast, "device_info", None)
    if info_fn is None:
        return False, "this build of scfast has no device_info() (CPU-only build)"
    try:
        info = info_fn()
    except Exception as e:  # a broken CUDA runtime must give a clear error, not a traceback
        return False, f"scfast.device_info() failed ({type(e).__name__}: {e})"
    get = info.get if isinstance(info, dict) else (lambda k, d=None: getattr(info, k, d))
    if not bool(get("cuda_built", False)):
        return False, "scfast was built without the cuda feature"
    if not bool(get("cuda_available", False)):
        return False, "scfast was built with cuda but no CUDA device is visible"
    return True, str(get("device_name", None) or "CUDA device visible")


def _check_harmony_cpu() -> None:
    try:
        import harmonypy  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "batch_key was given but harmonypy is not installed; "
            "install it (pip install harmonypy) or omit batch_key"
        ) from e


def _scfast_has_harmony() -> bool:
    try:
        import scfast
    except ImportError:
        return False
    return hasattr(scfast, "harmony")


def _harmony_cpu(adata, batch_key: str, seed: int) -> np.ndarray:
    """Harmony on X_pca via harmonypy directly.

    scanpy.external.pp.harmony_integrate (scanpy 1.12) transposes ``Z_corr``,
    which harmonypy >= 2 already returns as cells x dims; calling harmonypy
    here and fixing the orientation by shape works with both generations.
    """
    import logging

    import harmonypy

    logging.getLogger("harmonypy").setLevel(logging.WARNING)
    pcs = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    ho = harmonypy.run_harmony(pcs, adata.obs[[batch_key]], batch_key, random_state=seed, verbose=False)
    z = np.asarray(ho.Z_corr)
    if z.shape != pcs.shape:
        z = z.T
    return z.astype(np.float32)


def _peak_rss_bytes() -> int:
    # ru_maxrss is KiB on Linux, bytes on macOS.
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def run(
    slim_path: str | Path,
    results_path: str | Path,
    *,
    n_pcs: int = 30,
    n_neighbors: int = 15,
    resolution: float = 0.5,
    batch_key: str | None = None,
    seed: int = 0,
    device: str = "auto",
    max_value: float | None = 10.0,
    run_id: str | None = None,
    engine: str | None = None,
) -> dict[str, Any]:
    """Run the core embedding/clustering steps on a slim exchange file.

    Returns a summary dict: device, n_cells, n_genes, n_pcs (used), n_clusters,
    timings_s (per step), peak_host_rss_bytes and, on GPU, peak_gpu_pool_bytes.
    """
    if engine is not None and engine not in ENGINES:
        raise ValueError(f"engine must be one of {list(ENGINES)}, got {engine!r}")
    scfast_device = "cpu"
    if engine == "rust":
        if device not in ("auto", "gpu", "cpu"):
            raise ValueError(f"device must be 'auto', 'gpu' or 'cpu', got {device!r}")
        scfast = _import_scfast()  # fail before reading anything
        used = "cpu"
        if device in ("gpu", "auto"):
            # scfast on CUDA when this build has it and a device is visible (docs/scfast-integration.md,
            # "Pipeline wiring"); --device auto quietly stays on the CPU otherwise.
            cuda_ok, cuda_why = scfast_cuda_available(scfast)
            if cuda_ok:
                used, scfast_device = "gpu", "cuda"
            elif device == "gpu":
                raise DeviceError(f"engine 'rust' (scfast) has no GPU backend here ({cuda_why}); "
                                  "use --device cpu or auto, or --engine rapids")
    else:
        used = _resolve_device(device)
        if engine == "rapids" and used not in ("gpu", "cpu-as-gpu"):
            raise DeviceError("engine 'rapids' needs a usable GPU; use --engine scanpy or --device auto/gpu")
        if engine == "scanpy" and used == "gpu":
            used = "cpu"
        if engine is None:
            engine = "rapids" if used == "gpu" else "scanpy"
        if used == "cpu-as-gpu":
            engine = "scanpy"  # test-only override: the scanpy branch runs, reported as device cpu-as-gpu
    timings: dict[str, float] = {}

    @contextmanager
    def timed(name: str):
        t0 = time.perf_counter()
        yield
        timings[name] = round(time.perf_counter() - t0, 4)

    with timed("read"):
        adata = read_exchange(slim_path)
    ex = dict(adata.uns["exchange"])
    if ex.get("mode") != "slim":
        raise ExchangeError(f"gpu_core.run expects a slim exchange file, got mode {ex.get('mode')!r}")
    source_checksum = ex["source_counts_checksum"]
    if run_id is None:
        run_id = ex.get("run_id", "none")
    config_hash = ex.get("config_hash", "none")

    if batch_key is not None:
        if batch_key not in adata.obs.columns:
            raise ValueError(f"batch_key {batch_key!r} is not an obs column")
        if engine == "rust" and _scfast_has_harmony():
            pass  # scfast's Rust Harmony (CPU)
        elif used != "gpu" or engine == "rust":
            _check_harmony_cpu()  # Harmony with harmonypy on the CPU
    harmony_impl = None

    n_comps = int(min(n_pcs, min(adata.n_vars, adata.n_obs) - 1))
    if n_comps < 2:
        raise ValueError(f"too few genes/cells for PCA: {adata.n_obs} cells x {adata.n_vars} genes")
    rep = "X_harmony" if batch_key is not None else "X_pca"
    # Scaling with zero-centering densifies anyway (DESIGN 5.1: ~8 KB/cell at
    # 2,000 HVGs); do it explicitly so both branches start from dense float32.
    if engine != "rust":
        if hasattr(adata.X, "toarray"):
            adata.X = adata.X.toarray()
        adata.X = np.asarray(adata.X, dtype=np.float32)
    gpu_pool_peak = None

    if engine == "rust":
        import scipy.sparse as sp

        # scfast.pca scales each gene implicitly (no dense scaled matrix, no max_value clipping).
        X = sp.csr_matrix(adata.X, dtype=np.float32)
        X.sum_duplicates()
        X.sort_indices()
        with timed("pca"):
            pcs, _loadings, _sdev = scfast.pca(
                X, n_comps=n_comps, seed=seed, oversample=30, power_iters=10, device=scfast_device
            )
        adata.obsm["X_pca"] = np.asarray(pcs, dtype=np.float32)
        if batch_key is not None:
            with timed("harmony"):
                if _scfast_has_harmony():  # Rust Harmony (harmony 2.x algorithm and defaults)
                    adata.obsm["X_harmony"] = scfast.harmony(adata.obsm["X_pca"],
                                                             adata.obs[batch_key].astype(str).to_numpy(), seed=seed)
                    harmony_impl = "scfast"
                else:
                    adata.obsm["X_harmony"] = _harmony_cpu(adata, batch_key, seed)
                    harmony_impl = "harmonypy"
        emb = np.ascontiguousarray(adata.obsm[rep], dtype=np.float32)
        k = int(min(n_neighbors, adata.n_obs))
        with timed("neighbors"):
            knn_idx, knn_dist = scfast.knn(emb, k=k, seed=seed, device=scfast_device)
            snn = scfast.snn(knn_idx)
            n = adata.n_obs
            rows = np.repeat(np.arange(n), k - 1)
            dist = sp.csr_matrix((np.asarray(knn_dist)[:, 1:].ravel(), (rows, np.asarray(knn_idx)[:, 1:].ravel())), shape=(n, n))
            conn = snn.tolil()
            conn.setdiag(0)
            conn = conn.tocsr()
            conn.eliminate_zeros()
            adata.obsp["distances"] = dist.astype(np.float32)
            adata.obsp["connectivities"] = conn.astype(np.float32)
        with timed("leiden"):
            labels = scfast.leiden(snn, resolution=resolution, seed=seed, n_iterations=2)
            adata.obs[CLUSTER_KEY] = pd.Categorical([str(int(x)) for x in labels])
        with timed("umap"):
            import umap

            adata.obsm["X_umap"] = np.asarray(
                umap.UMAP(n_neighbors=k, random_state=seed, n_components=2).fit_transform(emb), dtype=np.float32
            )
    elif used == "gpu":
        import cupy as cp
        import rapids_singlecell as rsc

        pool = cp.get_default_memory_pool()
        with timed("to_gpu"):
            rsc.get.anndata_to_GPU(adata)
        with timed("scale"):
            rsc.pp.scale(adata, max_value=max_value)
        with timed("pca"):
            rsc.pp.pca(adata, n_comps=n_comps, random_state=seed)
        if batch_key is not None:
            with timed("harmony"):
                rsc.pp.harmony_integrate(adata, key=batch_key, basis="X_pca", adjusted_basis="X_harmony")
            harmony_impl = "rapids"
        with timed("neighbors"):
            rsc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=rep, random_state=seed)
        with timed("leiden"):
            rsc.tl.leiden(adata, resolution=resolution, random_state=seed, key_added=CLUSTER_KEY)
        with timed("umap"):
            rsc.tl.umap(adata, random_state=seed)
        gpu_pool_peak = int(pool.total_bytes())
        with timed("to_cpu"):
            rsc.get.anndata_to_CPU(adata, convert_all=True)
    else:
        import scanpy as sc

        with timed("scale"):
            sc.pp.scale(adata, max_value=max_value)
        with timed("pca"):
            sc.pp.pca(adata, n_comps=n_comps, random_state=seed, mask_var=None)
        if batch_key is not None:
            with timed("harmony"):
                adata.obsm["X_harmony"] = _harmony_cpu(adata, batch_key, seed)
            harmony_impl = "harmonypy"
        with timed("neighbors"):
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=rep, random_state=seed)
        with timed("leiden"):
            sc.tl.leiden(
                adata,
                resolution=resolution,
                random_state=seed,
                flavor="igraph",
                n_iterations=2,
                directed=False,
                key_added=CLUSTER_KEY,
            )
        with timed("umap"):
            sc.tl.umap(adata, random_state=seed)

    with timed("write"):
        import anndata as ad

        keep_obsm = {k: np.asarray(adata.obsm[k], dtype=np.float32) for k in ("X_pca", "X_harmony", "X_umap") if k in adata.obsm}
        res = ad.AnnData(
            obs=adata.obs.copy(),
            obsm=keep_obsm,
            obsp={k: adata.obsp[k] for k in ("connectivities", "distances")},
        )
        res.obs[CLUSTER_KEY] = res.obs[CLUSTER_KEY].astype("category")
        res.uns["gpu_core"] = {
            "device": used,
            "n_pcs": n_comps,
            "n_neighbors": int(n_neighbors),
            "resolution": float(resolution),
            "batch_key": batch_key or "",
            "seed": int(seed),
            "max_value": float(max_value) if max_value is not None else -1.0,
            "use_rep": rep,
            "engine": engine,
            "scfast_device": scfast_device if engine == "rust" else "",
        }
        write_exchange(
            res,
            results_path,
            "results",
            run_id=str(run_id),
            stage=STAGE,
            config_hash=str(config_hash),
            source_counts_checksum=source_checksum,
            cluster_key=CLUSTER_KEY,
        )

    summary: dict[str, Any] = {
        "device": used,
        "engine": engine,
        "n_cells": int(res.n_obs),
        "n_genes": int(adata.n_vars),
        "n_pcs": n_comps,
        "n_clusters": int(res.obs[CLUSTER_KEY].nunique()),
        "harmony": batch_key is not None,
        "harmony_impl": harmony_impl,
        "timings_s": timings,
        "scfast_device": scfast_device if engine == "rust" else None,
        "total_s": round(sum(timings.values()), 4),
        "peak_host_rss_bytes": _peak_rss_bytes(),
        "results_path": str(results_path),
        "source_counts_checksum": source_checksum,
    }
    if gpu_pool_peak is not None:
        summary["peak_gpu_pool_bytes"] = gpu_pool_peak
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m cellwarden.gpu_core", description=__doc__.split("\n")[0])
    p.add_argument("--in", dest="slim_path", required=True, help="slim exchange .h5ad")
    p.add_argument("--out", dest="results_path", required=True, help="results exchange .h5ad to write")
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--resolution", type=float, default=0.5)
    p.add_argument("--batch-key", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    p.add_argument("--engine", choices=ENGINES, default=None,
                   help="rapids | scanpy | rust (scfast); default: by device (gpu -> rapids, cpu -> scanpy)")
    p.add_argument("--max-value", type=float, default=10.0)
    p.add_argument("--run-id", default=None)
    a = p.parse_args(argv)
    try:
        summary = run(
            a.slim_path,
            a.results_path,
            n_pcs=a.n_pcs,
            n_neighbors=a.n_neighbors,
            resolution=a.resolution,
            batch_key=a.batch_key,
            seed=a.seed,
            device=a.device,
            max_value=a.max_value,
            run_id=a.run_id,
            engine=a.engine,
        )
    except (DeviceError, ExchangeError, ValueError, RuntimeError) as e:
        print(json.dumps({"error": type(e).__name__, "message": str(e)}), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
