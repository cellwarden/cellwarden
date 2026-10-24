"""scfast: fast single-cell kernels in Rust for Seurat- and Scanpy-style workflows.

The steps from raw counts to cluster labels (docs/RUST_ENGINE.md section 5)::

    import scfast
    pre = scfast.preprocess(X_counts_csr, n_hvg=2000)                   # normalise + vst HVGs, one pass
    pcs, loadings, sdev = scfast.pca(X_csr, n_comps=30, seed=0)       # no dense scaled matrix
    knn_idx, knn_dist   = scfast.knn(pcs, k=20)                         # self first, Euclidean
    snn                 = scfast.snn(knn_idx, prune=1/15)               # scipy CSR, Jaccard
    labels              = scfast.leiden(snn, resolution=0.5, seed=0, n_iterations=2)
    pcs_corrected       = scfast.harmony(pcs, batch_labels)                   # optional batch integration

Every call releases the GIL while the Rust kernel runs. Errors from the kernels arrive as
``ValueError`` (bad input or parameter) or ``RuntimeError`` (internal failure); a kernel
that is not implemented in this build raises ``NotImplementedError``.

Threads: the kernels use rayon's global pool, sized by ``RAYON_NUM_THREADS`` (default: all
cores). Inputs must not be modified from another thread while a call is running.
"""

from __future__ import annotations

import glob
import math
import operator
import os
import sys
from typing import Any, Literal

import numpy as np


def _find_cugraph_lib() -> None:
    """Point SCFAST_CUGRAPH_LIB at a pip (libcugraph-cu12) or conda (RAPIDS image) cuGraph, so
    GPU Leiden needs no environment variables. The Rust side reads the variable on first use."""
    if os.environ.get("SCFAST_CUGRAPH_LIB"):
        return
    try:
        import libcugraph  # pip wheel; load_library() also loads librmm / libraft it links to

        libcugraph.load_library()
        hits = glob.glob(os.path.join(os.path.dirname(libcugraph.__file__), "lib*", "libcugraph_c.so"))
    except Exception:
        hits = glob.glob(os.path.join(sys.prefix, "lib", "libcugraph_c.so"))
    if hits:
        os.environ["SCFAST_CUGRAPH_LIB"] = hits[0]


def _preload_cuda_libs() -> None:
    """Load this environment's CUDA libraries (conda ``lib/`` or pip ``nvidia/*/lib``) globally,
    so the CUDA backend's dlopen by soname (cudarc dynamic loading) finds them without
    LD_LIBRARY_PATH. Does nothing when the environment ships none (system CUDA is then used)."""
    import ctypes

    site = [p for p in sys.path if p.endswith("site-packages")]
    dirs = [os.path.join(sys.prefix, "lib")] + [d for s in site for d in glob.glob(os.path.join(s, "nvidia", "*", "lib"))]
    for name in ("libnvrtc.so.1[0-9]", "libcublasLt.so.1[0-9]", "libcublas.so.1[0-9]", "libcusolver.so.1[0-9]"):
        for d in dirs:
            hits = glob.glob(os.path.join(d, name))
            if hits:
                try:
                    ctypes.CDLL(hits[0], mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


_find_cugraph_lib()
_preload_cuda_libs()

from . import _scfast  # noqa: E402
from ._scfast import __version__  # noqa: E402

__all__ = ["__version__", "device_info", "harmony", "knn", "leiden", "leiden_device", "pca", "preprocess", "read_10x_h5", "snn", "umap"]

_I32_MAX = int(np.iinfo(np.int32).max)
_U64_MAX = int(np.iinfo(np.uint64).max)
_INDEX_DTYPES = (np.dtype(np.int32), np.dtype(np.int64))
_VALUE_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
_KNN_INDEX_DTYPES = (np.dtype(np.int32), np.dtype(np.int64), np.dtype(np.uint32), np.dtype(np.uint64))


# ---------------------------------------------------------------- argument checks

def _int(name: str, value: Any, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        v = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {v}")
    if maximum is not None and v > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {v}")
    return v


def _float(name: str, value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a number, got bool")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}") from None
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    return v


def _seed(seed: Any) -> int:
    return _int("seed", seed, 0, _U64_MAX)


def _resolve_device(device: Any) -> tuple[str, str]:
    """(requested, resolved) for a PCA / kNN device string (Phase 6).

    ``"auto"`` resolves to ``"cuda:0"`` only when this build has the ``cuda`` feature and a
    device is usable, else ``"cpu"``. An unknown string is ``ValueError``; ``"cuda"`` /
    ``"cuda:<i>"`` without a usable device is ``RuntimeError`` (message starting ``GPU:``).
    """
    if not isinstance(device, str):
        raise ValueError(f"device must be 'auto', 'cpu', 'cuda' or 'cuda:<index>', got {device!r}")
    requested = device.strip().lower()
    try:
        resolved = _scfast.device(requested)
    except ValueError:
        raise ValueError(f"device must be 'auto', 'cpu', 'cuda' or 'cuda:<index>', got {device!r}") from None
    except RuntimeError as e:
        raise RuntimeError(f"device={device!r}: {e}") from None
    return requested, resolved


def _with_cpu_fallback(requested: str, resolved: str, call: Any) -> Any:
    """Run ``call(resolved)``; when the device came from ``"auto"`` and the GPU fails
    (``RuntimeError`` whose message carries ``GPU:``, e.g. out of device memory), warn and
    run again on the CPU. An explicit device is never retried."""
    if resolved == "cpu" or requested != "auto":
        return call(resolved)
    try:
        return call(resolved)
    except RuntimeError as e:
        if "GPU: " not in str(e):
            raise
        import warnings

        warnings.warn(f"scfast: CUDA failed, falling back to the CPU: {e}", RuntimeWarning, stacklevel=3)
        return call("cpu")


def device_info() -> dict[str, Any]:
    """CUDA state of this build (never raises).

    Keys: ``cuda_built`` (compiled with the ``cuda`` feature), ``cuda_available`` (driver and
    NVRTC load and a device is visible), ``device_count``, ``device_name``,
    ``free_gb``, ``total_gb``, ``compute_capability`` (device 0; ``None`` when unavailable),
    ``cuda_error`` (why CUDA is unavailable, else ``None``) and ``auto_device`` (what
    ``device="auto"`` resolves to: ``"cpu"`` or ``"cuda:0"``).
    """
    return dict(_scfast.device_info())


def _describe(x: Any) -> str:
    fmt = getattr(x, "format", None)
    name = type(x).__name__
    return f"{name} (format {fmt!r})" if isinstance(fmt, str) else name


def _csr_parts(X: Any, what: str) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Shape and (indptr, indices, data) of a scipy CSR matrix, ready for the Rust side.

    float32/float64 data and int32/int64 indices are passed through untouched, so Rust
    borrows scipy's own buffers. Anything else is converted (a copy) as documented in pca().
    Structure and canonical order are checked in Rust (O(nnz), GIL released).
    """
    import scipy.sparse as sp

    if not sp.issparse(X) or X.format != "csr":
        hint = f" Convert it with {what}.tocsr() first (a copy)." if sp.issparse(X) else ""
        raise TypeError(
            f"{what} must be a scipy.sparse CSR matrix (csr_matrix or csr_array), got {_describe(X)}.{hint}"
        )
    if len(X.shape) != 2:
        raise ValueError(f"{what} must be 2-D, got shape {X.shape}")
    n_rows, n_cols = (int(s) for s in X.shape)

    indptr, indices, data = X.indptr, X.indices, X.data
    if indptr.dtype not in _INDEX_DTYPES or indices.dtype not in _INDEX_DTYPES or indptr.dtype != indices.dtype:
        # scipy always gives both the same int32/int64 dtype; this only handles hand-made ones.
        if indices.dtype == np.int32 and indptr.size and int(indptr[-1]) <= _I32_MAX:
            indptr = indptr.astype(np.int32)  # n_rows + 1 values: cheap
        else:
            indptr, indices = indptr.astype(np.int64), indices.astype(np.int64)
    if data.dtype not in _VALUE_DTYPES:
        if data.dtype.kind not in "biuf":
            raise TypeError(f"{what} must hold real numbers, got dtype {data.dtype}")
        data = data.astype(np.float64 if data.dtype.kind == "f" and data.dtype.itemsize > 8 else np.float32)
    # no-ops (same buffer) for scipy's own contiguous, native-endian arrays
    indptr = np.ascontiguousarray(indptr, dtype=indptr.dtype.newbyteorder("="))
    indices = np.ascontiguousarray(indices, dtype=indices.dtype.newbyteorder("="))
    data = np.ascontiguousarray(data, dtype=data.dtype.newbyteorder("="))
    return n_rows, n_cols, indptr, indices, data


def _inspect_csr(X: Any) -> dict[str, Any]:
    """Diagnostics: the Rust types and buffer addresses pca() would use for X (no kernel)."""
    return _scfast.csr_view_info(*_csr_parts(X, "X"))


# ---------------------------------------------------------------- public API

def pca(
    X: Any,
    n_comps: int = 30,
    seed: int = 0,
    scale: bool = True,
    oversample: int = 10,
    power_iters: int = 6,
    *,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    clip: float | None = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implicitly scaled PCA, replacing Seurat's ``ScaleData`` + ``RunPCA``.

    Centres (and with ``scale=True`` scales to unit variance) each gene implicitly and runs
    a block Krylov SVD, so the dense cells x genes scaled matrix is never built.

    Parameters
    ----------
    X
        scipy.sparse CSR matrix (``csr_matrix`` or ``csr_array``), cells x genes, usually
        the log-normalised HVG matrix. float32/float64 data with int32/int64 indices is
        read in place (zero copy); other real dtypes are converted to float32 first (a copy).
        Must be canonical (sorted, unique column indices per row); call ``X.sum_duplicates()``
        if not. Other sparse formats and dense arrays raise ``TypeError``. A
        ``cupyx.scipy.sparse`` CSR matrix stays on its GPU (copied device-to-device, gene
        statistics computed there; needs a CUDA ``device`` and no CPU fallback).
    n_comps
        Number of components (Seurat ``npcs``).
    seed
        RNG seed for the SVD start block; the same seed gives the same result.
    scale
        Divide each gene by its standard deviation (n - 1 denominator), as ``ScaleData``.
    oversample, power_iters
        Block Krylov SVD settings: extra directions per block (width ``n_comps +
        oversample``) and iterations (``power_iters + 1`` blocks). The defaults reach
        Seurat parity on PBMC 3k (kNN overlap on the top 10 PCs 0.977; exact SVD 0.9975).
    device
        ``"auto"`` (CUDA when this build has the ``cuda`` feature and a device is usable,
        else CPU), ``"cpu"``, ``"cuda"`` or ``"cuda:<index>"``. With ``"auto"`` a GPU
        failure (e.g. out of device memory) falls back to the CPU with a ``RuntimeWarning``;
        an explicit ``"cuda"`` that cannot be used raises ``RuntimeError``. On CUDA the
        sparse products run on the device in single precision, so results match the CPU to
        the parity thresholds rather than bitwise. See :func:`device_info`.
    clip
        Seurat's ``ScaleData(scale.max)``: scaled values above it are clipped, exactly, as a
        sparse correction to the implicit operator. ``None`` leaves them unclipped.

    Returns
    -------
    embeddings : float32 array (cells, n_comps)
        Cell embeddings U * d (Seurat ``Embeddings``, Scanpy ``obsm["X_pca"]``).
    loadings : float32 array (genes, n_comps)
        Gene loadings V (Seurat ``Loadings``, Scanpy ``varm["PCs"]``).
    stdev : float32 array (n_comps,)
        Component standard deviations d / sqrt(cells - 1) (Seurat ``Stdev``).
    """
    requested, resolved = _resolve_device(device)
    if _on_gpu(X):
        return _pca_gpu(X, _int("n_comps", n_comps, 1), _seed(seed), bool(scale), _int("oversample", oversample),
                        _int("power_iters", power_iters), resolved, None if clip is None else _float("clip", clip))
    n_rows, n_cols, indptr, indices, data = _csr_parts(X, "X")
    n_comps = _int("n_comps", n_comps, 1)
    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"X is empty (shape {(n_rows, n_cols)})")
    if n_comps > min(n_rows, n_cols):
        raise ValueError(f"n_comps ({n_comps}) must not exceed min(cells, genes) = {min(n_rows, n_cols)}")
    seed, scale = _seed(seed), bool(scale)
    oversample, power_iters = _int("oversample", oversample), _int("power_iters", power_iters)
    return _with_cpu_fallback(requested, resolved, lambda dev: _scfast.pca_csr(
        n_rows, n_cols, indptr, indices, data, n_comps, seed, scale, oversample, power_iters, dev,
        None if clip is None else _float("clip", clip),
    ))


def knn(
    X: Any,
    k: int = 20,
    method: Literal["auto", "exact", "kdtree", "hnsw"] = "auto",
    seed: int = 0,
    auto_threshold: int = 50_000,
    m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 64,
    *,
    device: Literal["auto", "cpu", "cuda"] = "auto",
    eps: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """k nearest neighbours (Euclidean), replacing the kNN part of Seurat's ``FindNeighbors``.

    Parameters
    ----------
    X
        Dense 2-D array, cells x dims (e.g. the first 30 PCs). Used as float32 C-order; any
        other real dtype or memory layout is converted first. The kernel takes its own
        copy of the (small) embedding matrix in any case.
    k
        Neighbours per cell INCLUDING the cell itself (Seurat ``k.param``), at least 2.
    method
        ``"exact"`` (exact: a k-d tree up to 16 dims, blocked brute force above),
        ``"kdtree"`` (approximate: the k-d tree search with the (1 + ``eps``) pruning
        rule; deterministic), ``"hnsw"`` (approximate, hnsw_rs) or ``"auto"``: exact
        when there are fewer than ``auto_threshold`` cells, else ``"kdtree"`` up to 16
        dims and HNSW with ``m``, ``ef_construction`` and ``ef_search`` above.
    seed
        Seeds the HNSW insertion order. Exact results are deterministic; HNSW results are
        not bit-reproducible across runs (hnsw_rs inserts in parallel).
    device
        ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"cuda:<index>"``; see :func:`pca`. On CUDA
        the search is exact brute force (fused distance + top-k kernel) at every size, ``method`` is
        ignored and ``k`` must be at most 129.
    eps
        ``"kdtree"`` skips a tree node once it is farther than the current k-th distance
        divided by ``1 + eps``; 0 is exact. 0.7 keeps about 0.95 of the exact neighbours
        (100k-1.3M cells, 10 PCs).

    Returns
    -------
    indices : int32 array (cells, k)
        Row i starts with i itself, then the k - 1 nearest other cells by increasing
        distance (ties by lower index).
    distances : float32 array (cells, k)
        Euclidean distances, 0 in the first column.
    """
    import scipy.sparse as sp

    requested, resolved = _resolve_device(device)
    if sp.issparse(X):
        raise TypeError("X must be a dense array (e.g. PCA embeddings), got a scipy sparse matrix")
    a = np.asarray(X)
    if a.ndim != 2:
        raise ValueError(f"X must be 2-D (cells x dims), got {a.ndim}-D with shape {a.shape}")
    if a.dtype.kind not in "biuf":
        raise TypeError(f"X must hold real numbers, got dtype {a.dtype}")
    n, d = a.shape
    if n == 0 or d == 0:
        raise ValueError(f"X is empty (shape {a.shape})")
    k = _int("k", k, 2)  # the cell itself plus at least one neighbour
    if k > n:
        raise ValueError(f"k ({k}) must not exceed the number of cells ({n}); k counts the cell itself")
    if method not in ("auto", "exact", "kdtree", "hnsw"):
        raise ValueError(f"method must be 'auto', 'exact', 'kdtree' or 'hnsw', got {method!r}")
    a = np.ascontiguousarray(a, dtype=np.float32)  # no copy when already float32 C-order
    seed, auto_threshold = _seed(seed), _int("auto_threshold", auto_threshold)
    m, ef_construction, ef_search = _int("m", m, 2), _int("ef_construction", ef_construction, 1), _int("ef_search", ef_search, 1)
    eps = float(eps)
    if not (np.isfinite(eps) and eps >= 0):
        raise ValueError(f"eps must be finite and >= 0, got {eps}")
    return _with_cpu_fallback(requested, resolved, lambda dev: _scfast.knn(
        a, k, method, seed, auto_threshold, m, ef_construction, ef_search, dev, eps,
    ))


def snn(indices: Any, prune: float = 1 / 15, keep_diagonal: bool = True) -> Any:
    """Shared-nearest-neighbour graph, replacing the SNN part of Seurat's ``FindNeighbors``.

    The weight between cells i and j is the Jaccard index of their neighbour sets,
    s / (2k - s) with s = |N(i) & N(j)|, and weights below ``prune`` are dropped
    (Seurat ``prune.SNN``). Any two cells sharing a neighbour are connected, not only kNN
    pairs.

    Parameters
    ----------
    indices
        Integer array (cells, k) of neighbour indices with the cell itself included, as
        returned by :func:`knn` (Seurat's ``Indices()`` also works). Each row is used as the
        neighbour set exactly as given; a cell listed twice in a row is an error. A CuPy
        array builds the graph on its GPU (Seurat's A A^T, as a cuSPARSE product; duplicates
        are not checked there) and returns a ``cupyx.scipy.sparse.csr_matrix``.
    prune
        Jaccard cut-off in [0, 1].
    keep_diagonal
        Keep the diagonal (weight 1), as Seurat's ``RNA_snn`` does.

    Returns
    -------
    scipy.sparse.csr_matrix, float32, (cells, cells)
        Symmetric, both directions of every edge stored, sorted column indices.
    """
    import scipy.sparse as sp

    if sp.issparse(indices):
        raise TypeError("indices must be a dense integer array (cells, k), got a scipy sparse matrix")
    idx = indices if _on_gpu(indices) else np.asarray(indices)
    if idx.ndim != 2:
        raise ValueError(f"indices must be 2-D (cells, k), got {idx.ndim}-D with shape {idx.shape}")
    if idx.dtype.kind not in "iu":
        raise TypeError(f"indices must be an integer array, got dtype {idx.dtype}")
    n, k = idx.shape
    if n == 0 or k == 0:
        raise ValueError(f"indices is empty (shape {idx.shape})")
    if n > _I32_MAX:
        raise ValueError(f"{n} cells exceed the int32 index range")
    prune = _float("prune", prune)
    if not 0.0 <= prune <= 1.0:
        raise ValueError(f"prune must be in [0, 1], got {prune}")
    if _on_gpu(idx):
        return _snn_gpu(idx, prune, bool(keep_diagonal))
    if idx.dtype not in _KNN_INDEX_DTYPES:
        idx = idx.astype(np.int64)
    idx = np.ascontiguousarray(idx, dtype=idx.dtype.newbyteorder("="))
    indptr, col, weights = _scfast.snn(idx, prune, bool(keep_diagonal))
    return sp.csr_matrix((weights, col, indptr), shape=(n, n))


def _on_gpu(a: Any) -> bool:
    return type(a).__module__.split(".")[0] in ("cupy", "cupyx")


def _pca_gpu(X: Any, n_comps: int, seed: int, scale: bool, oversample: int, power_iters: int, device: str,
             clip: float | None) -> Any:
    import cupy as cp
    import cupyx.scipy.sparse as csp

    if not csp.isspmatrix_csr(X):
        raise TypeError(f"X on the GPU must be a cupyx.scipy.sparse CSR matrix, got {type(X).__name__}")
    n_rows, n_cols = X.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"X is empty (shape {(n_rows, n_cols)})")
    if n_comps > min(n_rows, n_cols):
        raise ValueError(f"n_comps ({n_comps}) must not exceed min(cells, genes) = {min(n_rows, n_cols)}")
    if not X.has_canonical_format:
        X = X.copy()
        X.sum_duplicates()
    indptr = X.indptr if X.indptr.dtype in (cp.int32, cp.int64) else X.indptr.astype(cp.int64)
    indices, data = X.indices.astype(cp.int32, copy=False), X.data.astype(cp.float32, copy=False)
    cp.cuda.get_current_stream().synchronize()
    return _scfast.pca_csr_device(n_rows, n_cols, int(data.size), indptr.data.ptr, indptr.dtype == cp.int64,
                                  indices.data.ptr, data.data.ptr, n_comps, seed, scale, oversample, power_iters, device,
                                  clip)


def _snn_gpu(idx: Any, prune: float, keep_diagonal: bool) -> Any:
    import cupy as cp
    import cupyx.scipy.sparse as csp

    n, k = idx.shape
    a = csp.csr_matrix((cp.ones(n * k, dtype=cp.float32), cp.ascontiguousarray(idx, dtype=cp.int32).ravel(),
                        cp.arange(0, n * k + 1, k, dtype=cp.int32)), shape=(n, n))
    g = a @ a.T  # shared-neighbour counts
    s = g.data.astype(cp.float64)
    w = s / (k + (k - s))  # Seurat's expression, pruned in f64 as on the CPU
    g.data = cp.where(w < prune, 0.0, w).astype(cp.float32)
    if not keep_diagonal:
        g.setdiag(0)
    g.eliminate_zeros()
    g.sort_indices()
    return g


def leiden_device(device: str = "auto") -> str:
    """The device :func:`leiden` would run on for ``device``: ``"cpu"`` or ``"cuda:<index>"``.

    ``"auto"`` resolves to ``"cuda:0"`` only when this build of scfast has the ``cugraph``
    feature, ``libcugraph_c.so`` loads (``SCFAST_CUGRAPH_LIB`` overrides the search) and a
    CUDA device is usable; otherwise ``"cpu"``. An explicit ``"cuda"`` / ``"cuda:<index>"``
    that is not usable raises ``RuntimeError`` naming the reason.
    """
    if not isinstance(device, str):
        raise TypeError(f"device must be a string, got {type(device).__name__}")
    return _scfast.leiden_device(device)["resolved"]


def leiden(
    graph: Any,
    resolution: float = 0.5,
    seed: int = 0,
    n_iterations: int = 2,
    device: Literal["auto", "cpu", "cuda"] | str = "auto",
) -> np.ndarray:
    """Leiden clustering (RB / configuration objective), replacing ``FindClusters(algorithm = 4)``.

    Parameters
    ----------
    graph
        Square scipy.sparse CSR matrix (``csr_matrix`` or ``csr_array``): a symmetric
        weighted adjacency with both directions of every edge stored, such as the output
        of :func:`snn`, Seurat's ``RNA_snn`` or Scanpy's ``obsp["connectivities"]``.
        Weights must be finite and non-negative; they are used as float32. A diagonal is
        allowed and treated as Seurat does. Must be canonical (``graph.sum_duplicates()``).
        A ``cupyx.scipy.sparse`` matrix (e.g. :func:`snn` of CuPy indices) stays on its GPU
        and goes to cuGraph as is, without the checks (``device`` must then be CUDA).
    resolution
        RB resolution (Seurat ``resolution``); larger gives more clusters.
    seed
        Random seed; the same seed and graph give identical labels.
    n_iterations
        Leiden iterations (Seurat ``n.iter``), at least 1. cuGraph has no such knob: it runs to ``max_level`` 100.
    device
        ``"auto"`` (cuGraph Leiden when this build has the ``cugraph`` feature and a CUDA
        device is usable, else the CPU), ``"cpu"``, ``"cuda"`` or ``"cuda:<index>"`` (an
        error when cuGraph is not available). See :func:`leiden_device`. Labels follow the
        same numbering on both backends; they are compared by ARI, not expected identical.

    Returns
    -------
    int32 array (cells,)
        Cluster labels 0..c-1, numbered by decreasing cluster size (ties by lowest cell
        index), like Seurat's ``seurat_clusters`` levels.
    """
    if not isinstance(device, str):
        raise TypeError(f"device must be a string, got {type(device).__name__}")
    if _on_gpu(graph):
        return _leiden_gpu(graph, _float("resolution", resolution), _seed(seed), device)
    n_rows, n_cols, indptr, indices, data = _csr_parts(graph, "graph")
    if n_rows != n_cols:
        raise ValueError(f"graph must be square (cells x cells), got shape {(n_rows, n_cols)}")
    if n_rows > _I32_MAX:
        raise ValueError(f"{n_rows} cells exceed the int32 index range")
    resolution = _float("resolution", resolution)
    if resolution < 0:
        raise ValueError(f"resolution must be non-negative, got {resolution}")
    return _scfast.leiden_csr(
        n_rows, indptr, indices, data, resolution, _seed(seed), _int("n_iterations", n_iterations, 1), device
    )


def _leiden_gpu(graph: Any, resolution: float, seed: int, device: str) -> np.ndarray:
    import cupy as cp

    n, n_cols = graph.shape
    if n != n_cols:
        raise ValueError(f"graph must be square (cells x cells), got shape {(n, n_cols)}")
    if device == "auto":
        device = "cuda"
    coo = graph.tocoo()
    row, col = coo.row.astype(cp.int32, copy=False), coo.col.astype(cp.int32, copy=False)
    w = coo.data.astype(cp.float32, copy=False)
    cp.cuda.get_current_stream().synchronize()
    return _scfast.leiden_coo_device(n, int(w.size), row.data.ptr, col.data.ptr, w.data.ptr, resolution, seed, device)


def preprocess(
    X_counts: Any,
    n_hvg: int = 2000,
    scale_factor: float = 1e4,
    span: float = 0.3,
    clip: float | None = None,
) -> dict[str, Any]:
    """Fused log-normalisation and vst variable-gene statistics, replacing Seurat's
    ``NormalizeData`` + ``FindVariableFeatures(selection.method = "vst")`` (Scanpy's
    ``normalize_total`` + ``log1p`` + ``highly_variable_genes(flavor="seurat_v3")`` on the
    same statistics) in one streaming pass over the counts.

    Parameters
    ----------
    X_counts
        scipy.sparse CSR matrix (``csr_matrix`` or ``csr_array``), cells x genes, raw counts.
        float32/float64 data with int32/int64 indices is read in place (zero copy); other
        real dtypes (integer counts included) are converted to float32 first (a copy). Must
        be canonical (``X.sum_duplicates()``). Counts must be finite and non-negative.
    n_hvg
        Number of variable genes (Seurat ``nfeatures``), capped at the number of genes.
    scale_factor
        ``NormalizeData(scale.factor)``: values are ``log1p(x / library_size * scale_factor)``.
    span
        Loess span of the mean-variance trend (Seurat ``loess.span``).
    clip
        Clip for the standardised values (Seurat ``clip.max``); ``None`` means
        ``sqrt(n_cells)``.

    Returns
    -------
    dict
        ``"data"``: scipy.sparse.csr_matrix, float32, cells x genes, the log-normalised
        values (same sparsity pattern as the input, built on its index arrays).
        ``"library_size"``: float64 (cells,). ``"mean"``, ``"variance"``,
        ``"variance_expected"``, ``"variance_standardized"``: float64 (genes,), the vst
        columns (``vst.*`` in Seurat; constant genes get 0 expected and standardised
        variance). ``"hvg"``: int32 (n_hvg,) gene indices, most variable first.
        ``"hvg_mean"``, ``"hvg_sd"``: float64 (n_hvg,), mean and sample standard deviation
        of the log-normalised values of each HVG (what ``ScaleData`` / :func:`pca` scale
        with). ``"clip"``: the clip used. ``"X_hvg"``: ``data[:, hvg]`` as a canonical CSR
        matrix (columns in ``hvg`` order), ready for :func:`pca`.

    On the GPU (``X_counts`` a ``cupyx.scipy.sparse`` CSR matrix, or for more than 2^31 - 1
    non-zeros a tuple ``(data, indices, indptr, shape)`` of CuPy arrays: float32 counts,
    int32 columns, int64 row offsets) the passes run on the device and the per-gene vst
    step is the same Rust code, so the HVGs match the CPU result (the counts are integers,
    so the per-gene sums are exact). The result then has ``"X_hvg"`` as a cupyx CSR matrix
    and ``"library_size"`` as a CuPy array, and no ``"data"``, ``"hvg_mean"`` or ``"hvg_sd"``.

    Notes
    -----
    The loess is R's default ``loess(..., span, degree = 2)`` (``surface =
    "interpolate"``) that Seurat calls; expected variances match R to ~1e-11 (log10), so
    the HVG set is Seurat's.
    """
    import scipy.sparse as sp

    if isinstance(X_counts, tuple) or _on_gpu(X_counts):
        return _preprocess_gpu(X_counts, n_hvg, scale_factor, span, clip)
    n_rows, n_cols, indptr, indices, data = _csr_parts(X_counts, "X_counts")
    n_hvg = _int("n_hvg", n_hvg, 1)
    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"X_counts is empty (shape {(n_rows, n_cols)})")
    n_hvg = min(n_hvg, n_cols)
    scale_factor = _float("scale_factor", scale_factor)
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    span = _float("span", span)
    if span <= 0:
        raise ValueError(f"span must be positive, got {span}")
    if clip is not None:
        clip = _float("clip", clip)
        if clip <= 0:
            raise ValueError(f"clip must be positive, got {clip}")
    out = _scfast.preprocess_csr(n_rows, n_cols, indptr, indices, data, n_hvg, scale_factor, span, clip)
    out["data"] = _csr_validated(out["data"], indices, indptr, (n_rows, n_cols))
    hvg_indptr, hvg_indices = out.pop("hvg_indptr"), out.pop("hvg_indices")
    if int(hvg_indptr[-1]) <= _I32_MAX:
        hvg_indptr = hvg_indptr.astype(np.int32)
    else:
        hvg_indices = hvg_indices.astype(np.int64)
    out["X_hvg"] = _csr_validated(out.pop("hvg_data"), hvg_indices, hvg_indptr, (n_rows, n_hvg))
    return out


_PREPROCESS_KERNELS = r"""
// Warp per row: library size, and per-gene sum, sum of squares, count and max of the counts.
extern "C" __global__ void row_stats(const long long* indptr, const int* indices, const float* x, int n,
                                     double* lib, double* sum, double* sumsq,
                                     unsigned long long* cnt, int* mx) {
  const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n) return;
  double acc = 0.0;
  for (long long t = indptr[warp] + lane; t < indptr[warp + 1]; t += 32) {
    const float v = x[t];
    const int j = indices[t];
    acc += v;
    atomicAdd(&sum[j], (double)v);
    atomicAdd(&sumsq[j], (double)v * (double)v);
    atomicAdd(&cnt[j], 1ull);
    atomicMax(&mx[j], __float_as_int(v));  // counts are non-negative: int order = float order
  }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) lib[warp] = acc;
}

// Sum over stored entries of flagged genes of min(clip, (x - mean) / sd)^2.
extern "C" __global__ void clipped_sums(const int* indices, const float* x, unsigned long long nnz,
                                        const double* mean, const double* sd, const unsigned char* flag,
                                        double clip, double* out) {
  for (unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; t < nnz;
       t += (unsigned long long)gridDim.x * blockDim.x) {
    const int j = indices[t];
    if (!flag[j]) continue;
    const double z = fmin(((double)x[t] - mean[j]) / sd[j], clip);
    atomicAdd(&out[j], z * z);
  }
}

// Warp per row: number of stored entries in HVG columns (rank >= 0).
extern "C" __global__ void hvg_count(const long long* indptr, const int* indices, int n, const int* rank,
                                     long long* cnt) {
  const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n) return;
  int c = 0;
  for (long long t = indptr[warp] + lane; t < indptr[warp + 1]; t += 32) c += rank[indices[t]] >= 0;
  for (int o = 16; o > 0; o >>= 1) c += __shfl_down_sync(0xffffffffu, c, o);
  if (lane == 0) cnt[warp] = c;
}

// Block per row: the HVG entries renumbered by rank and sorted by it (each kept entry's
// position is the number of kept entries of the row with a smaller rank), values
// log1p(x * scale_factor / lib) as NormalizeData. Shared memory: 8 bytes per HVG.
extern "C" __global__ void hvg_fill(const long long* indptr, const int* indices, const float* x,
                                    const double* lib, double scale_factor, const int* rank,
                                    const long long* out_ptr, int* out_idx, float* out_val) {
  extern __shared__ int sh[];
  __shared__ int k;
  const int i = blockIdx.x;
  const long long base = out_ptr[i];
  const int m = (int)(out_ptr[i + 1] - base);
  int* r = sh;
  float* v = (float*)(sh + m);
  if (threadIdx.x == 0) k = 0;
  __syncthreads();
  const double factor = lib[i] > 0.0 ? scale_factor / lib[i] : 0.0;
  for (long long t = indptr[i] + threadIdx.x; t < indptr[i + 1]; t += blockDim.x) {
    const int q = rank[indices[t]];
    if (q < 0) continue;
    const int s = atomicAdd(&k, 1);
    r[s] = q;
    v[s] = (float)log1p((double)x[t] * factor);
  }
  __syncthreads();
  for (int s = threadIdx.x; s < m; s += blockDim.x) {
    int pos = 0;
    for (int u = 0; u < m; ++u) pos += r[u] < r[s];
    out_idx[base + pos] = r[s];
    out_val[base + pos] = v[s];
  }
}
"""


def _preprocess_gpu(X: Any, n_hvg: int, scale_factor: float, span: float, clip: float | None) -> dict[str, Any]:
    import cupy as cp
    import cupyx.scipy.sparse as csp

    if isinstance(X, tuple):
        data, indices, indptr, (n_rows, n_cols) = X
    else:
        if not csp.isspmatrix_csr(X):
            raise TypeError(f"X_counts on the GPU must be a cupyx.scipy.sparse CSR matrix, got {type(X).__name__}")
        if not X.has_canonical_format:
            X = X.copy()
            X.sum_duplicates()
        data, indices, indptr, (n_rows, n_cols) = X.data, X.indices, X.indptr, X.shape
    data = data.astype(cp.float32, copy=False)
    indices = indices.astype(cp.int32, copy=False)
    indptr = indptr.astype(cp.int64, copy=False)
    n_hvg = min(_int("n_hvg", n_hvg, 1), n_cols)
    if n_rows == 0 or n_cols == 0:
        raise ValueError(f"X_counts is empty (shape {(n_rows, n_cols)})")
    clip = float(np.sqrt(n_rows)) if clip is None else _float("clip", clip)
    mod = cp.RawModule(code=_PREPROCESS_KERNELS)
    nnz = int(data.size)
    warps = ((n_rows * 32 + 255) // 256,)
    lib = cp.empty(n_rows, cp.float64)
    s, s2 = cp.zeros(n_cols, cp.float64), cp.zeros(n_cols, cp.float64)
    cnt, mx = cp.zeros(n_cols, cp.uint64), cp.zeros(n_cols, cp.int32)
    mod.get_function("row_stats")(warps, (256,), (indptr, indices, data, np.int32(n_rows), lib, s, s2, cnt, mx))
    v = _scfast.vst_from_sums(cp.asnumpy(s), cp.asnumpy(s2), cp.asnumpy(mx.view(cp.float32)).astype(np.float64),
                              cp.asnumpy(cnt), n_rows, float(span), clip)
    vs, flag = v["variance_standardized"], v["needs_clip"]
    if flag.any():
        out = cp.zeros(n_cols, cp.float64)
        mod.get_function("clipped_sums")((4096,), (256,), (indices, data, np.uint64(nnz), cp.asarray(v["mean"]),
                                                           cp.asarray(v["sd"]), cp.asarray(flag.astype(np.uint8)),
                                                           np.float64(clip), out))
        clipped, nz = cp.asnumpy(out), cp.asnumpy(cnt).astype(np.float64)
        z0 = np.divide(v["mean"], v["sd"], out=np.zeros(n_cols), where=flag)
        vs = vs.copy()
        vs[flag] = (clipped[flag] + (n_rows - nz[flag]) * z0[flag] ** 2) / (n_rows - 1)
    hvg = np.lexsort((np.arange(n_cols), -vs))[:n_hvg].astype(np.int32)  # ties by lower index, as the CPU
    rank = np.full(n_cols, -1, np.int32)
    rank[hvg] = np.arange(n_hvg, dtype=np.int32)
    rank_d = cp.asarray(rank)
    counts = cp.empty(n_rows, cp.int64)
    mod.get_function("hvg_count")(warps, (256,), (indptr, indices, np.int32(n_rows), rank_d, counts))
    out_ptr = cp.concatenate([cp.zeros(1, cp.int64), cp.cumsum(counts)])
    kept = int(out_ptr[-1])
    out_idx, out_val = cp.empty(kept, cp.int32), cp.empty(kept, cp.float32)
    mod.get_function("hvg_fill")((n_rows,), (128,), (indptr, indices, data, lib, np.float64(scale_factor), rank_d,
                                                     out_ptr, out_idx, out_val), shared_mem=8 * n_hvg)
    if kept <= _I32_MAX:
        out_ptr = out_ptr.astype(cp.int32)
    x_hvg = csp.csr_matrix((out_val, out_idx, out_ptr), shape=(n_rows, n_hvg))
    return {"library_size": lib, "mean": v["mean"], "variance": v["variance"], "variance_expected": v["variance_expected"],
            "variance_standardized": vs, "hvg": hvg, "clip": clip, "X_hvg": x_hvg}


def _csr_validated(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, shape: tuple[int, int]) -> Any:
    """A canonical scipy CSR around arrays the Rust side has already checked, without copies.

    ``csr_matrix((data, indices, indptr))`` scans int64 index arrays to pick a dtype
    (about 1 s for the 2.6e9 non-zeros of the 1.3M-cell brain data).
    """
    import scipy.sparse as sp

    m = sp.csr_matrix(shape, dtype=data.dtype)
    m.data, m.indices, m.indptr = data, indices, indptr
    m.has_canonical_format = True
    return m


def _read_values(path: str, ds: Any, dtype: Any) -> np.ndarray:
    """A 1-D h5py dataset as ``dtype``; plain-gzip chunks are inflated in parallel in Rust."""
    plain_gzip = (ds.chunks is not None and ds.ndim == 1 and ds.compression == "gzip" and not ds.shuffle
                  and not ds.fletcher32 and ds.scaleoffset is None and ds.dtype.str[1:] in ("i4", "i8", "f4", "f8")
                  and ds.dtype.byteorder in "<=|" and hasattr(ds.id, "chunk_iter"))
    if plain_gzip:
        chunks: list[tuple[int, int, int]] = []
        ds.id.chunk_iter(lambda c: chunks.append((c.chunk_offset[0], c.byte_offset, c.size)))
        chunks.sort()
        info = np.array(chunks, dtype=np.uint64).reshape(-1, 3)
        step = ds.chunks[0]
        if np.array_equal(info[:, 0], np.arange(len(chunks), dtype=np.uint64) * step):
            return _scfast.inflate_chunks(path, np.ascontiguousarray(info[:, 1]), np.ascontiguousarray(info[:, 2]),
                                          step, ds.shape[0], ds.dtype.str[1:], dtype == np.float32, dtype == np.int64)
    return ds[()].astype(dtype, copy=False)


def read_10x_h5(path: str | Any, arrays: bool = False) -> tuple[Any, np.ndarray, np.ndarray]:
    """Read a Cell Ranger ``.h5`` (v2 or v3) as CSR counts, cells x genes.

    The file's genes x cells CSC arrays are a cells x genes CSR as they are, so nothing is
    transposed. Plain-gzip chunks (h5py / Cell Ranger without shuffle) are read and inflated
    in parallel in Rust; other layouts go through h5py.

    Returns
    -------
    (X, barcodes, gene_names)
        X: scipy.sparse.csr_matrix, float32 counts, int32 indices (int64 above 2^31 - 1
        non-zeros); with ``arrays=True`` the tuple ``(data float32, indices int32, indptr
        int64, shape)`` instead (no scipy matrix, int32 indices at any size: what a GPU upload
        wants). Ready for
        :func:`preprocess`. barcodes, gene_names: str arrays. Files with features other
        than Gene Expression raise ``ValueError``.
    """
    import h5py
    import scipy.sparse as sp

    path = str(path)
    with h5py.File(path, "r") as f:
        g = f["matrix"] if "matrix" in f else f[next(iter(f))]
        n_genes, n_cells = (int(s) for s in g["shape"][()])
        if "features" in g:
            names = g["features/name"][()].astype(str)
            if "feature_type" in g["features"] and (g["features/feature_type"][()] != b"Gene Expression").any():
                raise ValueError(f"{path}: has features other than Gene Expression")
        else:
            names = g["gene_names"][()].astype(str)
        indptr = g["indptr"][()]
        # scipy wants indices and indptr in one dtype: int64 once the non-zeros pass int32.
        wide = int(indptr[-1]) > _I32_MAX
        if arrays:
            X = (_read_values(path, g["data"], np.float32), _read_values(path, g["indices"], np.int32),
                 indptr.astype(np.int64), (n_cells, n_genes))
        else:
            index_dtype = np.int64 if wide else np.int32
            parts = (_read_values(path, g["data"], np.float32), _read_values(path, g["indices"], index_dtype),
                     indptr.astype(index_dtype))
            # Wide arrays skip scipy's index-dtype scan (about 1 s at 2.6e9 non-zeros); preprocess
            # and pca check the structure in Rust.
            X = _csr_validated(*parts, (n_cells, n_genes)) if wide else sp.csr_matrix(parts, shape=(n_cells, n_genes))
        barcodes = g["barcodes"][()].astype(str)
    return X, barcodes, names


def _find_ab(spread: float, min_dist: float) -> tuple[float, float]:
    from scipy.optimize import curve_fit

    x = np.linspace(0, spread * 3, 300)
    y = np.where(x < min_dist, 1.0, np.exp(-(x - min_dist) / spread))
    (a, b), _ = curve_fit(lambda x, a, b: 1.0 / (1.0 + a * x ** (2 * b)), x, y)
    return float(a), float(b)


def harmony(
    emb: Any,
    batch_labels: Any,
    *,
    theta: float = 2.0,
    lambda_: float | None = None,
    sigma: float = 0.1,
    nclust: int | None = None,
    max_iter: int = 10,
    seed: int = 0,
    device: Literal["auto", "cpu", "cuda"] | str = "auto",
    init_centroids: Any = None,
    early_stop: bool = True,
    max_iter_cluster: int = 4,
    epsilon_cluster: float = 1e-3,
    epsilon_harmony: float = 1e-2,
    block_size: float = 0.05,
    tau: float = 0.0,
    alpha: float = 0.2,
    batch_prop_cutoff: float = 1e-5,
    kmeans_nstart: int = 1,
    kmeans_iter: int = 10,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Harmony batch integration of an embedding, replacing ``harmony::RunHarmony`` /
    ``harmonypy.run_harmony``.

    The algorithm of the harmony R package 2.x (as in the cellwarden image, 2.0.5): soft k-means
    with a batch-diversity penalty on the cosine-normalised cells, then a per-cluster ridge
    regression of the batch effect (the intercept is not penalised), repeated until the
    objective settles. Defaults are those of ``RunHarmony()`` and ``harmony_options()``.
    Computes in float64; the result is bitwise reproducible for a given ``seed`` and any
    number of threads, but not equal to R's output (harmony draws from R's RNG).

    Parameters
    ----------
    emb
        Dense (cells, dims) array, e.g. ``obsm["X_pca"]``. float32 and float64 are used as
        they are (the result has the same dtype); other real dtypes become float32.
    batch_labels
        One label per cell (any hashable values numpy can sort, e.g. strings or ints, or a
        pandas Categorical); factorised with ``np.unique``. One batch variable.
    theta
        Diversity penalty (0 disables it). ``RunHarmony(theta = 2)``.
    lambda_
        Ridge penalty; ``None`` (harmony 2.x default ``lambda = NULL``) estimates it per
        cluster and batch as ``alpha`` times the expected number of cells. harmony 1.x
        defaulted to 1.
    sigma
        Soft k-means width.
    nclust
        Number of clusters; ``None`` is ``min(round(cells / 30), 100)``.
    max_iter
        Maximum rounds of clustering + correction.
    seed
        Seeds the k-means initialisation and the block order.
    device
        ``"auto"`` or ``"cpu"`` (both run on the CPU: there is no GPU Harmony yet);
        ``"cuda"`` raises ``RuntimeError``.
    init_centroids
        Optional (nclust, dims) starting centroids instead of k-means (e.g. ``t(harmonyObj$Y)``
        from R, for a like-for-like comparison); they are L2-normalised.
    early_stop
        ``False`` runs all ``max_iter`` rounds (``RunHarmony(early_stop = FALSE)``).
    max_iter_cluster, epsilon_cluster, epsilon_harmony, block_size, tau, alpha, batch_prop_cutoff
        ``harmony_options()``.
    kmeans_nstart, kmeans_iter
        Seeded k-means starts and Lloyd steps for the initial centroids (harmony 2.x: 1 and 10).
    return_info
        Also return a dict with ``objective`` (per round, starting with the initial value),
        ``objective_kmeans`` (per clustering pass), ``kmeans_rounds``, ``iters``,
        ``converged``, ``nclust`` and ``device``.

    Returns
    -------
    numpy array (cells, dims)
        The corrected embedding, in the input cell order (``obsm["X_pca_harmony"]``); with
        ``return_info``, a tuple ``(embedding, info)``.
    """
    import scipy.sparse as sp

    if sp.issparse(emb):
        raise TypeError("emb must be a dense array (e.g. PCA embeddings), got a scipy sparse matrix")
    a = np.asarray(emb)
    if a.ndim != 2:
        raise ValueError(f"emb must be 2-D (cells x dims), got {a.ndim}-D with shape {a.shape}")
    if a.dtype.kind not in "biuf":
        raise TypeError(f"emb must hold real numbers, got dtype {a.dtype}")
    n, d = a.shape
    if n == 0 or d == 0:
        raise ValueError(f"emb is empty (shape {a.shape})")
    a = np.ascontiguousarray(a, dtype=a.dtype if a.dtype in _VALUE_DTYPES else np.float32)
    a = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("="))
    labels = np.asarray(batch_labels)
    if labels.ndim != 1 or labels.shape[0] != n:
        raise ValueError(f"batch_labels must be 1-D with one label per cell ({n}), got shape {labels.shape}")
    if labels.dtype.kind == "f" and np.isnan(labels).any():
        raise ValueError("batch_labels contains NaN")
    if labels.dtype == object and any(v is None for v in labels):
        raise ValueError("batch_labels contains None")
    _, codes = np.unique(labels, return_inverse=True)
    codes = np.ascontiguousarray(codes.reshape(-1), dtype=np.uint32)
    n_batches = int(codes.max()) + 1
    requested, resolved = _resolve_device(device)
    if requested == "auto":
        resolved = "cpu"  # no CUDA Harmony yet; "auto" means the CPU until there is one
    init = None
    if init_centroids is not None:
        init = np.ascontiguousarray(np.asarray(init_centroids), dtype=np.float32)
        if init.ndim != 2 or init.shape[1] != d:
            raise ValueError(f"init_centroids must be (nclust, {d}), got shape {init.shape}")
    eps_h = _float("epsilon_harmony", epsilon_harmony) if early_stop else -math.inf
    out, info = _scfast.harmony_dense(
        a, codes, n_batches, _float("theta", theta), None if lambda_ is None else _float("lambda_", lambda_),
        _float("sigma", sigma), None if nclust is None else _int("nclust", nclust, 1), _int("max_iter", max_iter, 1),
        _int("max_iter_cluster", max_iter_cluster, 1), _float("epsilon_cluster", epsilon_cluster), eps_h,
        _float("block_size", block_size), _float("tau", tau), _float("alpha", alpha),
        _float("batch_prop_cutoff", batch_prop_cutoff), _int("kmeans_nstart", kmeans_nstart, 1),
        _int("kmeans_iter", kmeans_iter), _seed(seed), resolved, init,
    )
    return (out, dict(info)) if return_info else out


def umap(
    indices: Any,
    distances: Any,
    init: Any,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_epochs: int | None = None,
    negative_sample_rate: int = 5,
    learning_rate: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """UMAP layout from a kNN graph, replacing ``RunUMAP`` / ``sc.tl.umap``.

    ``indices`` / ``distances`` are the (cells, k) output of :func:`knn` (the cell itself
    in column 0). ``init`` is a (cells, 2) starting layout, e.g. the first two PCs; it is
    rescaled to [0, 10] as umap-learn does. ``n_epochs`` defaults to umap-learn's choice
    (500 below 10,000 cells, else 200). ``min_dist`` 0.5 is Scanpy's default. Returns a
    float32 (cells, 2) array.
    """
    idx = np.ascontiguousarray(indices, dtype=np.int32)
    dist = np.ascontiguousarray(distances, dtype=np.float32)
    start = np.ascontiguousarray(np.asarray(init)[:, :2], dtype=np.float32)
    if idx.ndim != 2 or idx.shape != dist.shape or start.shape != (idx.shape[0], 2):
        raise ValueError(f"indices {idx.shape}, distances {dist.shape} and init {start.shape} do not match")
    if n_epochs is None:
        n_epochs = 500 if idx.shape[0] <= 10_000 else 200
    a, b = _find_ab(_float("spread", spread), _float("min_dist", min_dist))
    return _scfast.umap(idx, dist, start, a, b, _int("n_epochs", n_epochs, 1),
                        _int("negative_sample_rate", negative_sample_rate, 0),
                        _float("learning_rate", learning_rate), _seed(seed))
