"""Type stubs for scfast._scfast, the native (Rust, PyO3) core of scfast.

These are the low-level entry points behind scfast.pca / knn / snn / leiden. They take numpy
arrays that the Python layer has already checked and laid out (C-contiguous, native byte
order, supported dtypes) and do no scipy handling. Every function releases the GIL while
it works. Error mapping:

- invalid input or parameter (scfast_core::Error::InvalidInput / InvalidParameter) -> ValueError
- failure inside a dependency (scfast_core::Error::Backend) -> RuntimeError
- a Rust panic -> RuntimeError; an unimplemented kernel -> NotImplementedError
"""

from typing import Any

import numpy as np
import numpy.typing as npt

__version__: str
"""Version of the Rust crate (workspace version in rust/Cargo.toml)."""

def pca_csr(
    n_rows: int,
    n_cols: int,
    indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64],
    indices: npt.NDArray[np.int32] | npt.NDArray[np.int64],
    data: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    n_comps: int,
    seed: int,
    scale: bool,
    oversample: int,
    power_iters: int,
    device: str = "cpu",
    clip: float | None = None,
    /,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Implicitly scaled PCA of a CSR matrix (cells x genes) given as raw arrays.

    device is "cpu", "auto", "cuda" or "cuda:<index>" (Phase 6); an unusable CUDA request
    raises RuntimeError("GPU: ...").

    The three arrays are borrowed without copying: indptr and indices must share one dtype
    (int32 or int64), data is float32 or float64, all C-contiguous. Rows must be canonical
    (strictly increasing column indices).

    Returns (embeddings (n_rows, k), loadings (n_cols, k), stdev (k,)), all float32.
    """

def pca_csr_device(
    n_rows: int,
    n_cols: int,
    nnz: int,
    indptr: int,
    indptr_i64: bool,
    indices: int,
    data: int,
    n_comps: int,
    seed: int,
    scale: bool,
    oversample: int,
    power_iters: int,
    device: str,
    clip: float | None = None,
    /,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """pca_csr for a canonical CSR matrix already on the GPU (device pointers)."""

def csr_view_info(
    n_rows: int,
    n_cols: int,
    indptr: npt.NDArray[Any],
    indices: npt.NDArray[Any],
    data: npt.NDArray[Any],
    /,
) -> dict[str, Any]:
    """Run pca_csr's dtype dispatch and validation without the kernel.

    Returns {"index_type": "i32" | "i64", "value_type": "f32" | "f64", "indptr_ptr",
    "indices_ptr", "data_ptr" (addresses of the borrowed buffers), "n_rows", "n_cols", "nnz"}.
    Used by the tests to prove the CSR input is read in place.
    """

def knn(
    x: npt.NDArray[np.float32],
    k: int,
    method: str,
    seed: int,
    auto_threshold: int,
    m: int,
    ef_construction: int,
    ef_search: int,
    device: str = "cpu",
    /,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """k nearest neighbours of each row of a C-contiguous float32 (n, d) array.

    device is "cpu", "auto", "cuda" or "cuda:<index>"; on CUDA method is ignored.

    method is "exact", "hnsw" or "auto" (exact below auto_threshold rows, else HNSW with
    m / ef_construction / ef_search). Non-finite values raise ValueError.

    Returns (indices int32 (n, k), distances float32 (n, k)), the row itself first.
    """

def snn(
    indices: npt.NDArray[np.int32] | npt.NDArray[np.int64] | npt.NDArray[np.uint32] | npt.NDArray[np.uint64],
    prune: float,
    keep_diagonal: bool,
    /,
) -> tuple[npt.NDArray[np.int32] | npt.NDArray[np.int64], npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Shared-nearest-neighbour (Jaccard) graph from C-contiguous (n, k) kNN indices.

    Returns the CSR arrays (indptr, indices, weights) of the full symmetric n x n graph;
    indptr is int32 when the number of stored entries fits, else int64.
    """

def leiden_csr(
    n: int,
    indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64],
    indices: npt.NDArray[np.int32] | npt.NDArray[np.int64],
    data: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    resolution: float,
    seed: int,
    n_iterations: int,
    /,
    device: str = "cpu",
) -> npt.NDArray[np.int32]:
    """Leiden (RB objective) on a square, symmetric, canonical CSR graph given as raw arrays.

    The graph is copied once into the clustering library's own buffers. Returns int32
    labels numbered by decreasing cluster size. device: "cpu", "auto", "cuda" or
    "cuda:<index>" (cuGraph, feature `cugraph`); an unusable explicit CUDA device raises
    RuntimeError.
    """

def leiden_coo_device(
    n: int,
    nnz: int,
    src: int,
    dst: int,
    weights: int,
    resolution: float,
    seed: int,
    device: str,
    /,
) -> npt.NDArray[np.int32]:
    """Leiden on cuGraph for a COO graph already on the device (int32 / int32 / float32
    device pointers, nnz each). Unchecked beyond sizes."""

def leiden_device(device: str, /) -> dict[str, Any]:
    """Resolve a Leiden device string.

    Returns {"requested", "resolved" ("cpu" or "cuda:<index>"), "cugraph_compiled": bool}.
    "auto" resolves to "cuda:0" only when the cugraph feature is built, libcugraph_c.so
    loads and a CUDA device is usable; an unusable explicit "cuda" raises RuntimeError.
    """

def inflate_chunks(
    path: str,
    offsets: npt.NDArray[np.uint64],
    sizes: npt.NDArray[np.uint64],
    chunk_len: int,
    n: int,
    src: str,
    to_f32: bool,
    to_i64: bool = False,
    /,
) -> npt.NDArray[np.float32] | npt.NDArray[np.int32] | npt.NDArray[np.int64]:
    """Inflate the zlib chunks of a 1-D HDF5 dataset, in element order, to float32, int64 or int32."""

def vst_from_sums(
    sum: npt.NDArray[np.float64],
    sumsq: npt.NDArray[np.float64],
    max: npt.NDArray[np.float64],
    nnz: npt.NDArray[np.uint64],
    n_rows: int,
    span: float,
    clip: float,
    /,
) -> dict[str, Any]:
    """vst statistics from per-gene sums of the raw counts: {"mean", "variance",
    "variance_expected", "variance_standardized", "sd": float64, "needs_clip": bool}."""

def preprocess_csr(
    n_rows: int,
    n_cols: int,
    indptr: npt.NDArray[np.int32] | npt.NDArray[np.int64],
    indices: npt.NDArray[np.int32] | npt.NDArray[np.int64],
    data: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    n_hvg: int,
    scale_factor: float,
    span: float,
    clip: float | None,
    /,
) -> dict[str, Any]:
    """Fused normalisation + vst HVG statistics of a CSR counts matrix (cells x genes).

    The arrays are borrowed as in pca_csr. Returns {"data": float32 (nnz,) log-normalised
    values in input order, "library_size": float64 (n_rows,), "mean", "variance",
    "variance_expected", "variance_standardized": float64 (n_cols,), "hvg": int32 (n_hvg,)
    gene indices most variable first, "hvg_mean", "hvg_sd": float64 (n_hvg,), "clip": float, "hvg_indptr": int64,
    "hvg_indices": int32, "hvg_data": float32: the HVG columns as a canonical CSR in HVG order}.
    """

def device_info() -> dict[str, Any]:
    """CUDA state of this build: cuda_built, cuda_available, device_count, device_name,
    free_gb, total_gb, compute_capability, cuda_error, auto_device. Never raises."""

def device(device: str, /) -> str:
    """Resolve a PCA / kNN device string to "cpu" or "cuda:<index>". ValueError for an
    unknown string, RuntimeError("GPU: ...") for an unusable CUDA request."""

def _probe_detach(seconds: float, /) -> float:
    """Busy-wait for `seconds` with the GIL released, through the kernels' own wrapper.

    Returns the measured duration. Test helper for GIL release.
    """

def _probe_error(kind: str, message: str, /) -> None:
    """Raise what a scfast_core::Error of `kind` ("invalid_input", "invalid_parameter",
    "backend") maps to. Test helper for error mapping."""

def _probe_panic(kind: str, /) -> None:
    """Panic inside the kernel wrapper: "panic", "unimplemented" (like a stub kernel) or
    "rayon" (on a rayon worker thread). Test helper: must raise, never abort."""

def harmony_dense(
    emb: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    batches: npt.NDArray[np.uint32],
    n_batches: int,
    theta: float,
    lambda_: float | None,
    sigma: float,
    nclust: int | None,
    max_iter: int,
    max_iter_cluster: int,
    epsilon_cluster: float,
    epsilon_harmony: float,
    block_size: float,
    tau: float,
    alpha: float,
    batch_prop_cutoff: float,
    kmeans_nstart: int,
    kmeans_iter: int,
    seed: int,
    device: str = "cpu",
    init_centroids: npt.NDArray[np.float32] | None = None,
) -> tuple[npt.NDArray[np.float32] | npt.NDArray[np.float64], dict[str, Any]]:
    """Harmony (harmony 2.x algorithm) on a C-contiguous (n, d) float32 / float64 embedding
    with 0-based batch codes below n_batches (every batch non-empty).

    lambda_ None is the automatic ridge penalty (alpha E); nclust None is
    min(round(n / 30), 100); epsilon_harmony -inf disables early stopping. device "cuda"
    raises RuntimeError("GPU: ..."): there is no CUDA backend yet.

    Returns (corrected embedding (n, d) in the input dtype, {"objective", "objective_kmeans",
    "kmeans_rounds", "iters", "converged", "nclust", "device"}).
    """

def umap(
    indices: npt.NDArray[np.int32],
    distances: npt.NDArray[np.float32],
    init: npt.NDArray[np.float32],
    a: float,
    b: float,
    n_epochs: int,
    negative_sample_rate: int,
    learning_rate: float,
    seed: int,
) -> npt.NDArray[np.float32]: ...
