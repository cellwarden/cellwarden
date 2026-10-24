"""Tests for the scfast Python bindings (rust/scfast-py, docs/RUST_ENGINE.md section 5).

Build and install first (see rust/scfast-py/pyproject.toml), e.g.

    maturin build --release -m rust/scfast-py/Cargo.toml -o dist && pip install dist/scfast-*.whl

The module is skipped when scfast is not importable. Functional tests call the real
kernels; while a kernel is still a stub (unimplemented!(), raised as NotImplementedError)
they are reported as xfail with reason "kernel not implemented yet". Input handling, dtype
dispatch, error mapping and GIL release are tested without needing any kernel.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sp = pytest.importorskip("scipy.sparse")
scfast = pytest.importorskip("scfast")
from scfast import _scfast  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
STUB_REASON = "kernel not implemented yet"


@contextlib.contextmanager
def kernel():
    """Run a kernel call; a stub kernel (NotImplementedError) makes the test xfail."""
    try:
        yield
    except NotImplementedError:  # raised for unimplemented!() panics only
        pytest.xfail(STUB_REASON)


# ---------------------------------------------------------------- data and references

def expr_csr(n_cells=400, n_genes=120, n_groups=5, seed=0, dtype=np.float32):
    """Log-normalised gamma-Poisson counts with marker genes per group, plus one all-zero
    gene and one constant gene. Returns (CSR cells x genes, group of each cell)."""
    rng = np.random.default_rng(seed)
    groups = np.sort(rng.integers(0, n_groups, n_cells))
    base = rng.gamma(1.0, 0.6, size=n_genes)
    mu = np.tile(base, (n_groups, 1))
    for g in range(n_groups):
        mu[g, rng.choice(n_genes - 2, 15, replace=False)] *= 8.0
    counts = rng.poisson(mu[groups] * rng.gamma(10.0, 0.1, size=(n_cells, n_genes))).astype(np.float64)
    counts[:, -2] = 0.0  # zero-variance genes: sd is taken as 1
    counts[:, -1] = 3.0
    lib = counts.sum(1, keepdims=True)
    x = np.log1p(counts / lib * 1e4).astype(dtype)
    X = sp.csr_matrix(x)
    X.sum_duplicates()
    return X, groups


def scaled_dense(X, scale=True):
    a = X.toarray().astype(np.float64)
    z = a - a.mean(0)
    if scale:
        sd = a.std(0, ddof=1)  # ScaleData: n - 1 denominator
        sd[sd == 0] = 1.0
        z /= sd
    return z


def blobs(n=600, d=10, centers=6, seed=0, spread=0.3):
    rng = np.random.default_rng(seed)
    c = rng.normal(0, 3, size=(centers, d))
    lab = rng.integers(0, centers, n)
    return (c[lab] + rng.normal(0, spread, size=(n, d))).astype(np.float32), lab


def brute_knn(x, k):
    """Exact kNN from float32 input, distances in f64: self first, then by (distance, index)."""
    a = x.astype(np.float64)
    d2 = np.maximum(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1), 0.0)
    n = a.shape[0]
    idx = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        d2[i, i] = -1.0  # self first
        order = np.lexsort((np.arange(n), d2[i]))
        idx[i] = order[:k]
        d2[i, i] = 0.0
    dist = np.sqrt(np.take_along_axis(d2, idx, 1)).astype(np.float32)
    return idx, dist


def snn_reference(idx, prune, keep_diagonal=True):
    """Seurat ComputeSNN: A A^T on the binary kNN matrix, Jaccard s / (k + (k - s)), prune."""
    n, k = idx.shape
    a = sp.csr_matrix((np.ones(n * k), idx.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))
    s = (a @ a.T).toarray()
    j = s / (k + (k - s))
    j[j < prune] = 0.0
    if not keep_diagonal:
        np.fill_diagonal(j, 0.0)
    return j


def planted_graph(sizes=(150, 120, 100, 80, 60, 40), p_in=0.3, p_out=0.001, seed=0):
    """Symmetric 0/1 adjacency (no diagonal) with planted clusters, largest first."""
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(len(sizes)), sizes)
    n = truth.size
    p = np.where(truth[:, None] == truth[None, :], p_in, p_out)
    upper = np.triu(rng.random((n, n)) < p, 1)
    g = sp.csr_matrix((upper | upper.T).astype(np.float32))
    g.sum_duplicates()
    return g, truth


def ari(a, b):
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    c = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(c, (a, b), 1)

    def pairs(x):
        return (x * (x - 1) / 2).sum()

    tot = a.size * (a.size - 1) / 2
    sa, sb, sab = pairs(c.sum(1)), pairs(c.sum(0)), pairs(c)
    expected = sa * sb / tot
    return (sab - expected) / ((sa + sb) / 2 - expected)


def with_index_dtype(X, dtype):
    """Copy of X whose indptr and indices have `dtype` (scipy's constructor would downcast
    small int64 arrays to int32, so they are set after construction)."""
    Y = X.copy()
    Y.indptr = Y.indptr.astype(dtype)
    Y.indices = Y.indices.astype(dtype)
    return Y


def align_signs(a, ref):
    s = np.sign((a * ref).sum(0))
    s[s == 0] = 1
    return a * s


# ---------------------------------------------------------------- module

def test_version_matches_cargo_workspace():
    assert isinstance(scfast.__version__, str)
    assert scfast.__version__ == _scfast.__version__
    tomllib = pytest.importorskip("tomllib")
    cargo = tomllib.loads((REPO / "rust" / "Cargo.toml").read_text())
    assert scfast.__version__ == cargo["workspace"]["package"]["version"]
    from importlib import metadata

    try:
        assert metadata.version("scfast") == scfast.__version__
    except metadata.PackageNotFoundError:  # imported from a source tree, not installed
        pass


def test_public_api():
    assert set(scfast.__all__) == {"__version__", "device_info", "pca", "knn", "snn", "leiden", "leiden_device", "preprocess",
                                 "read_10x_h5", "umap", "harmony"}
    for name in ("pca", "knn", "snn", "leiden", "read_10x_h5", "umap", "harmony"):
        assert getattr(scfast, name).__doc__


# ---------------------------------------------------------------- error and panic mapping

@pytest.mark.parametrize("kind, exc, text", [
    ("invalid_input", ValueError, "invalid input: boom"),
    ("invalid_parameter", ValueError, "invalid parameter: boom"),
    ("backend", RuntimeError, "boom"),
])
def test_core_errors_map_to_python_exceptions(kind, exc, text):
    with pytest.raises(exc) as info:
        _scfast._probe_error(kind, "boom")
    assert text in str(info.value)
    if exc is RuntimeError:
        assert not isinstance(info.value, ValueError)


@pytest.mark.parametrize("kind", ["panic", "rayon"])
def test_panic_raises_runtime_error_and_module_survives(kind):
    with pytest.raises(RuntimeError, match="Rust panic") as info:
        _scfast._probe_panic(kind)
    assert not isinstance(info.value, NotImplementedError)
    # the process did not abort and the module still works
    with pytest.raises(ValueError):
        _scfast._probe_error("invalid_input", "still alive")


def test_unimplemented_kernel_raises_not_implemented_error():
    with pytest.raises(NotImplementedError, match="kernel not implemented yet"):
        _scfast._probe_panic("unimplemented")
    assert issubclass(NotImplementedError, RuntimeError)


# ---------------------------------------------------------------- GIL release

def run_with_ticker(fn):
    """Calls fn() while a Python thread records timestamps. Returns (result, seconds,
    number of ticks strictly inside the call). A call that holds the GIL gets ~0 ticks."""
    ticks, stop = [], threading.Event()

    def tick():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.0005)

    old = sys.getswitchinterval()
    sys.setswitchinterval(0.0005)
    t = threading.Thread(target=tick, daemon=True)
    t.start()
    try:
        time.sleep(0.02)
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
    finally:
        stop.set()
        t.join()
        sys.setswitchinterval(old)
    inside = sum(1 for x in ticks if t0 + 0.02 < x < t1 - 0.02)
    return out, t1 - t0, inside


def test_gil_harness_detects_a_gil_holding_call():
    # builtin sum over a range runs in C without releasing the GIL: the ticker must stall
    _, secs, inside = run_with_ticker(lambda: sum(range(20_000_000)))
    if secs < 0.15:
        pytest.skip("GIL-holding control call finished too fast to measure")
    assert inside <= 2


def test_gil_released_by_kernel_wrapper():
    _, secs, inside = run_with_ticker(lambda: _scfast._probe_detach(0.4))
    assert secs >= 0.4
    assert inside >= 10, f"only {inside} ticks of another thread during a {secs:.2f}s call"


def _grow_until_slow(make, call, start, target=0.25, max_steps=5):
    size = start
    for _ in range(max_steps):
        arg = make(size)
        out, secs, inside = run_with_ticker(lambda: call(arg))
        if secs >= target:
            return secs, inside
        size *= 2
    pytest.skip("kernel too fast to observe GIL release")


def test_gil_released_during_pca():
    def make(n):
        return sp.random(n, 1000, density=0.1, format="csr", dtype=np.float32, random_state=1)

    with kernel():
        secs, inside = _grow_until_slow(make, lambda X: scfast.pca(X, n_comps=30), 5000)
    assert inside >= 5, f"only {inside} ticks during a {secs:.2f}s pca call"


def test_gil_released_during_knn():
    def make(n):
        return np.random.default_rng(2).normal(size=(n, 30)).astype(np.float32)

    with kernel():
        secs, inside = _grow_until_slow(make, lambda x: scfast.knn(x, k=20, method="exact"), 4000)
    assert inside >= 5, f"only {inside} ticks during a {secs:.2f}s knn call"


# ---------------------------------------------------------------- CSR input: zero copy and dtype dispatch

@pytest.mark.parametrize("index_dtype, index_type", [(np.int32, "i32"), (np.int64, "i64")])
@pytest.mark.parametrize("value_dtype, value_type", [(np.float32, "f32"), (np.float64, "f64")])
def test_csr_zero_copy_dispatch(index_dtype, index_type, value_dtype, value_type):
    X, _ = expr_csr(60, 30, dtype=value_dtype)
    X = with_index_dtype(X, index_dtype)
    assert X.indptr.dtype == index_dtype and X.data.dtype == value_dtype
    info = scfast._inspect_csr(X)
    assert info["index_type"] == index_type and info["value_type"] == value_type
    assert (info["n_rows"], info["n_cols"], info["nnz"]) == (60, 30, X.nnz)
    # the Rust CsrView points at scipy's own buffers
    assert info["indptr_ptr"] == X.indptr.ctypes.data
    assert info["indices_ptr"] == X.indices.ctypes.data
    assert info["data_ptr"] == X.data.ctypes.data


def test_csr_array_is_accepted_zero_copy():
    X, _ = expr_csr(40, 20)
    A = sp.csr_array(X)
    info = scfast._inspect_csr(A)
    assert info["data_ptr"] == A.data.ctypes.data


def test_non_contiguous_data_is_copied_not_rejected():
    X, _ = expr_csr(40, 20)
    buf = np.zeros(2 * X.nnz, dtype=np.float32)
    buf[::2] = X.data
    X.data = buf[::2]  # strided view
    assert not X.data.flags.c_contiguous
    info = scfast._inspect_csr(X)
    assert info["data_ptr"] != X.data.ctypes.data  # copied
    assert info["indices_ptr"] == X.indices.ctypes.data  # still zero copy


def test_mismatched_index_dtypes_and_other_value_dtypes_are_converted():
    X, _ = expr_csr(40, 20)
    X.indptr = X.indptr.astype(np.int64)  # indices stay int32: only indptr is cast
    info = scfast._inspect_csr(X)
    assert info["index_type"] == "i32"
    assert info["indices_ptr"] == X.indices.ctypes.data
    Y = X.copy()
    Y.data = np.round(Y.data).astype(np.int16)
    assert scfast._inspect_csr(Y)["value_type"] == "f32"


@pytest.mark.parametrize("make", [
    lambda X: X.toarray(),
    lambda X: X.tocsc(),
    lambda X: X.tocoo(),
    lambda X: X.toarray().tolist(),
], ids=["dense", "csc", "coo", "list"])
def test_pca_rejects_non_csr(make):
    X, _ = expr_csr(40, 20)
    with pytest.raises(TypeError, match="CSR"):
        scfast.pca(make(X), n_comps=5)


def test_pca_rejects_csc_with_tocsr_hint():
    X, _ = expr_csr(40, 20)
    with pytest.raises(TypeError, match=r"tocsr\(\)"):
        scfast.pca(X.tocsc(), n_comps=5)


@pytest.mark.parametrize("cols", [[3, 1], [1, 1]], ids=["unsorted", "duplicate"])
def test_pca_rejects_non_canonical_csr(cols):
    X = sp.csr_matrix((np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                       np.array(cols + [0, 2]), np.array([0, 2, 3, 4])), shape=(3, 5))
    with pytest.raises(ValueError, match="canonical"):
        scfast.pca(X, n_comps=1)
    X.sum_duplicates()
    assert scfast._inspect_csr(X)["nnz"] == X.nnz


def test_pca_rejects_malformed_csr_structure():
    X, _ = expr_csr(40, 20)
    Y = X.copy()
    Y.indices[5] = 20  # column out of range: caught by CsrView::new -> ValueError
    with pytest.raises(ValueError, match="out of range"):
        scfast.pca(Y, n_comps=5)
    Z = X.copy()
    Z.indptr = Z.indptr[:-1]
    with pytest.raises(ValueError, match="indptr"):
        scfast.pca(Z, n_comps=5)


@pytest.mark.parametrize("kwargs, exc", [
    ({"n_comps": 0}, ValueError),
    ({"n_comps": 25}, ValueError),  # > min(cells, genes) = 20
    ({"n_comps": 2.5}, TypeError),
    ({"n_comps": True}, TypeError),
    ({"seed": -1}, ValueError),
    ({"oversample": -1}, ValueError),
    ({"device": "tpu"}, ValueError),
    ({"device": "cuda"}, RuntimeError),
])
def test_pca_parameter_errors(kwargs, exc):
    X, _ = expr_csr(40, 20)
    with pytest.raises(exc):
        scfast.pca(X, **{"n_comps": 5, **kwargs})


def test_pca_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        scfast.pca(sp.csr_matrix((0, 10), dtype=np.float32), n_comps=1)


# ---------------------------------------------------------------- PCA (functional)

def test_pca_matches_numpy_svd_of_scaled_matrix():
    X, _ = expr_csr()
    n, g = X.shape
    k = 10
    with kernel():
        # oversample large enough that the sketch spans all genes: exact up to rounding
        emb, load, sdev = scfast.pca(X, n_comps=k, seed=0, oversample=g, power_iters=2)
    assert emb.dtype == load.dtype == sdev.dtype == np.float32
    assert emb.shape == (n, k) and load.shape == (g, k) and sdev.shape == (k,)
    z = scaled_dense(X)
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    np.testing.assert_allclose(sdev * np.sqrt(n - 1), s[:k], rtol=1e-4)
    # orthonormal loadings, embeddings are the projection of the scaled matrix
    np.testing.assert_allclose(load.T @ load, np.eye(k), atol=1e-4)
    np.testing.assert_allclose(emb, z @ load, atol=1e-3 * s[0])
    # the well-separated group components match the SVD vectors up to sign
    top = 4
    np.testing.assert_allclose(align_signs(load[:, :top], vt[:top].T), vt[:top].T, atol=2e-3)
    np.testing.assert_allclose(align_signs(emb[:, :top], u[:, :top] * s[:top]), u[:, :top] * s[:top],
                               atol=2e-3 * s[0])


def test_pca_default_settings_recover_leading_components():
    X, _ = expr_csr()
    with kernel():
        _, load, sdev = scfast.pca(X, n_comps=10, seed=0)
    s = np.linalg.svd(scaled_dense(X), compute_uv=False)
    np.testing.assert_allclose(sdev[:4] * np.sqrt(X.shape[0] - 1), s[:4], rtol=1e-3)
    assert np.all(np.diff(sdev) <= 0)


def test_pca_unscaled_matches_centred_svd():
    X, _ = expr_csr()
    with kernel():
        _, _, sdev = scfast.pca(X, n_comps=6, scale=False, oversample=X.shape[1])
    s = np.linalg.svd(scaled_dense(X, scale=False), compute_uv=False)
    np.testing.assert_allclose(sdev * np.sqrt(X.shape[0] - 1), s[:6], rtol=1e-4)


def test_pca_sign_convention_and_determinism():
    X, _ = expr_csr()
    with kernel():
        a = scfast.pca(X, n_comps=8, seed=3)
        b = scfast.pca(X, n_comps=8, seed=3)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    load = a[1]
    # largest-magnitude loading of every component is positive
    assert np.all(load[np.abs(load).argmax(0), np.arange(load.shape[1])] > 0)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("value_dtype", [np.float32, np.float64])
def test_pca_same_result_for_every_input_dtype(index_dtype, value_dtype):
    X, _ = expr_csr(200, 60)  # float32 values: exactly representable in float64
    Y = with_index_dtype(X.astype(value_dtype), index_dtype)
    with kernel():
        ref = scfast.pca(X, n_comps=5, seed=0)
        out = scfast.pca(Y, n_comps=5, seed=0)
    for r, o in zip(ref, out):
        np.testing.assert_allclose(o, r, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------- kNN

@pytest.mark.parametrize("make, exc", [
    (lambda: np.zeros(10, dtype=np.float32), ValueError),
    (lambda: np.zeros((4, 3, 2), dtype=np.float32), ValueError),
    (lambda: np.zeros((0, 3), dtype=np.float32), ValueError),
    (lambda: sp.random(10, 3, density=0.5, format="csr"), TypeError),
    (lambda: np.zeros((10, 3), dtype=np.complex64), TypeError),
    (lambda: np.array([["a", "b"]] * 3), TypeError),
], ids=["1d", "3d", "empty", "sparse", "complex", "strings"])
def test_knn_rejects_bad_input(make, exc):
    with pytest.raises(exc):
        scfast.knn(make(), k=2)


@pytest.mark.parametrize("kwargs, exc", [
    ({"k": 0}, ValueError),
    ({"k": 1}, ValueError),
    ({"k": 11}, ValueError),  # more than the 10 cells
    ({"k": 2.0}, TypeError),
    ({"method": "annoy"}, ValueError),
    ({"seed": -3}, ValueError),
    ({"m": 1}, ValueError),
    ({"device": "cuda"}, RuntimeError),
])
def test_knn_parameter_errors(kwargs, exc):
    x = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    with pytest.raises(exc):
        scfast.knn(x, **{"k": 3, **kwargs})


# ---------------------------------------------------------------- device (Phase 6)

def test_device_info_shape():
    info = scfast.device_info()
    assert set(info) == {"cuda_built", "cuda_available", "device_count", "device_name", "free_gb",
                         "total_gb", "compute_capability", "cuda_error", "auto_device"}
    assert isinstance(info["cuda_built"], bool) and isinstance(info["cuda_available"], bool)
    if not info["cuda_built"]:
        assert not info["cuda_available"] and info["device_count"] == 0
        assert "cuda" in info["cuda_error"]
    if not info["cuda_available"]:
        assert info["auto_device"] == "cpu"
        assert info["device_name"] is None
        assert isinstance(info["cuda_error"], str)
    else:
        assert info["auto_device"] == "cuda:0" and info["device_count"] >= 1


def test_device_strings_resolve():
    assert _scfast.device("cpu") == "cpu"
    assert _scfast.device("CPU") == "cpu"
    assert _scfast.device("auto") == scfast.device_info()["auto_device"]
    with pytest.raises(ValueError, match="device"):
        _scfast.device("tpu")
    if not scfast.device_info()["cuda_available"]:
        with pytest.raises(RuntimeError, match="GPU"):
            _scfast.device("cuda:0")
        X, _ = expr_csr(40, 20)
        with pytest.raises(RuntimeError, match="GPU"):
            scfast.pca(X, n_comps=5, device="cuda")
        with pytest.raises(RuntimeError, match="GPU"):
            scfast.knn(np.zeros((10, 3), np.float32), k=3, device="gpu")


def test_auto_falls_back_to_cpu_on_gpu_error_only():
    from scfast import _with_cpu_fallback

    calls = []

    def call(dev):
        calls.append(dev)
        if dev != "cpu":
            raise RuntimeError("scfast.pca: GPU: out of device memory (alloc)")
        return "ok"

    with pytest.warns(RuntimeWarning, match="falling back to the CPU"):
        assert _with_cpu_fallback("auto", "cuda:0", call) == "ok"
    assert calls == ["cuda:0", "cpu"]
    calls.clear()
    with pytest.raises(RuntimeError, match="GPU"):
        _with_cpu_fallback("cuda", "cuda:0", call)
    assert calls == ["cuda:0"]

    def other(dev):
        raise RuntimeError("scfast.pca: faer thin SVD: boom")

    with pytest.raises(RuntimeError, match="faer"):
        _with_cpu_fallback("auto", "cuda:0", other)


def test_cpu_and_auto_agree_here():
    X, _ = expr_csr(60, 30)
    a = scfast.pca(X, n_comps=5, device="cpu")
    b = scfast.pca(X, n_comps=5, device="auto")
    if not scfast.device_info()["cuda_available"]:
        for u, v in zip(a, b):
            assert np.array_equal(u, v)


def test_knn_rejects_non_finite_values():
    x = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    x[4, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        scfast.knn(x, k=3)


def test_knn_exact_matches_brute_force():
    x, _ = blobs(400, 8)
    with kernel():
        idx, dist = scfast.knn(x, k=15, method="exact")
    assert idx.dtype == np.int32 and dist.dtype == np.float32
    assert idx.shape == dist.shape == (400, 15)
    ref_idx, ref_dist = brute_knn(x, 15)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(idx[:, 0], np.arange(400))
    assert np.all(dist[:, 0] == 0) and np.all(np.diff(dist, axis=1) >= 0)


def test_knn_k2_is_self_plus_nearest():
    x, _ = blobs(50, 4)
    with pytest.raises(ValueError):
        scfast.knn(x, k=1)  # k counts the cell itself: at least 2
    with kernel():
        idx, dist = scfast.knn(x, k=2, method="exact")
    ref_idx, ref_dist = brute_knn(x, 2)
    np.testing.assert_array_equal(idx, ref_idx)
    np.testing.assert_allclose(dist, ref_dist, rtol=1e-6, atol=1e-6)


def _recall(idx, ref):
    return np.mean([len(set(a[1:]) & set(b[1:])) / (ref.shape[1] - 1) for a, b in zip(idx, ref)])


def test_knn_hnsw_recall_and_auto_threshold():
    x, _ = blobs(1500, 10, centers=8, spread=1.0)
    ref, _ = brute_knn(x, 20)
    with kernel():
        h_idx, h_dist = scfast.knn(x, k=20, method="hnsw", seed=0)
        a_hnsw, _ = scfast.knn(x, k=20, method="auto", auto_threshold=1000)    # 1500 >= 1000, 10 dims: k-d tree
        a_exact, _ = scfast.knn(x, k=20, method="auto", auto_threshold=10_000)  # exact
    np.testing.assert_array_equal(h_idx[:, 0], np.arange(1500))
    assert np.all(np.diff(h_dist, axis=1) >= 0)
    assert _recall(h_idx, ref) >= 0.9
    assert _recall(a_hnsw, ref) >= 0.9
    np.testing.assert_array_equal(a_exact, ref)


@pytest.mark.parametrize("convert", [
    lambda x: x.astype(np.float64),
    lambda x: np.asfortranarray(x),
    lambda x: np.repeat(x, 2, axis=1)[:, ::2],  # non-contiguous view
    lambda x: x.tolist(),
], ids=["float64", "fortran", "strided", "list"])
def test_knn_converts_dtype_and_layout(convert):
    x, _ = blobs(200, 6)
    with kernel():
        ref = scfast.knn(x, k=10, method="exact")
        out = scfast.knn(convert(x), k=10, method="exact")
    np.testing.assert_array_equal(out[0], ref[0])
    np.testing.assert_allclose(out[1], ref[1], rtol=1e-6)


# ---------------------------------------------------------------- SNN

@pytest.mark.parametrize("make, exc", [
    (lambda: np.arange(10), ValueError),
    (lambda: np.zeros((5, 3), dtype=np.float32), TypeError),
    (lambda: np.zeros((0, 3), dtype=np.int32), ValueError),
    (lambda: np.array([[0, 1], [1, 5], [2, 0]]), ValueError),   # index 5 >= 3 cells
    (lambda: np.array([[0, -1], [1, 0], [2, 0]]), ValueError),  # negative
], ids=["1d", "float", "empty", "out-of-range", "negative"])
def test_snn_rejects_bad_indices(make, exc):
    with pytest.raises(exc):
        scfast.snn(make())


@pytest.mark.parametrize("prune", [-0.1, 1.5, float("nan")])
def test_snn_rejects_bad_prune(prune):
    with pytest.raises(ValueError):
        scfast.snn(np.array([[0, 1], [1, 0]]), prune=prune)


def test_snn_rejects_repeated_neighbour():
    with kernel(), pytest.raises(ValueError):
        scfast.snn(np.array([[0, 1, 1], [1, 0, 2], [2, 0, 1]]))


@pytest.mark.parametrize("keep_diagonal", [True, False])
@pytest.mark.parametrize("prune", [1 / 15, 0.0, 0.3])
def test_snn_jaccard_weights_match_seurat_definition(prune, keep_diagonal):
    x, _ = blobs(300, 5, centers=4, spread=0.8)
    idx, _ = brute_knn(x, 10)
    with kernel():
        g = scfast.snn(idx, prune=prune, keep_diagonal=keep_diagonal)
    assert isinstance(g, sp.csr_matrix) and g.dtype == np.float32 and g.shape == (300, 300)
    ref = snn_reference(idx, prune, keep_diagonal)
    dense = g.toarray()
    np.testing.assert_array_equal(dense != 0, ref != 0)
    np.testing.assert_allclose(dense, ref.astype(np.float32), rtol=1e-6)
    assert abs(g - g.T).max() == 0  # exactly symmetric
    assert g.has_sorted_indices and np.all(g.data != 0)
    if keep_diagonal:
        np.testing.assert_array_equal(g.diagonal(), 1.0)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64, np.int16])
def test_snn_accepts_integer_dtypes(dtype):
    x, _ = blobs(120, 4)
    idx, _ = brute_knn(x, 8)
    with kernel():
        ref = scfast.snn(idx.astype(np.int64))
        out = scfast.snn(idx.astype(dtype))
    assert (out != ref).nnz == 0


# ---------------------------------------------------------------- Leiden

@pytest.mark.parametrize("make, exc", [
    (lambda g: g.toarray(), TypeError),
    (lambda g: g.tocoo(), TypeError),
    (lambda g: g[:, :-1].tocsr(), ValueError),  # not square
], ids=["dense", "coo", "non-square"])
def test_leiden_rejects_bad_graphs(make, exc):
    g, _ = planted_graph((20, 20))
    with pytest.raises(exc):
        scfast.leiden(make(g))


@pytest.mark.parametrize("kwargs", [{"resolution": -1}, {"resolution": float("nan")},
                                    {"n_iterations": 0}, {"seed": -1}])
def test_leiden_parameter_errors(kwargs):
    g, _ = planted_graph((20, 20))
    with pytest.raises(ValueError):
        scfast.leiden(g, **kwargs)


def test_leiden_rejects_non_canonical_graph():
    g = sp.csr_matrix((np.ones(3, dtype=np.float32), np.array([1, 1, 0]), np.array([0, 2, 3])), shape=(2, 2))
    with pytest.raises(ValueError, match="canonical"):
        scfast.leiden(g)


@pytest.mark.parametrize("tamper", ["asymmetric", "negative", "nan"])
def test_leiden_rejects_invalid_weights(tamper):
    g, _ = planted_graph((20, 20))
    if tamper == "asymmetric":
        g = sp.csr_matrix(sp.triu(g))  # only one direction of each edge
    else:
        g.data[0] = -1.0 if tamper == "negative" else np.nan
    with kernel(), pytest.raises(ValueError):
        scfast.leiden(g)


def test_leiden_recovers_planted_clusters():
    g, truth = planted_graph()
    with kernel():
        labels = scfast.leiden(g, resolution=0.5, seed=0, n_iterations=2)
    assert labels.dtype == np.int32 and labels.shape == truth.shape
    assert ari(labels, truth) > 0.99
    # labels numbered by decreasing cluster size: here exactly the planted numbering
    np.testing.assert_array_equal(labels, truth)


def test_leiden_is_deterministic_and_accepts_other_graph_types():
    g, truth = planted_graph(seed=1)
    with kernel():
        a = scfast.leiden(g, seed=7)
        b = scfast.leiden(g, seed=7)
        c = scfast.leiden(sp.csr_array(g.astype(np.float64)), seed=7)
        d = scfast.leiden(g.astype(np.int32), seed=7)
        e = scfast.leiden((g + sp.identity(g.shape[0], format="csr", dtype=np.float32)).tocsr(), seed=7)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)
    np.testing.assert_array_equal(a, d)
    assert ari(e, truth) > 0.99  # a diagonal (as in Seurat's RNA_snn) is accepted


# ---------------------------------------------------------------- preprocess

@pytest.mark.parametrize("cols", [[3, 1], [1, 1]], ids=["unsorted", "duplicate"])
def test_preprocess_rejects_non_canonical_csr(cols):
    X = sp.csr_matrix((np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                       np.array(cols + [0, 2]), np.array([0, 2, 3, 4])), shape=(3, 5))
    with pytest.raises(ValueError, match="canonical|duplicate"):
        scfast.preprocess(X, n_hvg=2)


def test_preprocess_x_hvg_is_data_columns_in_hvg_order():
    rng = np.random.default_rng(1)
    counts = sp.random(300, 80, density=0.2, format="csr", random_state=rng, dtype=np.float32)
    counts.data = np.ceil(counts.data * 10)
    out = scfast.preprocess(counts, n_hvg=25)
    expected = out["data"][:, out["hvg"]]
    expected.sort_indices()
    x = out["X_hvg"]
    assert x.shape == (300, 25) and x.has_sorted_indices
    np.testing.assert_array_equal(x.indptr, expected.indptr)
    np.testing.assert_array_equal(x.indices, expected.indices)
    np.testing.assert_array_equal(x.data, expected.data)


# ---------------------------------------------------------------- 10x h5

@pytest.mark.parametrize("shuffle, int_dtype", [(False, np.int64), (False, np.int32), (True, np.int64)])
def test_read_10x_h5_matches_counts(tmp_path, monkeypatch, shuffle, int_dtype):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(0)
    counts = sp.random(50, 30, density=0.3, format="csr", random_state=rng, dtype=np.float64)
    counts.data = np.ceil(counts.data * 20)
    path = tmp_path / "m.h5"
    with h5py.File(path, "w") as f:  # genes x cells CSC: the arrays of the cells x genes CSR
        g = f.create_group("matrix")
        kw = {"compression": "gzip", "shuffle": shuffle, "chunks": (7,)}
        g.create_dataset("data", data=counts.data.astype(np.int32), **kw)
        g.create_dataset("indices", data=counts.indices.astype(int_dtype), **kw)
        g.create_dataset("indptr", data=counts.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array([30, 50], dtype=np.int32))
        g.create_dataset("barcodes", data=np.array([f"c{i}" for i in range(50)], dtype="S"))
        g.create_dataset("features/name", data=np.array([f"g{j}" for j in range(30)], dtype="S"))
        g.create_dataset("features/feature_type", data=np.array(["Gene Expression"] * 30, dtype="S"))
    X, barcodes, names = scfast.read_10x_h5(path)
    assert X.dtype == np.float32 and X.indices.dtype == np.int32 and X.shape == (50, 30)
    assert (X != counts).nnz == 0
    assert barcodes[3] == "c3" and names[4] == "g4"
    # Above 2^31 - 1 non-zeros the arrays are int64 (forced here by lowering the limit).
    monkeypatch.setattr(scfast, "_I32_MAX", 10)
    W, _, _ = scfast.read_10x_h5(path)
    assert W.indices.dtype == np.int64 and W.indptr.dtype == np.int64
    assert (W != counts).nnz == 0
    data, indices, indptr, shape = scfast.read_10x_h5(path, arrays=True)[0]
    assert (data.dtype, indices.dtype, indptr.dtype, shape) == (np.float32, np.int32, np.int64, (50, 30))
    assert (sp.csr_matrix((data, indices, indptr), shape=shape) != counts).nnz == 0


# ---------------------------------------------------------------- end to end

def test_pipeline_recovers_groups():
    X, groups = expr_csr(n_cells=600, n_genes=150, n_groups=5, seed=4)
    with kernel():
        pcs, _, _ = scfast.pca(X, n_comps=10, seed=0)
        idx, _ = scfast.knn(pcs, k=20)
        graph = scfast.snn(idx, prune=1 / 15)
        labels = scfast.leiden(graph, resolution=0.5, seed=0, n_iterations=2)
    assert ari(labels, groups) >= 0.9
