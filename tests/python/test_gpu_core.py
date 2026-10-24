import json
import subprocess
import sys
import types
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp

from cellwarden import gpu_core
from cellwarden.exchange import counts_checksum, read_exchange, validate_exchange, write_exchange

REPO = Path(__file__).resolve().parents[2]


def _synthetic(n_cells=600, n_genes=400, n_groups=3, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), n_cells // n_groups)
    base = rng.gamma(2.0, 1.0, size=n_genes)
    mu = np.tile(base, (n_groups, 1))
    for g in range(n_groups):
        markers = rng.choice(n_genes, 40, replace=False)
        mu[g, markers] *= 8.0
    lam = mu[groups] * rng.gamma(10.0, 0.1, size=(n_cells, n_genes))  # NB = gamma-Poisson
    counts = sp.csr_matrix(rng.poisson(lam).astype(np.int32))
    obs = pd.DataFrame(
        {
            "truth": pd.Categorical([f"g{g}" for g in groups]),
            "batch": pd.Categorical(np.where(np.arange(n_cells) % 2 == 0, "b1", "b2")),
        },
        index=[f"CELL_{i:05d}-1" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"GENE_{i}" if i % 5 else f"MT_like_{i}" for i in range(n_genes)])
    a = ad.AnnData(X=counts.astype(np.float32), obs=obs, var=var)
    a.layers["counts"] = counts
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=200)
    return a


@pytest.fixture(scope="module")
def slim(tmp_path_factory):
    a = _synthetic()
    p = tmp_path_factory.mktemp("gc") / "slim.h5ad"
    write_exchange(a, p, "slim", run_id="test-run", stage="S1_qc", config_hash="cfg")
    return p, counts_checksum(a.layers["counts"]), a


def test_run_cpu(slim, tmp_path):
    p, checksum, a = slim
    out = tmp_path / "results.h5ad"
    summary = gpu_core.run(p, out, n_pcs=30, device="cpu", seed=0)
    assert summary["device"] == "cpu"
    assert summary["n_cells"] == 600
    assert summary["n_clusters"] >= 2
    assert summary["n_pcs"] == 30
    for step in ("read", "scale", "pca", "neighbors", "leiden", "umap", "write"):
        assert step in summary["timings_s"]
    assert summary["peak_host_rss_bytes"] > 0
    json.dumps(summary)

    assert validate_exchange(out) == []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = ad.read_h5ad(out)
    assert r.shape == (600, 0)
    assert list(r.obs_names) == list(a.obs_names)
    assert r.obsm["X_pca"].shape == (600, 30) and r.obsm["X_pca"].dtype == np.float32
    assert r.obsm["X_umap"].shape == (600, 2)
    assert "X_harmony" not in r.obsm
    assert r.obsp["connectivities"].shape == (600, 600)
    assert r.obsp["distances"].shape == (600, 600)
    assert isinstance(r.obs["leiden"].dtype, pd.CategoricalDtype)
    ex = r.uns["exchange"]
    assert ex["mode"] == "results"
    assert ex["cluster_key"] == "leiden"
    assert ex["source_counts_checksum"] == checksum
    assert ex["run_id"] == "test-run" and ex["stage"] == "S2_gpu_core" and ex["config_hash"] == "cfg"
    # clusters recover the planted groups
    ct = pd.crosstab(r.obs["leiden"], a.obs["truth"])
    purity = ct.max(axis=1).sum() / ct.to_numpy().sum()
    assert purity > 0.9


def test_run_is_deterministic(slim, tmp_path):
    p, _, _ = slim
    gpu_core.run(p, tmp_path / "a.h5ad", device="cpu", seed=1)
    gpu_core.run(p, tmp_path / "b.h5ad", device="cpu", seed=1)
    a, b = read_exchange(tmp_path / "a.h5ad"), read_exchange(tmp_path / "b.h5ad")
    assert list(a.obs["leiden"]) == list(b.obs["leiden"])
    np.testing.assert_allclose(a.obsm["X_umap"], b.obsm["X_umap"])


def test_n_pcs_capped(tmp_path):
    a = _synthetic(n_cells=60, n_genes=40)
    a.var["highly_variable"] = False
    a.var.iloc[:12, a.var.columns.get_loc("highly_variable")] = True
    p = tmp_path / "s.h5ad"
    write_exchange(a, p, "slim")
    s = gpu_core.run(p, tmp_path / "r.h5ad", n_pcs=50, n_neighbors=10, device="cpu")
    assert s["n_pcs"] == 11
    assert read_exchange(tmp_path / "r.h5ad").obsm["X_pca"].shape == (60, 11)


def test_run_harmony_cpu(slim, tmp_path):
    pytest.importorskip("harmonypy")
    p, checksum, _ = slim
    s = gpu_core.run(p, tmp_path / "h.h5ad", batch_key="batch", device="cpu")
    assert s["harmony"] and "harmony" in s["timings_s"]
    r = read_exchange(tmp_path / "h.h5ad")
    assert r.obsm["X_harmony"].shape == r.obsm["X_pca"].shape
    assert r.uns["exchange"]["source_counts_checksum"] == checksum


def test_bad_batch_key(slim, tmp_path):
    with pytest.raises(ValueError, match="batch_key"):
        gpu_core.run(slim[0], tmp_path / "x.h5ad", batch_key="nope", device="cpu")


def test_rejects_non_slim(tmp_path):
    a = _synthetic(n_cells=60, n_genes=40)
    p = tmp_path / "full.h5ad"
    write_exchange(a, p, "full")
    with pytest.raises(Exception, match="slim"):
        gpu_core.run(p, tmp_path / "r.h5ad", device="cpu")


def test_device_gpu_raises_without_gpu(slim, tmp_path):
    ok, _ = gpu_core.gpu_available()
    if ok:
        pytest.skip("a GPU is available here")
    with pytest.raises(gpu_core.DeviceError, match="gpu"):
        gpu_core.run(slim[0], tmp_path / "g.h5ad", device="gpu")
    assert not (tmp_path / "g.h5ad").exists()


def test_auto_falls_back_to_cpu(slim, tmp_path):
    ok, _ = gpu_core.gpu_available()
    s = gpu_core.run(slim[0], tmp_path / "auto.h5ad", device="auto")
    assert s["device"] == ("gpu" if ok else "cpu")


def test_bad_device(slim, tmp_path):
    with pytest.raises(ValueError):
        gpu_core.run(slim[0], tmp_path / "x.h5ad", device="tpu")


def test_cli(slim, tmp_path):
    out = tmp_path / "cli.h5ad"
    env_path = str(REPO / "python")
    proc = subprocess.run(
        [sys.executable, "-m", "cellwarden.gpu_core", "--in", str(slim[0]), "--out", str(out),
         "--n-pcs", "20", "--resolution", "0.3", "--device", "cpu", "--seed", "0"],
        capture_output=True, text=True, env={"PYTHONPATH": env_path, "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["device"] == "cpu" and summary["n_pcs"] == 20
    assert validate_exchange(out) == []
    if not gpu_core.gpu_available()[0]:
        proc = subprocess.run(
            [sys.executable, "-m", "cellwarden.gpu_core", "--in", str(slim[0]), "--out", str(out), "--device", "gpu"],
            capture_output=True, text=True, env={"PYTHONPATH": env_path, "PATH": "/usr/bin:/bin"},
        )
        assert proc.returncode == 2
        assert "DeviceError" in proc.stderr


# --------------------------------------------------------------------------
# --engine
# --------------------------------------------------------------------------


def test_engine_default_unchanged(slim, tmp_path):
    s = gpu_core.run(slim[0], tmp_path / "d.h5ad", device="cpu")
    assert s["engine"] == "scanpy"
    s = gpu_core.run(slim[0], tmp_path / "e.h5ad", device="cpu", engine="scanpy")
    assert s["engine"] == "scanpy" and s["device"] == "cpu"


def test_engine_invalid_and_rapids_without_gpu(slim, tmp_path):
    with pytest.raises(ValueError, match="engine"):
        gpu_core.run(slim[0], tmp_path / "x.h5ad", device="cpu", engine="nope")
    if gpu_core.gpu_available()[0]:
        pytest.skip("a GPU is available here")
    with pytest.raises(gpu_core.DeviceError):
        gpu_core.run(slim[0], tmp_path / "x.h5ad", device="cpu", engine="rapids")


def test_engine_rust_missing_scfast_gives_clear_error(slim, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "scfast", None)  # makes `import scfast` raise ImportError
    with pytest.raises(RuntimeError, match="maturin develop"):
        gpu_core.run(slim[0], tmp_path / "r.h5ad", engine="rust")
    assert not (tmp_path / "r.h5ad").exists()
    # and through the CLI: exit code 2 with a JSON error
    assert gpu_core.main(["--in", str(slim[0]), "--out", str(tmp_path / "r.h5ad"), "--engine", "rust"]) == 2


def test_engine_rust_rejects_gpu_device(slim, tmp_path):
    pytest.importorskip("scfast")
    with pytest.raises(gpu_core.DeviceError, match="no GPU backend"):
        gpu_core.run(slim[0], tmp_path / "r.h5ad", device="gpu", engine="rust")


def test_engine_rust_run(slim, tmp_path):
    pytest.importorskip("scfast")
    p, checksum, a = slim
    out = tmp_path / "rust.h5ad"
    s = gpu_core.run(p, out, n_pcs=30, device="auto", seed=0, engine="rust")
    assert s["engine"] == "rust" and s["device"] == "cpu" and s["n_clusters"] >= 2
    assert validate_exchange(out) == []
    r = read_exchange(out)
    assert r.obsm["X_pca"].shape == (600, 30) and r.obsm["X_umap"].shape == (600, 2)
    assert r.obsp["connectivities"].shape == (600, 600) and r.obsp["distances"].shape == (600, 600)
    assert r.uns["exchange"]["source_counts_checksum"] == checksum
    assert r.uns["gpu_core"]["engine"] == "rust"
    ct = pd.crosstab(r.obs["leiden"], a.obs["truth"])
    assert ct.max(axis=1).sum() / ct.to_numpy().sum() > 0.9
    s2 = gpu_core.run(p, tmp_path / "rust2.h5ad", seed=0, engine="rust")
    assert list(read_exchange(tmp_path / "rust2.h5ad").obs["leiden"]) == list(r.obs["leiden"])
    assert s2["n_clusters"] == s["n_clusters"]


def test_engine_rust_harmony(slim, tmp_path):
    pytest.importorskip("scfast")
    pytest.importorskip("harmonypy")
    s = gpu_core.run(slim[0], tmp_path / "h.h5ad", batch_key="batch", engine="rust")
    assert s["harmony"] and "harmony" in s["timings_s"]
    assert read_exchange(tmp_path / "h.h5ad").obsm["X_harmony"].shape == (600, 30)


# --engine rust on the GPU: scfast CUDA when scfast.device_info() reports it (docs/scfast-integration.md)


class _FakeScfast:
    """A stand-in scfast module: records the device each call received."""

    def __init__(self, info=None, raise_info=None):
        self._info = info
        self._raise = raise_info
        self.calls = []

    def device_info(self):
        if self._raise is not None:
            raise self._raise
        return self._info


def test_scfast_cuda_available_reads_device_info():
    class NoInfo:  # a CPU-only build has no device_info at all
        pass

    assert gpu_core.scfast_cuda_available(NoInfo()) == (False, "this build of scfast has no device_info() (CPU-only build)")
    ok, why = gpu_core.scfast_cuda_available(_FakeScfast({}))
    assert not ok and "without the cuda feature" in why
    ok, why = gpu_core.scfast_cuda_available(_FakeScfast({"cuda_built": True}))
    assert not ok and "no CUDA device is visible" in why
    ok, why = gpu_core.scfast_cuda_available(_FakeScfast({"cuda_built": True, "cuda_available": False}))
    assert not ok
    ok, why = gpu_core.scfast_cuda_available(_FakeScfast({"cuda_built": True, "cuda_available": True, "device_name": "L4"}))
    assert ok and why == "L4"
    ok, why = gpu_core.scfast_cuda_available(_FakeScfast(raise_info=RuntimeError("libcuda missing")))
    assert not ok and "libcuda missing" in why


def test_engine_rust_gpu_uses_cuda_when_device_info_says_so(slim, tmp_path, monkeypatch):
    scfast = pytest.importorskip("scfast")
    seen = {}

    def pca(X, n_comps, seed, device="auto", **kwargs):
        seen["pca"] = device
        return scfast.pca(X, n_comps=n_comps, seed=seed)

    def knn(emb, k, seed, device="auto"):
        seen["knn"] = device
        return scfast.knn(emb, k=k, seed=seed)

    fake = types.SimpleNamespace(pca=pca, knn=knn, snn=scfast.snn, leiden=scfast.leiden,
                                 device_info=lambda: {"cuda_built": True, "cuda_available": True})
    monkeypatch.setattr(gpu_core, "_import_scfast", lambda: fake)
    s = gpu_core.run(slim[0], tmp_path / "cuda.h5ad", device="gpu", engine="rust", seed=0)
    assert s["engine"] == "rust" and s["device"] == "gpu" and s["scfast_device"] == "cuda"
    assert seen == {"pca": "cuda", "knn": "cuda"}
    assert read_exchange(tmp_path / "cuda.h5ad").uns["gpu_core"]["scfast_device"] == "cuda"
    # --device auto also picks CUDA when it is there ...
    seen.clear()
    s = gpu_core.run(slim[0], tmp_path / "cuda2.h5ad", device="auto", engine="rust", seed=0)
    assert s["device"] == "gpu" and seen == {"pca": "cuda", "knn": "cuda"}
    # ... and stays on the CPU without it, while --device gpu is a clear error
    fake.device_info = lambda: {"cuda_built": True, "cuda_available": False}
    seen.clear()
    s = gpu_core.run(slim[0], tmp_path / "cpu.h5ad", device="auto", engine="rust", seed=0)
    assert s["device"] == "cpu" and s["scfast_device"] == "cpu" and seen == {"pca": "cpu", "knn": "cpu"}
    with pytest.raises(gpu_core.DeviceError, match="no CUDA device is visible"):
        gpu_core.run(slim[0], tmp_path / "err.h5ad", device="gpu", engine="rust")


def test_rapids_leiden_keeps_cugraph_level_default(slim, tmp_path, monkeypatch):
    seen = {}

    def leiden(adata, **kw):
        seen.update(kw)
        sc.tl.leiden(adata, resolution=kw["resolution"], random_state=0, flavor="igraph", n_iterations=2,
                     directed=False, key_added=kw["key_added"])

    rsc = types.SimpleNamespace(
        get=types.SimpleNamespace(anndata_to_GPU=lambda a: None, anndata_to_CPU=lambda a, convert_all=True: None),
        pp=types.SimpleNamespace(scale=lambda a, max_value: sc.pp.scale(a, max_value=max_value),
                                 pca=lambda a, n_comps, random_state: sc.pp.pca(a, n_comps=n_comps, mask_var=None),
                                 neighbors=lambda a, **kw: sc.pp.neighbors(a, n_neighbors=kw["n_neighbors"],
                                                                           use_rep=kw["use_rep"])),
        tl=types.SimpleNamespace(leiden=leiden, umap=lambda a, random_state: sc.tl.umap(a)),
    )
    pool = types.SimpleNamespace(total_bytes=lambda: 0)
    monkeypatch.setitem(sys.modules, "rapids_singlecell", rsc)
    monkeypatch.setitem(sys.modules, "cupy", types.SimpleNamespace(get_default_memory_pool=lambda: pool,
                                                               ndarray=type("ndarray", (), {})))
    monkeypatch.setattr(gpu_core, "gpu_available", lambda *a, **k: (True, "fake GPU"))
    s = gpu_core.run(slim[0], tmp_path / "rsc.h5ad", device="gpu", engine="rapids", seed=0)
    assert s["device"] == "gpu"
    assert "n_iterations" not in seen
