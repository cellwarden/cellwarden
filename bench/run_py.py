"""Benchmark one Python engine on one prepared dataset (docs/RUST_ENGINE.md section 8).

Engines:
  scanpy      Scanpy on CPU
  scanpy-gpu  Scanpy workflow on GPU (rapids-singlecell / CuPy)
  rapids      alias for scanpy-gpu (same implementation)
  scfast-cpu  scfast Rust engine, CPU backend   (records 'unavailable' until built)
  scfast-gpu  scfast Rust engine, CUDA backend  (needs a wheel built with --features cuda,cugraph)

Steps (same names and parameters as bench/run_seurat.R): read, normalize, hvg,
scale, pca, neighbors, cluster, umap; GPU engines add to_gpu / to_cpu transfers.
Repeat 0 is a warm-up and is flagged; repeats 1..N are measured.

    python bench/run_py.py --engine scanpy --data bench/results/run1/data/n2638 \
        --out bench/results/run1 --reps 3 --threads 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

T_PROCESS = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (stdlib-only import; numpy not loaded yet)

ENGINES = ("scanpy", "scanpy-gpu", "rapids", "scfast-cpu", "scfast-gpu")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--engine", choices=ENGINES, required=True)
    p.add_argument("--data", required=True, help="prepared folder with matrix.h5 (bench/prep.py)")
    p.add_argument("--out", required=True, help="run folder; results.jsonl is appended here")
    p.add_argument("--reps", type=int, default=3, help="measured repeats (plus one warm-up)")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-pcs", type=int, default=30, help="components to compute (Seurat npcs)")
    p.add_argument("--dims", type=int, default=10,
                   help="leading PCs used for neighbours and UMAP (seurat_pbmc3k.R uses dims = 1:10)")
    p.add_argument("--k", type=int, default=20, help="neighbours including the cell itself (Seurat k.param)")
    p.add_argument("--resolution", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-umap", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------- engines

class ScanpyEngine:
    device = "cpu"

    def startup(self):
        import importlib.metadata as md

        import scanpy as sc

        self.sc, self.md = sc, md
        return {"scanpy": md.version("scanpy"), "anndata": md.version("anndata")}

    def read(self, path):
        a = self.sc.read_10x_h5(path)
        a.var_names_make_unique()
        return a

    def normalize(self, a, cfg):
        a.layers["counts"] = a.X.copy()
        self.sc.pp.normalize_total(a, target_sum=1e4)
        self.sc.pp.log1p(a)

    def hvg(self, a, cfg):
        self.sc.pp.highly_variable_genes(a, n_top_genes=cfg.n_hvg, flavor="seurat_v3", layer="counts")

    def scale(self, a, cfg):
        b = a[:, a.var.highly_variable].copy()
        del b.layers["counts"]
        self.sc.pp.scale(b, max_value=10)
        return b

    def pca(self, a, cfg):
        self.sc.pp.pca(a, n_comps=cfg.n_pcs, random_state=cfg.seed)

    def neighbors(self, a, cfg):
        self.sc.pp.neighbors(a, n_neighbors=cfg.k, n_pcs=cfg.dims, random_state=cfg.seed)

    def cluster(self, a, cfg):
        self.sc.tl.leiden(a, resolution=cfg.resolution, random_state=cfg.seed, flavor="igraph",
                          n_iterations=2, directed=False, key_added="cluster")

    def umap(self, a, cfg):
        self.sc.tl.umap(a, random_state=cfg.seed)


class RapidsEngine(ScanpyEngine):
    device = "gpu"

    def startup(self):
        versions = super().startup()
        import cupy as cp
        import rapids_singlecell as rsc
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator

        # A pooling allocator for CuPy and cuGraph: without one cuGraph Leiden at 500k cells
        # spends half its time in cudaMalloc / cudaFree. The driver's stream-ordered pool
        # (cudaMallocAsync), because RMM's fixed pool fragments and runs out at 500k cells.
        rmm.mr.set_current_device_resource(rmm.mr.CudaAsyncMemoryResource())
        cp.cuda.set_allocator(rmm_cupy_allocator)
        cp.zeros(1)  # create the CUDA context now, so it counts as start-up
        cp.cuda.Device().synchronize()
        self.cp, self.rsc = cp, rsc
        versions.update({"rapids_singlecell": rsc.__version__, "cupy": cp.__version__})
        return versions

    def sync(self):
        self.cp.cuda.Device().synchronize()

    def gpu_mem(self):
        free, total = self.cp.cuda.runtime.memGetInfo()
        return int(total - free)  # includes the RMM pool and scfast's own allocations

    def to_gpu(self, a):
        self.rsc.get.anndata_to_GPU(a)

    def to_cpu(self, a):
        self.rsc.get.anndata_to_CPU(a, convert_all=True)

    def normalize(self, a, cfg):
        a.layers["counts"] = a.X.copy()
        self.rsc.pp.normalize_total(a, target_sum=1e4)
        self.rsc.pp.log1p(a)

    def hvg(self, a, cfg):
        self.rsc.pp.highly_variable_genes(a, n_top_genes=cfg.n_hvg, flavor="seurat_v3", layer="counts")

    def scale(self, a, cfg):
        b = a[:, a.var.highly_variable].copy()
        del b.layers["counts"]
        self.rsc.pp.scale(b, max_value=10)
        return b

    def pca(self, a, cfg):
        self.rsc.pp.pca(a, n_comps=cfg.n_pcs, random_state=cfg.seed)

    def neighbors(self, a, cfg):
        self.rsc.pp.neighbors(a, n_neighbors=cfg.k, n_pcs=cfg.dims, random_state=cfg.seed)

    def cluster(self, a, cfg):
        # rsc passes n_iterations to cuGraph as max_level (aggregation depth), not leidenalg's
        # repeat count: 2 stops the merge early, so keep its default of 100.
        self.rsc.tl.leiden(a, resolution=cfg.resolution, random_state=cfg.seed, key_added="cluster")

    def umap(self, a, cfg):
        self.rsc.tl.umap(a, random_state=cfg.seed)


class Unavailable(RuntimeError):
    pass


def _seurat_leiden_seed(seed: int) -> int:
    """leidenbase (Seurat >= 5.1) requires seed >= 1; bench default is 0."""
    return seed if seed >= 1 else 1


def make_engine(name: str):
    if name == "scanpy":
        return ScanpyEngine()
    if name in ("rapids", "scanpy-gpu"):
        return RapidsEngine()
    try:
        import scfast  # noqa: F401
    except ImportError as e:
        raise Unavailable(f"scfast is not built ({e}); see rust/scfast-py") from e
    return ScfastGpuEngine() if name == "scfast-gpu" else ScfastEngine()


class ScfastEngine:
    """scfast end to end, without scanpy or AnnData: the run is a SimpleNamespace with
    AnnData's attribute names.

    'normalize' is scfast.preprocess, one Rust pass that log-normalises, computes Seurat's
    vst statistics (the HVGs are Seurat's, not scanpy's seurat_v3), selects the HVGs and
    builds their CSR, so 'hvg' and 'scale' only hand that CSR on (scfast's PCA scales and
    clips implicitly). kNN is scfast's "auto": exact below 50k cells, the eps = 0.7 k-d tree
    above (about 0.95 of the exact neighbours; scfast.knn). Leiden clusters Seurat's SNN
    graph; UMAP lays out the kNN graph (umap-learn's fuzzy set) from a PCA start. The kNN
    distances go to obsp["distances"] so the parity outputs line up.
    """

    device = "cpu"
    scfast_device = "cpu"
    leiden_device = "cpu"

    def startup(self):
        import scfast

        self.scfast = scfast
        return {"scfast": scfast.__version__}

    def read(self, path):
        X, barcodes, _ = self.scfast.read_10x_h5(path)
        return SimpleNamespace(X=X, obs_names=barcodes, obs={}, obsm={}, obsp={}, varm={}, uns={})

    def normalize(self, a, cfg):
        a.uns["preprocess"] = self.scfast.preprocess(a.X, n_hvg=cfg.n_hvg)

    def hvg(self, a, cfg):
        a.uns["hvg"] = a.uns["preprocess"]["hvg"]

    def scale(self, a, cfg):
        a.X = a.uns.pop("preprocess")["X_hvg"]
        return a

    def pca(self, a, cfg):
        import scipy.sparse as sp

        x = a.X if self.device == "gpu" else sp.csr_matrix(a.X)  # a CuPy matrix stays on the GPU
        emb, load, sdev = self.scfast.pca(x, n_comps=cfg.n_pcs, seed=cfg.seed, device=self.scfast_device)
        a.obsm["X_pca"] = emb
        a.varm["PCs"] = load
        a.uns["pca"] = {"variance": (sdev.astype("float64") ** 2)}

    def neighbors(self, a, cfg):
        import numpy as np
        import scipy.sparse as sp

        idx, dist = self.knn_search(np.ascontiguousarray(a.obsm["X_pca"][:, : cfg.dims]), cfg.k)
        n = idx.shape[0]
        indptr = np.arange(0, n * (cfg.k - 1) + 1, cfg.k - 1)
        a.obsp["distances"] = sp.csr_matrix((dist[:, 1:].ravel(), idx[:, 1:].ravel(), indptr), shape=(n, n))
        self.snn = self.scfast.snn(self.on_device(idx))  # Seurat's SNN (Jaccard), what Leiden clusters on
        self.knn = idx, dist

    def knn_search(self, x, k):
        return self.scfast.knn(x, k=k, method="auto", device=self.scfast_device)

    def on_device(self, x):
        return x

    def cluster(self, a, cfg):
        labels = self.scfast.leiden(self.snn, resolution=cfg.resolution, seed=_seurat_leiden_seed(cfg.seed),
                                    n_iterations=2, device=self.leiden_device)
        a.obs["cluster"] = labels.astype(str)

    def umap(self, a, cfg):
        a.obsm["X_umap"] = self.scfast.umap(*self.knn, a.obsm["X_pca"][:, :2], seed=cfg.seed)


class ScfastGpuEngine(ScfastEngine, RapidsEngine):
    """ScfastEngine on the device.

    scfast.read_10x_h5(arrays=True) reads on the CPU into float32 data, int32 indices and
    int64 indptr at any size; 'to_gpu' uploads those raw counts. scfast.preprocess then runs on the device (the HVGs are
    Seurat's vst ones, as scfast-cpu), PCA (clip exact), kNN and the SNN graph run on
    scfast's CUDA backend (above CUVS_MIN_CELLS the kNN is cuVS IVF-Flat), Leiden on cuGraph there (the backend used is recorded as
    `leiden_device` in the environment file), and UMAP is rapids-singlecell's (cuML) on the
    fuzzy simplicial set built on the GPU from the same kNN (fuzzy_graph_gpu).
    """

    device = "gpu"
    scfast_device = "cuda"
    normalize, hvg, umap = ScfastEngine.normalize, ScfastEngine.hvg, RapidsEngine.umap

    def startup(self):
        versions = {**RapidsEngine.startup(self), **ScfastEngine.startup(self)}
        info = self.scfast.device_info()
        if not info["cuda_available"]:
            raise RuntimeError(f"scfast CUDA backend unavailable: {info['cuda_error']}")
        self.leiden_device = self.scfast.leiden_device("auto")
        versions.update({"scfast_gpu": info["device_name"], "leiden_device": self.leiden_device})
        return versions

    def read(self, path):
        X, barcodes, _ = self.scfast.read_10x_h5(path, arrays=True)
        return SimpleNamespace(X=X, obs_names=barcodes, obs={}, obsm={}, obsp={}, varm={}, uns={})

    def to_gpu(self, a):
        data, indices, indptr, shape = a.X
        # A tuple: cupyx CSR is limited to 2^31 - 1 non-zeros.
        a.X = (self.cp.asarray(data), self.cp.asarray(indices), self.cp.asarray(indptr), shape)

    def scale(self, a, cfg):
        import anndata as ad
        import pandas as pd

        return ad.AnnData(X=ScfastEngine.scale(self, a, cfg).X, obs=pd.DataFrame(index=pd.Index(a.obs_names)))

    def gpu_mem(self):
        free, total = self.cp.cuda.runtime.memGetInfo()
        return int(total - free)  # scfast allocates outside CuPy's pool

    def on_device(self, x):
        return self.cp.asarray(x)

    def knn_search(self, x, k):
        if x.shape[0] <= CUVS_MIN_CELLS:
            return ScfastEngine.knn_search(self, x, k)
        return cuvs_knn(self.cp.asarray(x, dtype=self.cp.float32), k)

    def neighbors(self, a, cfg):
        super().neighbors(a, cfg)
        idx, dist = self.knn
        a.obsp["connectivities"] = fuzzy_graph_gpu(self.cp.asarray(idx), self.cp.asarray(dist))
        a.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                              "params": {"n_neighbors": cfg.k, "method": "umap", "metric": "euclidean"}}


CUVS_MIN_CELLS = 200_000


def cuvs_knn(x, k):
    """Approximate kNN from cuVS IVF-Flat, in scfast.knn's layout: the cell itself in column 0,
    then k - 1 neighbours by increasing Euclidean distance (int32 / float32, on the host)."""
    import cupy as cp
    from cuvs.neighbors import ivf_flat

    index = ivf_flat.build(ivf_flat.IndexParams(n_lists=1024, metric="sqeuclidean"), x)
    d, i = ivf_flat.search(ivf_flat.SearchParams(n_probes=32), index, x, k)
    d, i = cp.asarray(d), cp.asarray(i).astype(cp.int32)
    n = x.shape[0]
    is_self = i == cp.arange(n, dtype=cp.int32)[:, None]
    is_self[~is_self.any(axis=1), -1] = True  # self missed: drop the farthest instead
    order = cp.argsort(is_self.astype(cp.int8), axis=1)[:, : k - 1]
    idx = cp.concatenate([cp.arange(n, dtype=cp.int32)[:, None], cp.take_along_axis(i, order, axis=1)], axis=1)
    dist = cp.concatenate([cp.zeros((n, 1), cp.float32),
                           cp.sqrt(cp.maximum(cp.take_along_axis(d, order, axis=1), 0))], axis=1)
    return cp.asnumpy(idx), cp.asnumpy(dist.astype(cp.float32))


def fuzzy_graph_gpu(idx, dist):
    """umap-learn's fuzzy_simplicial_set (local_connectivity 1, set_op_mix_ratio 1) from CuPy
    kNN arrays with the cell itself in column 0. Matches cuML's to 4e-6 at 500k cells in a
    quarter of the time (0.21 vs 0.79 s on an L4)."""
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import numpy as np

    n, k = idx.shape
    d = dist[:, 1:].astype(cp.float32)
    rho = cp.where(d > 0, d, cp.inf).min(axis=1)
    rho = cp.where(cp.isinf(rho), 0, rho)
    excess = d - rho[:, None]
    # Bisection for sigma: sum_j exp(-max(d_j - rho, 0) / sigma) = log2(k), 64 steps as umap-learn.
    lo, hi, mid = cp.zeros(n, cp.float32), cp.full(n, cp.inf, cp.float32), cp.ones(n, cp.float32)
    for _ in range(64):
        big = cp.where(excess > 0, cp.exp(-excess / mid[:, None]), 1.0).sum(axis=1) > np.log2(k)
        hi = cp.where(big, mid, hi)
        lo = cp.where(big, lo, mid)
        mid = cp.where(big | ~cp.isinf(hi), (lo + hi) / 2, mid * 2)
    sigma = cp.maximum(mid, 1e-3 * cp.where(rho > 0, d.mean(axis=1), d.mean()))
    w = cp.where(excess > 0, cp.exp(-excess / sigma[:, None]), 1.0).astype(cp.float32)
    w = cp.where(idx[:, 1:] == cp.arange(n)[:, None], 0, w)
    p = csp.csr_matrix((w.ravel(), idx[:, 1:].astype(cp.int32).ravel(),
                        cp.arange(0, n * (k - 1) + 1, k - 1, dtype=cp.int32)), shape=(n, n))
    pt = p.T.tocsr()
    g = p + pt - p.multiply(pt)
    g.eliminate_zeros()
    return g


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    cfg = parse(argv)
    common.set_threads(cfg.threads)  # before numpy / scanpy are imported
    data = Path(cfg.data)
    h5 = data / "matrix.h5"
    prep = json.loads((data.parent / "prep.json").read_text()) if (data.parent / "prep.json").exists() else {}
    n_cells = sum(1 for _ in (data / "cells.tsv").open()) - 1
    out = Path(cfg.out)
    base = {"engine": cfg.engine, "cells": n_cells, "genes": prep.get("genes_qc"),
            "dataset": data.name, "threads": cfg.threads, "commit": common.repo_commit()}

    try:
        engine = make_engine(cfg.engine)
    except Unavailable as e:
        common.Recorder(out / "results.jsonl", {**base, "device": "n/a"}).write(
            step="all", kind="startup", status="unavailable", error=str(e), wall_s=None)
        print(f"{cfg.engine}: unavailable: {e}")
        return 0

    rec = common.Recorder(out / "results.jsonl", {**base, "device": engine.device})
    rec.warmup = True
    t0 = time.perf_counter()
    try:
        versions = engine.startup()
    except Exception as e:
        rec.write(step="startup", kind="startup", status="unavailable", error=f"{type(e).__name__}: {e}",
                  wall_s=None)
        print(f"{cfg.engine}: start-up failed: {e}")
        return 0
    rss, _ = common.peak_rss_bytes()
    rec.write(step="startup", kind="startup", status="ok", error=None,
              wall_s=round(time.perf_counter() - t0, 6),
              process_to_ready_s=round(time.perf_counter() - T_PROCESS, 6), peak_rss_gb=round(rss / 1e9, 4))
    if engine.device == "gpu":
        rec.sync, rec.gpu_mem = engine.sync, engine.gpu_mem

    env = common.environment(versions)
    env_path = out / "env" / f"{cfg.engine}_{data.name}.json"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(json.dumps({**env, "config": vars(cfg)}, indent=2))
    rec.base["node"] = env["node"]

    import numpy as np

    for rep in range(cfg.reps + 1):
        rec.rep, rec.warmup = rep, rep == 0
        try:
            with rec.step("read", kind="io"):
                a = engine.read(str(h5))
            if engine.device == "gpu":
                with rec.step("to_gpu", kind="transfer"):
                    engine.to_gpu(a)
            with rec.step("normalize"):
                engine.normalize(a, cfg)
            with rec.step("hvg"):
                engine.hvg(a, cfg)
            with rec.step("scale"):
                a = engine.scale(a, cfg)
            with rec.step("pca"):
                engine.pca(a, cfg)
            with rec.step("neighbors"):
                engine.neighbors(a, cfg)
            with rec.step("cluster"):
                engine.cluster(a, cfg)
            if not cfg.skip_umap:
                with rec.step("umap"):
                    engine.umap(a, cfg)
            if engine.device == "gpu":
                with rec.step("to_cpu", kind="transfer"):
                    engine.to_cpu(a)
        except Exception as e:  # the failed step is already recorded; stop this engine
            print(f"{cfg.engine} rep {rep}: {type(e).__name__}: {e}")
            return 1
        if rep == 1 or (cfg.reps == 0 and rep == 0):  # parity outputs from the first measured repeat
            knn = common.knn_from_distances(a.obsp["distances"], cfg.k - 1)
            common.write_outputs(out / "outputs" / cfg.engine / data.name, a.obs_names,
                                 np.asarray(a.obsm["X_pca"])[:, : cfg.n_pcs], knn,
                                 a.obs["cluster"].astype(str).tolist())
        del a
    print(f"{cfg.engine} {data.name}: done ({cfg.reps} measured repeats)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
