"""S1b: doublet scores and a quick pre-clustering (docs/stages-spec.md section 3.3).

    python -m cellwarden.stages.s1b_doublets --run-dir DIR

Inputs: ``S1/metrics_counts.h5ad`` (exchange, full mode) and the partial
``S1/qc_metrics.tsv.gz`` written by S1a. Output: the complete
``S1/qc_metrics.tsv.gz`` (same rows in the same order, S1a's columns
unchanged byte for byte, plus ``doublet_score``, ``doublet_call`` and
``pre_cluster``). Rows are matched to the exchange file by ``cell_id``
(barcode), never by position.

Doublets (``doublets.method``):

* ``scrublet``: per sample, on raw counts, ``random_state = seed``. GPU:
  ``rapids_singlecell.pp.scrublet`` when ``device`` is ``auto``/``gpu`` and
  a GPU is available; otherwise ``scanpy.pp.scrublet``. Samples with fewer
  than 100 cells are skipped (score NA, call FALSE, warning), as are
  samples where Scrublet fails (warning). The device follows the device plan
  (``cellwarden.resources.choose_plan`` on all cells x pre-clustering HVGs and
  ``resources.gpu_mem_cap_gb``): ``auto`` uses the GPU only when it fits.
* ``scdblfinder``: runs in R. If S1a already wrote ``doublet_score`` and
  ``doublet_call``, they are kept; otherwise S1b fails with a message
  (the R implementation is not part of build step 2).
* ``none``: ``doublet_score`` NA, ``doublet_call`` FALSE.

Engine (top-level ``engine``, default ``auto``): when it is not ``seurat`` and
S1a (the scfast path, ``R/stages/doublets_scfast.R``) already wrote
``doublet_score``, ``doublet_call`` and ``pre_cluster``, S1b is a pass-through
(with ``auto`` only if ``metrics_counts.h5ad`` is absent, as it is on that path):
it only checks the table, logs the method, and leaves ``qc_metrics.tsv.gz``
untouched (no ``metrics_counts.h5ad`` exists on that path). With
``engine: seurat``, or when the columns are missing (S1a resolved ``auto`` to
Seurat), the behaviour below applies.

Pre-clustering: ``PRECLUSTER_N_HVG`` (2,000, or all genes if fewer)
highly variable genes (scanpy ``seurat`` flavour on the log-normalised X),
then ``cellwarden.gpu_core.run`` on a temporary slim exchange file with 20 PCs,
15 neighbours and Leiden at resolution 0.3 (seed = ``seed``), on the same
device. Labels go to ``pre_cluster`` as strings. ``gpu_core`` also
computes a UMAP, which is discarded. The temporary files are deleted.

Value formats: ``doublet_score`` as ``%.6g``, ``NA`` when missing;
``doublet_call`` as ``TRUE``/``FALSE`` (R logical literals).
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import StageError, log, read_tsv_str, stage_main, write_tsv_gz

STAGE = "S1b"
MIN_CELLS_SCRUBLET = 100
PRECLUSTER_N_HVG = 2000
PRECLUSTER_N_PCS = 20
PRECLUSTER_N_NEIGHBORS = 15
PRECLUSTER_RESOLUTION = 0.3
S1B_COLUMNS = ("doublet_score", "doublet_call", "pre_cluster")


def choose_device(setting: str, n_cells: int, n_hvg: int, cap_gb: float | None) -> tuple[str, str, dict[str, Any]]:
    """(device for gpu_core, device actually used, plan).

    The device plan of spec section 5 (``cellwarden.resources.choose_plan``) is
    applied to the pre-clustering matrix (all cells x HVGs), so ``auto``
    falls back to the CPU when it would not fit the GPU cap; ``gpu`` fails
    without a usable GPU. The used device is ``gpu``, ``cpu`` or
    ``cpu-as-gpu`` (``CELLWARDEN_ALLOW_CPU_AS_GPU=1`` test override).
    """
    from .. import gpu_core, resources

    try:
        info = None if setting == "cpu" else resources.gpu_info(cap_gb)
        plan = resources.choose_plan(n_cells, n_hvg, setting, info, gpu_mem_cap_gb=cap_gb)
        core = plan["device"]
        return core, gpu_core._resolve_device(core), plan
    except (resources.PlanError, gpu_core.DeviceError) as e:
        raise StageError(str(e)) from e


def _scrublet_one(sub, seed: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    if device == "gpu":  # pragma: no cover - needs a GPU; follows the rapids-singlecell docs
        import rapids_singlecell as rsc

        rsc.get.anndata_to_GPU(sub)
        rsc.pp.scrublet(sub, random_state=seed, verbose=False)
        rsc.get.anndata_to_CPU(sub, convert_all=True)
    else:
        import scanpy as sc

        sc.pp.scrublet(sub, random_state=seed, verbose=False)
    score = sub.obs["doublet_score"].to_numpy(dtype=float)
    call = sub.obs["predicted_doublet"].fillna(False).to_numpy(dtype=bool)
    return score, call


def run_scrublet(adata, sample_of_cell: np.ndarray, seed: int, device: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-sample Scrublet on ``layers['counts']``. Returns (score, call, warnings)."""
    import anndata as ad
    import scipy.sparse as sp

    n = adata.n_obs
    score = np.full(n, np.nan)
    call = np.zeros(n, dtype=bool)
    notes: list[str] = []
    counts = sp.csr_matrix(adata.layers["counts"])
    for s in sorted(set(sample_of_cell)):
        idx = np.flatnonzero(sample_of_cell == s)
        if idx.size < MIN_CELLS_SCRUBLET:
            msg = f"sample {s}: {idx.size} cells < {MIN_CELLS_SCRUBLET}; doublet detection skipped (score NA, call FALSE)"
            warnings.warn(msg, stacklevel=2)
            notes.append(msg)
            continue
        sub = ad.AnnData(X=counts[idx].astype(np.float32))
        sub.obs_names = adata.obs_names[idx]
        sub.var_names = adata.var_names
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sc_s, sc_c = _scrublet_one(sub, seed, device)
            for w in caught:
                notes.append(f"sample {s}: scrublet warning: {w.message}")
        except Exception as e:
            msg = f"sample {s}: scrublet failed ({type(e).__name__}: {e}); score NA, call FALSE"
            warnings.warn(msg, stacklevel=2)
            notes.append(msg)
            continue
        score[idx] = sc_s
        call[idx] = sc_c
        log(f"  scrublet {s}: {idx.size} cells, {int(sc_c.sum())} called doublets ({100 * sc_c.mean():.1f}%)")
    return score, call, notes


def pre_cluster(adata, run_dir: Path, cfg: dict[str, Any], device_setting: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Coarse Leiden clusters of all cells via cellwarden.gpu_core. Returns (labels, gpu_core summary)."""
    import anndata as ad
    import scanpy as sc

    from .. import gpu_core
    from ..exchange import read_exchange, write_exchange

    tmp = run_dir / "S1" / ".s1b_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        x = ad.AnnData(X=adata.X.copy())
        x.obs_names = adata.obs_names
        x.var_names = adata.var_names
        n_top = int(min(PRECLUSTER_N_HVG, x.n_vars))
        if n_top < x.n_vars:
            sc.pp.highly_variable_genes(x, n_top_genes=n_top, flavor="seurat")
            hv = x.var["highly_variable"].to_numpy(dtype=bool)
        else:
            hv = np.ones(x.n_vars, dtype=bool)
        x.var = pd.DataFrame({"highly_variable": hv}, index=x.var_names)
        for c in ("gene_symbol_original", "gene_id"):
            if c in adata.var.columns:
                x.var[c] = adata.var[c].to_numpy()
        ex = dict(adata.uns["exchange"])
        slim = tmp / "preclust_slim.h5ad"
        res = tmp / "preclust_results.h5ad"
        write_exchange(x, slim, "slim", run_id=str(cfg.get("run_id", "none")), stage="S1b_precluster",
                       config_hash=str(cfg.get("config_hash", "none")),
                       source_counts_checksum=ex["counts_checksum"])
        summary = gpu_core.run(slim, res, n_pcs=PRECLUSTER_N_PCS, n_neighbors=PRECLUSTER_N_NEIGHBORS,
                               resolution=PRECLUSTER_RESOLUTION, seed=int(cfg.get("seed", 0)), device=device_setting)
        r = read_exchange(res)
        key = r.uns["exchange"]["cluster_key"]
        labels = r.obs[key].astype(str).reindex(adata.obs_names)
        if labels.isna().any():
            raise StageError("pre-clustering results are missing cells")
        return labels.to_numpy(dtype=object), summary
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _fmt_score(x: float) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.6g}"


def run(run_dir: Path, cfg: dict[str, Any]) -> None:
    from ..exchange import read_exchange

    s1 = run_dir / "S1"
    h5 = s1 / "metrics_counts.h5ad"
    tsv = s1 / "qc_metrics.tsv.gz"
    method = (cfg.get("doublets") or {}).get("method", "scrublet")
    engine = cfg.get("engine") or "auto"
    # scfast path: S1a wrote the columns and no h5ad. With engine auto, an h5ad next to the columns means
    # the Seurat path produced them (S1b rerun on its own output), so that case keeps recomputing.
    if tsv.exists() and engine != "seurat" and (engine == "scfast" or not h5.exists()):
        have = pd.read_csv(tsv, sep="\t", nrows=0).columns
        if all(c in have for c in S1B_COLUMNS):
            log(f"engine {engine}: S1a already wrote {', '.join(S1B_COLUMNS)} "
                f"(doublets.method={method}); S1b is a pass-through, {tsv.name} left unchanged")
            return
    for p in (h5, tsv):
        if not p.exists():
            raise StageError(f"input {p} not found (S1a must run first)")
    device_setting = cfg.get("device", "auto")
    seed = int(cfg.get("seed", 0))

    metrics = read_tsv_str(tsv)
    for c in ("cell_id", "sample_id"):
        if c not in metrics.columns:
            raise StageError(f"{tsv} lacks column {c!r}")
    if not metrics["cell_id"].is_unique:
        raise StageError(f"{tsv}: cell_id values are not unique")
    if method == "scdblfinder" and not {"doublet_score", "doublet_call"} <= set(metrics.columns):
        raise StageError(
            "doublets.method is 'scdblfinder', which runs in R (S1a) and is not implemented in this build step; "
            "S1a did not provide doublet_score/doublet_call. Set doublets.method to 'scrublet' or 'none'."
        )
    drop = [c for c in S1B_COLUMNS if c in metrics.columns and not (method == "scdblfinder" and c != "pre_cluster")]
    if drop:
        log(f"input already has {drop}; recomputing them")
        metrics = metrics.drop(columns=drop)

    log(f"reading {h5}")
    adata = read_exchange(h5)
    if adata.uns["exchange"].get("mode") != "full":
        raise StageError(f"{h5} must be an exchange file in full mode")
    cells = pd.Index(adata.obs_names.astype(str))
    tsv_cells = pd.Index(metrics["cell_id"])
    if len(cells) != len(tsv_cells) or not tsv_cells.isin(cells).all():
        missing = tsv_cells[~tsv_cells.isin(cells)][:5].tolist()
        extra = cells[~cells.isin(tsv_cells)][:5].tolist()
        raise StageError(
            f"cells in {tsv.name} ({len(tsv_cells)}) and {h5.name} ({len(cells)}) differ; "
            f"e.g. only in the table: {missing}, only in the h5ad: {extra}"
        )
    sample_by_cell = pd.Series(metrics["sample_id"].to_numpy(), index=tsv_cells).reindex(cells).to_numpy()
    cap = (cfg.get("resources") or {}).get("gpu_mem_cap_gb")
    core_device, device, plan = choose_device(device_setting, adata.n_obs, min(PRECLUSTER_N_HVG, adata.n_vars),
                                              float(cap) if cap is not None else None)
    log(f"{adata.n_obs} cells x {adata.n_vars} genes, {len(set(sample_by_cell))} sample(s); "
        f"doublets.method={method}; device {device_setting} -> {device} ({plan['reason']})")

    if method == "scrublet":
        score, call, notes = run_scrublet(adata, sample_by_cell, seed, device)
        for n in notes:
            log("  " + n)
        sc_s = pd.Series(score, index=cells).reindex(tsv_cells).to_numpy()
        sc_c = pd.Series(call, index=cells).reindex(tsv_cells).to_numpy(dtype=bool)
        metrics["doublet_score"] = [_fmt_score(v) for v in sc_s]
        metrics["doublet_call"] = np.where(sc_c, "TRUE", "FALSE")
    elif method == "none":
        metrics["doublet_score"] = "NA"
        metrics["doublet_call"] = "FALSE"
        log("doublets.method none: doublet detection skipped")
    else:
        log("doublets.method scdblfinder: keeping doublet_score/doublet_call from S1a")

    log("pre-clustering (HVGs, 20 PCs, Leiden resolution 0.3) via cellwarden.gpu_core")
    labels, summary = pre_cluster(adata, run_dir, cfg, core_device)
    metrics["pre_cluster"] = pd.Series(labels, index=cells).reindex(tsv_cells).to_numpy()
    log(f"pre-clustering: {summary['n_clusters']} clusters on {summary['device']} in {summary['total_s']:.1f}s")

    write_tsv_gz(metrics, tsv)
    log(f"wrote {tsv} ({len(metrics)} rows, {metrics.shape[1]} columns)")


def main(argv: list[str] | None = None) -> int:
    return stage_main(STAGE, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
