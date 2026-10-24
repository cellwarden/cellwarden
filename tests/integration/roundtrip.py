#!/usr/bin/env python3
"""Python half of the Seurat <-> AnnData round-trip check (tests/integration).

Driven by run_roundtrip.sh; every subcommand exits non-zero on the first
failed assertion and prints what went wrong.

    roundtrip.py check-r-files  WORKDIR   # step 2: read R-written full/slim files
    roundtrip.py check-results  WORKDIR   # step 3: check gpu_core output, make refuse cases
    roundtrip.py write-python   WORKDIR   # step 5: Python writes a full file from the fixture

Needs PYTHONPATH to include <repo>/python (for cellwarden).
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import warnings
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "pbmc_like"
EXPECTED = json.loads((FIXTURE / "expected.json").read_text())
RUN_ID = "roundtrip-it"
CONFIG_HASH = "none"

N_PCS = 30          # must agree with run_roundtrip.sh / roundtrip.R

_failures: list[str] = []


# ----------------------------------------------------------------------- helpers
def check(cond, msg: str) -> None:
    if not cond:
        _failures.append(msg)
        print(f"  FAIL  {msg}")
    else:
        print(f"  ok    {msg}")


def finish(step: str) -> None:
    if _failures:
        print(f"{step}: {len(_failures)} check(s) failed", file=sys.stderr)
        sys.exit(1)
    print(f"{step}: all checks passed")


def sha_names(names) -> str:
    return hashlib.sha256("\n".join(map(str, names)).encode("utf-8")).hexdigest()


def indep_checksum(m) -> str:
    """Spec section 6, independent of cellwarden (cells x genes)."""
    m = sp.csr_matrix(m, copy=True)
    m.sort_indices()
    h = hashlib.sha256()
    h.update(struct.pack("<qq", *m.shape))
    h.update(m.indptr.astype("<i8").tobytes())
    h.update(m.indices.astype("<i8").tobytes())
    h.update(m.data.astype("<f8").tobytes())
    return h.hexdigest()


def fixture_tables():
    feats = pd.read_csv(FIXTURE / "features.tsv", sep="\t", header=None,
                        names=["gene_id", "symbol", "feature_type"], dtype=str,
                        keep_default_na=False)
    bcs = [ln.rstrip("\n") for ln in open(FIXTURE / "barcodes.tsv")]
    cells = pd.read_csv(FIXTURE / "cells.tsv", sep="\t", dtype=str, keep_default_na=False)
    return feats, bcs, cells


def read_h5ad_strict(path: Path):
    """anndata.read_h5ad with UserWarning (and anndata's own warnings) as errors."""
    import anndata as ad
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.filterwarnings("error", category=UserWarning)
        adata = ad.read_h5ad(path)
    other = [f"{w.category.__name__}: {w.message}" for w in rec]
    check(not other, f"{path.name}: anndata.read_h5ad raised no other warnings {other or ''}")
    return adata


def h5_attr(f, key, attr):
    v = f[key].attrs.get(attr)
    return v.decode() if isinstance(v, bytes) else v


def check_encodings(path: Path, mode: str) -> None:
    """Spot-check spec section 1 encoding attributes with raw h5py."""
    with h5py.File(path, "r") as f:
        check(h5_attr(f, "/", "encoding-type") == "anndata", f"{path.name}: root encoding-type anndata")
        check(h5_attr(f, "obs", "encoding-type") == "dataframe", f"{path.name}: obs is dataframe")
        check(h5_attr(f, "var", "encoding-type") == "dataframe", f"{path.name}: var is dataframe")
        check(h5_attr(f, "uns/exchange", "encoding-type") == "dict", f"{path.name}: uns/exchange is dict")
        if mode in ("full", "slim"):
            check(h5_attr(f, "X", "encoding-type") == "csr_matrix", f"{path.name}: X is csr_matrix")
            check(f["X/data"].dtype == np.float32, f"{path.name}: X data float32 (got {f['X/data'].dtype})")
        if mode == "full":
            c = f["layers/counts"]
            check(h5_attr(f, "layers/counts", "encoding-type") == "csr_matrix",
                  f"{path.name}: layers/counts is csr_matrix")
            check(c["data"].dtype in (np.int32, np.int64),
                  f"{path.name}: counts stored as int32/int64 (got {c['data'].dtype})")
            check(c["data"].dtype == np.int32 or EXPECTED["max_count"] > 2**31 - 1,
                  f"{path.name}: counts int32 when values fit")


def check_uns(adata, *, mode: str, writer: str, n_genes: int, path: Path) -> dict:
    ex = dict(adata.uns.get("exchange", {}))
    check(bool(ex), f"{path.name}: uns['exchange'] present")
    check(ex.get("schema_version") == "1.0", f"{path.name}: schema_version == '1.0' (got {ex.get('schema_version')!r})")
    check(ex.get("mode") == mode, f"{path.name}: mode == {mode!r} (got {ex.get('mode')!r})")
    check(ex.get("writer") == writer, f"{path.name}: writer == {writer!r} (got {ex.get('writer')!r})")
    check(ex.get("run_id") == RUN_ID, f"{path.name}: run_id == {RUN_ID!r} (got {ex.get('run_id')!r})")
    check(isinstance(ex.get("stage"), str) and ex.get("stage"), f"{path.name}: stage is a non-empty string")
    check(ex.get("config_hash") == CONFIG_HASH, f"{path.name}: config_hash == {CONFIG_HASH!r}")
    try:
        datetime.fromisoformat(str(ex.get("created_utc")).replace("Z", "+00:00"))
        ok = True
    except ValueError:
        ok = False
    check(ok, f"{path.name}: created_utc is ISO 8601 ({ex.get('created_utc')!r})")
    check(int(ex.get("n_cells", -1)) == EXPECTED["n_cells"], f"{path.name}: n_cells == {EXPECTED['n_cells']}")
    check(int(ex.get("n_genes", -1)) == n_genes, f"{path.name}: n_genes == {n_genes} (got {ex.get('n_genes')})")
    check(ex.get("cells_sha256") == EXPECTED["cells_sha256"], f"{path.name}: cells_sha256 == fixture barcodes sha")
    check(ex.get("cells_sha256") == sha_names(adata.obs_names), f"{path.name}: cells_sha256 matches obs_names")
    check(ex.get("genes_sha256") == sha_names(adata.var_names), f"{path.name}: genes_sha256 matches var_names")
    pk = ex.get("packages")
    check(isinstance(pk, dict) and len(pk) > 0 and all(isinstance(v, str) for v in pk.values()),
          f"{path.name}: packages is a non-empty str->str mapping")
    if mode == "full":
        check(ex.get("counts_checksum") == EXPECTED["counts_checksum"],
              f"{path.name}: uns counts_checksum == expected")
    else:
        check(ex.get("source_counts_checksum") == EXPECTED["counts_checksum"],
              f"{path.name}: source_counts_checksum == expected (got {ex.get('source_counts_checksum')!r})")
    if mode == "results":
        check(isinstance(ex.get("cluster_key"), str) and ex["cluster_key"] in adata.obs.columns,
              f"{path.name}: cluster_key {ex.get('cluster_key')!r} names an obs column")
    return ex


def check_obs(adata, bcs, cells, path: Path) -> None:
    check(list(adata.obs_names) == bcs, f"{path.name}: obs_names == fixture barcodes, same order")
    for col in ("sample", "cell_type"):
        lv = EXPECTED["obs_levels"][col]
        ok = col in adata.obs and isinstance(adata.obs[col].dtype, pd.CategoricalDtype)
        check(ok, f"{path.name}: obs['{col}'] is categorical")
        if ok:
            got = list(adata.obs[col].cat.categories)
            check(got == lv, f"{path.name}: obs['{col}'] levels in order {lv} (got {got})")
            check(list(adata.obs[col].astype(str)) == list(cells[col]),
                  f"{path.name}: obs['{col}'] values match fixture")


# --------------------------------------------------------------------- subcommands
def check_r_files(work: Path) -> None:
    from cellwarden import exchange as ex
    feats, bcs, cells = fixture_tables()
    hvgs = [ln.rstrip("\n") for ln in open(work / "hvgs.txt")]
    seurat_names = [s.replace("_", "-") for s in feats["symbol"]]

    for mode in ("full", "slim"):
        path = work / f"r_{mode}.h5ad"
        print(f"[{path.name}]")
        check_encodings(path, mode)
        adata = read_h5ad_strict(path)
        try:
            ad2 = ex.read_exchange(path)
            check(ad2.shape == adata.shape, f"{path.name}: cellwarden.read_exchange (validate=True) ok, same shape")
        except Exception as e:  # noqa: BLE001
            check(False, f"{path.name}: cellwarden.read_exchange raised {type(e).__name__}: {e}")
        probs = ex.validate_exchange(path)
        check(not probs, f"{path.name}: python validate_exchange is empty {probs or ''}")

        n_genes = EXPECTED["n_genes"] if mode == "full" else len(hvgs)
        check(adata.shape == (EXPECTED["n_cells"], n_genes), f"{path.name}: shape {adata.shape}")
        check_uns(adata, mode=mode, writer="R", n_genes=n_genes, path=path)
        check_obs(adata, bcs, cells, path)

        check("gene_symbol_original" in adata.var, f"{path.name}: var['gene_symbol_original'] present")
        check("gene_id" in adata.var, f"{path.name}: var['gene_id'] present")
        if "gene_symbol_original" in adata.var and "gene_id" in adata.var:
            orig = dict(zip(seurat_names, feats["symbol"]))
            gid = dict(zip(seurat_names, feats["gene_id"]))
            check(list(adata.var["gene_symbol_original"]) == [orig.get(g) for g in adata.var_names],
                  f"{path.name}: gene_symbol_original maps each var_name to the original symbol")
            check(list(adata.var["gene_id"]) == [gid.get(g) for g in adata.var_names],
                  f"{path.name}: gene_id carries the Ensembl IDs")
            us = [g for g in EXPECTED["underscore_genes"] if g in set(adata.var["gene_symbol_original"])]
            if mode == "full":
                check(us == EXPECTED["underscore_genes"],
                      f"{path.name}: all {len(EXPECTED['underscore_genes'])} underscore symbols preserved")
            print(f"        underscore symbols present: {len(us)}")

        if mode == "full":
            check(list(adata.var_names) == seurat_names, f"{path.name}: var_names == Seurat feature names in order")
            check(adata.uns["exchange"]["genes_sha256"] == EXPECTED["genes_sha256_seurat"],
                  f"{path.name}: genes_sha256 == sha of Seurat names")
            check("counts" in adata.layers, f"{path.name}: layers['counts'] present")
            if "counts" in adata.layers:
                c = adata.layers["counts"]
                check(ex.counts_checksum(c) == EXPECTED["counts_checksum"],
                      f"{path.name}: cellwarden counts_checksum(layers['counts']) == expected")
                check(indep_checksum(c) == EXPECTED["counts_checksum"],
                      f"{path.name}: independent checksum of layers['counts'] == expected")
                # X is LogNormalize(scale 1e4) of counts
                c = sp.csr_matrix(c, dtype=np.float64)
                lib = np.asarray(c.sum(1)).ravel()
                ref = sp.diags(1e4 / lib) @ c
                ref.data = np.log1p(ref.data)
                diff = abs(sp.csr_matrix(adata.X, dtype=np.float64) - ref).max()
                check(diff < 1e-4, f"{path.name}: X == log1p(counts/total*1e4) (max diff {diff:.2e})")
            check(int(adata.var["highly_variable"].sum()) == len(hvgs) if "highly_variable" in adata.var else False,
                  f"{path.name}: var['highly_variable'] marks {len(hvgs)} genes")
        else:
            check(set(adata.var_names) == set(hvgs), f"{path.name}: var_names == the {len(hvgs)} R VariableFeatures")
            check("counts" not in adata.layers, f"{path.name}: no layers['counts'] in slim")
            check("counts_checksum" not in adata.uns["exchange"], f"{path.name}: no counts_checksum in slim")
            if "highly_variable" in adata.var:
                check(bool(adata.var["highly_variable"].all()), f"{path.name}: highly_variable all True")
            full = read_h5ad_strict(work / "r_full.h5ad")
            sub = full[:, list(adata.var_names)].X
            d = abs(sp.csr_matrix(sub) - sp.csr_matrix(adata.X)).max()
            check(d == 0, f"{path.name}: slim X equals full X restricted to HVGs (max diff {d})")
    finish("check-r-files")


def _rewrite_results(src: Path, dst: Path, *, drop: int | None = None,
                     shuffle_seed: int | None = None, extra_barcode: str | None = None) -> None:
    """Rewrite a results file via cellwarden.write_exchange, dropping cell `drop`
    and/or permuting the cell order and/or adding a cell the object does not
    have (always keeping the original source_counts_checksum)."""
    import anndata as ad
    from cellwarden import exchange as ex
    a = ad.read_h5ad(src)
    meta = dict(a.uns["exchange"])
    idx = np.arange(a.n_obs)
    if drop is not None:
        idx = idx[idx != drop]
    if shuffle_seed is not None:
        idx = np.random.default_rng(shuffle_seed).permutation(idx)
    b = a[idx].copy()
    if extra_barcode is not None:          # append a copy of cell 0 under a new barcode
        e = a[[0]].copy()
        e.obs_names = [extra_barcode]
        b = ad.concat([b, e], merge="same", uns_merge="first", pairwise=True)
        for k in ("connectivities", "distances"):  # concat leaves obsp only if pairwise
            b.obsp[k] = sp.csr_matrix(b.obsp[k])
    b.uns = {k: v for k, v in b.uns.items() if k != "exchange"}
    ex.write_exchange(b, dst, "results", run_id=meta["run_id"], stage=meta["stage"],
                      config_hash=meta["config_hash"],
                      source_counts_checksum=meta["source_counts_checksum"],
                      cluster_key=meta["cluster_key"])


def check_results(work: Path) -> None:
    from cellwarden import exchange as ex
    feats, bcs, cells = fixture_tables()
    path = work / "results.h5ad"
    print(f"[{path.name}]")
    try:
        out = json.loads((work / "gpu_core.stdout").read_text())
        check(isinstance(out, dict), "gpu_core CLI printed a JSON object")
        print(f"        gpu_core JSON: {json.dumps(out)[:400]}")
    except Exception as e:  # noqa: BLE001
        check(False, f"gpu_core CLI stdout is JSON ({type(e).__name__}: {e})")
    check_encodings(path, "results")
    a = read_h5ad_strict(path)
    probs = ex.validate_exchange(path)
    check(not probs, f"{path.name}: python validate_exchange is empty {probs or ''}")
    try:
        ex.read_exchange(path)
        check(True, f"{path.name}: cellwarden.read_exchange ok")
    except Exception as e:  # noqa: BLE001
        check(False, f"{path.name}: cellwarden.read_exchange raised {type(e).__name__}: {e}")
    n = EXPECTED["n_cells"]
    check(a.shape == (n, 0), f"{path.name}: X shape is cells x 0 (got {a.shape})")
    check(list(a.obs_names) == bcs, f"{path.name}: obs_names == fixture barcodes")
    exm = check_uns(a, mode="results", writer="python", n_genes=0, path=path)
    for key, dims in (("X_pca", N_PCS), ("X_harmony", N_PCS), ("X_umap", 2)):
        ok = key in a.obsm
        check(ok, f"{path.name}: obsm['{key}'] present")
        if ok:
            m = np.asarray(a.obsm[key])
            check(m.shape == (n, dims), f"{path.name}: obsm['{key}'] shape {(n, dims)} (got {m.shape})")
            check(m.dtype == np.float32, f"{path.name}: obsm['{key}'] float32 (got {m.dtype})")
            check(np.isfinite(m).all(), f"{path.name}: obsm['{key}'] finite")
    for key in ("connectivities", "distances"):
        ok = key in a.obsp
        check(ok, f"{path.name}: obsp['{key}'] present")
        if ok:
            check(sp.issparse(a.obsp[key]) and a.obsp[key].shape == (n, n),
                  f"{path.name}: obsp['{key}'] sparse {n}x{n}")
    ck = exm.get("cluster_key")
    if ck in a.obs:
        check(isinstance(a.obs[ck].dtype, pd.CategoricalDtype), f"{path.name}: obs['{ck}'] categorical")
        k = a.obs[ck].nunique()
        check(2 <= k <= 40, f"{path.name}: {k} clusters found")
        # Planted structure should be recovered: each cell type maps mostly to one cluster.
        tab = pd.crosstab(cells["cell_type"].values, a.obs[ck].astype(str).values)
        purity = tab.max(axis=1).sum() / n
        print(f"        cluster vs cell_type purity: {purity:.2f}")
        check(purity > 0.8, f"{path.name}: clusters recover planted cell types (purity {purity:.2f} > 0.8)")
        pd.DataFrame({"barcode": a.obs_names, "cluster": a.obs[ck].astype(str).values}).to_csv(
            work / "results_clusters.tsv", sep="\t", index=False)

    # Refuse-case files for step 4.
    bad = work / "results_badchecksum.h5ad"
    bad.write_bytes(path.read_bytes())
    with h5py.File(bad, "r+") as f:
        node = f["uns/exchange/source_counts_checksum"]
        v = node[()]
        v = v.decode() if isinstance(v, bytes) else str(v)
        newv = ("0" if v[0] != "0" else "1") + v[1:]
        attrs = dict(node.attrs)          # keep the writer's own encoding attributes
        del f["uns/exchange/source_counts_checksum"]
        ds = f["uns/exchange"].create_dataset("source_counts_checksum", data=newv,
                                              dtype=h5py.string_dtype("utf-8"))
        for k, val in attrs.items():
            ds.attrs[k] = val
    probs = ex.validate_exchange(bad)
    check(not probs, f"{bad.name}: altered-checksum copy is still spec-valid {probs or ''}")
    b2 = read_h5ad_strict(bad)
    check(b2.uns["exchange"]["source_counts_checksum"] != EXPECTED["counts_checksum"],
          f"{bad.name}: source_counts_checksum altered")

    sub = work / "results_subset.h5ad"
    try:
        _rewrite_results(path, sub, drop=7)
        s = read_h5ad_strict(sub)
        check(s.n_obs == n - 1 and bcs[7] not in set(s.obs_names),
              f"{sub.name}: subset results drops barcode {bcs[7]}")
        probs = ex.validate_exchange(sub)
        check(not probs, f"{sub.name}: subset results is spec-valid {probs or ''}")
        (work / "dropped_barcode.txt").write_text(bcs[7] + "\n")
    except Exception as e:  # noqa: BLE001
        check(False, f"{sub.name}: building subset results raised {type(e).__name__}: {e}")
    shuf = work / "results_shuffled.h5ad"
    try:
        _rewrite_results(path, shuf, shuffle_seed=1)
        s = read_h5ad_strict(shuf)
        check(s.n_obs == n and list(s.obs_names) != bcs and set(s.obs_names) == set(bcs),
              f"{shuf.name}: same cells as results.h5ad, different order")
        probs = ex.validate_exchange(shuf)
        check(not probs, f"{shuf.name}: shuffled results is spec-valid {probs or ''}")
    except Exception as e:  # noqa: BLE001
        check(False, f"{shuf.name}: building shuffled results raised {type(e).__name__}: {e}")
    extra = work / "results_extra.h5ad"
    try:
        _rewrite_results(path, extra, extra_barcode="NOTINOBJECTAAAAA-9")
        s = read_h5ad_strict(extra)
        check(s.n_obs == n + 1 and "NOTINOBJECTAAAAA-9" in set(s.obs_names),
              f"{extra.name}: results with one extra barcode")
        check(s.uns["exchange"]["source_counts_checksum"] == EXPECTED["counts_checksum"],
              f"{extra.name}: keeps the correct source_counts_checksum")
        probs = ex.validate_exchange(extra)
        check(not probs, f"{extra.name}: extra-barcode results is spec-valid {probs or ''}")
    except Exception as e:  # noqa: BLE001
        check(False, f"{extra.name}: building extra-barcode results raised {type(e).__name__}: {e}")
    finish("check-results")


def write_python(work: Path) -> None:
    import anndata as ad
    import scanpy as sc
    from cellwarden import exchange as ex
    feats, bcs, cells = fixture_tables()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)   # scipy spmatrix default change
        mm = scipy.io.mmread(FIXTURE / "matrix.mtx")
    counts = sp.csr_matrix(mm.T, dtype=np.int32)
    check(indep_checksum(counts) == EXPECTED["counts_checksum"], "fixture counts checksum (independent) == expected")
    check(ex.counts_checksum(counts) == EXPECTED["counts_checksum"], "cellwarden counts_checksum(fixture) == expected")

    obs = pd.DataFrame(index=pd.Index(bcs))
    for col in ("sample", "cell_type"):
        obs[col] = pd.Categorical(cells[col].values, categories=EXPECTED["obs_levels"][col])
    obs["n_counts"] = np.asarray(counts.sum(1)).ravel().astype(np.int64)
    obs["is_donor_A"] = (cells["sample"] == "donor_A").values
    obs["label"] = cells["cell_type"].str.lower().values          # plain strings
    var = pd.DataFrame(index=pd.Index(list(feats["symbol"])))      # originals, underscores kept
    var["gene_id"] = feats["gene_id"].values
    var["gene_symbol_original"] = feats["symbol"].values
    adata = ad.AnnData(X=counts.astype(np.float32), obs=obs, var=var)
    adata.layers["counts"] = counts.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X, dtype=np.float32)
    hv = np.zeros(adata.n_vars, dtype=bool)
    hv[adata.var_names.isin(EXPECTED["marker_genes_flat"])] = True
    adata.var["highly_variable"] = hv

    path = work / "py_full.h5ad"
    ex.write_exchange(adata, path, "full", run_id=RUN_ID, stage="IT_python_full",
                      config_hash=CONFIG_HASH, source_counts_checksum=None, cluster_key=None)
    probs = ex.validate_exchange(path)
    check(not probs, f"{path.name}: python validate_exchange is empty {probs or ''}")
    a = read_h5ad_strict(path)
    check_encodings(path, "full")
    check_uns(a, mode="full", writer="python", n_genes=EXPECTED["n_genes"], path=path)
    check(list(a.var_names) == list(feats["symbol"]), f"{path.name}: var_names keep original symbols")
    finish("write-python")


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    cmd, work = sys.argv[1], Path(sys.argv[2])
    {"check-r-files": check_r_files, "check-results": check_results,
     "write-python": write_python}[cmd](work)


if __name__ == "__main__":
    main()
