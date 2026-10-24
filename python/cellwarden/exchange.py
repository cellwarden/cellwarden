"""cellwarden exchange format (Seurat <-> AnnData), Python side.

Implements docs/exchange-spec.md v1.0: writing, reading and validating
exchange ``.h5ad`` files, and the language-independent counts checksum.

Checksum canonical form (spec section 6)
----------------------------------------
Before hashing, the matrix is converted to cells x genes CSR, duplicate
entries are summed, column indices are sorted within each row, and **explicit
zeros are eliminated** (stored entries whose value is 0 are dropped, and
``indptr`` is recomputed accordingly). The checksum therefore depends only on
the mathematical matrix, not on how a particular library chose to store it.
The R implementation must do the same (e.g. ``Matrix::drop0()`` on the
``dgCMatrix`` before reading ``@p``, ``@i`` and ``@x``).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "SCHEMA_VERSION",
    "MODES",
    "ExchangeError",
    "counts_checksum",
    "write_exchange",
    "read_exchange",
    "validate_exchange",
]

SCHEMA_VERSION = "1.0"
SUPPORTED_MAJOR = 1
MODES = ("full", "slim", "results")

_COMMON_KEYS = (
    "schema_version",
    "mode",
    "run_id",
    "stage",
    "writer",
    "created_utc",
    "config_hash",
    "n_cells",
    "n_genes",
    "cells_sha256",
    "genes_sha256",
    "packages",
)
_MODE_KEYS = {
    "full": ("counts_checksum",),
    "slim": ("source_counts_checksum",),
    "results": ("source_counts_checksum", "cluster_key"),
}
_STRING_KEYS = (
    "schema_version",
    "mode",
    "run_id",
    "stage",
    "writer",
    "created_utc",
    "config_hash",
    "cells_sha256",
    "genes_sha256",
    "counts_checksum",
    "source_counts_checksum",
    "cluster_key",
)


class ExchangeError(Exception):
    """An exchange file (or object to be written) violates the spec.

    ``problems`` holds the individual problems as a list of strings.
    """

    def __init__(self, problems: list[str] | str):
        if isinstance(problems, str):
            problems = [problems]
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


# --------------------------------------------------------------------------
# Checksums
# --------------------------------------------------------------------------


def counts_checksum(m: Any) -> str:
    """SHA-256 fingerprint of a cells x genes counts matrix (spec section 6).

    ``m`` may be any scipy sparse matrix/array or a dense array. It is
    canonicalised to CSR with duplicates summed, sorted column indices and
    explicit zeros eliminated, then hashed as: shape (2 x int64 LE), indptr
    (int64 LE), indices (int64 LE), data (float64 LE).
    """
    if sp.issparse(m):
        csr = sp.csr_matrix(m, copy=True)
    else:
        csr = sp.csr_matrix(np.asarray(m))
    csr.sum_duplicates()  # also sorts indices
    csr.eliminate_zeros()
    csr.sort_indices()
    h = hashlib.sha256()
    h.update(np.asarray(csr.shape, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(csr.indptr, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(csr.indices, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(csr.data, dtype="<f8").tobytes())
    return h.hexdigest()


def _names_sha256(names) -> str:
    return hashlib.sha256("\n".join(map(str, names)).encode("utf-8")).hexdigest()


def _packages() -> dict[str, str]:
    out = {}
    import platform

    out["python"] = platform.python_version()
    for pkg in ("anndata", "scanpy", "numpy", "rapids_singlecell"):
        dist = pkg.replace("_", "-")
        try:
            out[pkg] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            if pkg != "rapids_singlecell":
                out[pkg] = "not-installed"
    return out


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def _int_counts(m: Any) -> Any:
    """Return counts as int32 (if they fit) or int64; raise if non-integer."""
    is_sparse = sp.issparse(m)
    if is_sparse:
        m = sp.csr_matrix(m)
        vals = m.data
    else:
        m = np.asarray(m)
        vals = m
    if vals.size and not np.issubdtype(vals.dtype, np.integer):
        if not np.all(np.isfinite(vals)) or not np.all(vals == np.round(vals)):
            raise ExchangeError("layers['counts'] contains non-integer values")
    if vals.size:
        lo, hi = vals.min(), vals.max()
        i32 = np.iinfo(np.int32)
        dtype = np.int32 if (lo >= i32.min and hi <= i32.max) else np.int64
    else:
        dtype = np.int32
    return m.astype(dtype)


def _float32(m: Any) -> Any:
    if sp.issparse(m):
        return sp.csr_matrix(m, dtype=np.float32)
    return np.asarray(m, dtype=np.float32)


def _normalise_frame(df: pd.DataFrame, what: str) -> pd.DataFrame:
    """Coerce dataframe columns to the spec's on-disk types."""
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_bool_dtype(s.dtype) and not s.isna().any():
            df[col] = s.astype(bool)
        elif str(s.dtype) in ("boolean", "Int8", "Int16", "Int32", "Int64") and s.isna().any():
            warnings.warn(
                f"{what} column {col!r} has missing values; writing it as nullable "
                f"({'boolean' if str(s.dtype) == 'boolean' else 'integer'})",
                UserWarning,
                stacklevel=3,
            )
        elif pd.api.types.is_string_dtype(s.dtype) or s.dtype == object:
            if s.isna().any():
                warnings.warn(
                    f"{what} column {col!r} is a string column with missing values; "
                    "writing it as a categorical (missing = code -1)",
                    UserWarning,
                    stacklevel=3,
                )
                df[col] = s.astype("category")
            else:
                df[col] = s.astype(str)
    return df


def write_exchange(
    adata: ad.AnnData,
    path: str | Path,
    mode: str,
    *,
    run_id: str = "none",
    stage: str = "unknown",
    config_hash: str = "none",
    source_counts_checksum: str | None = None,
    cluster_key: str | None = None,
) -> None:
    """Write ``adata`` as an exchange file in ``mode`` (full / slim / results).

    - full: X (float32) + layers["counts"] (int32/int64), all genes;
      ``counts_checksum`` computed from the counts.
    - slim: X restricted to ``var["highly_variable"]`` genes; no layers.
    - results: X of shape (n_cells, 0); obs, obsm, obsp kept; no var columns.

    ``source_counts_checksum`` (slim/results) defaults to an existing
    ``uns["exchange"]`` value, else the checksum of ``layers["counts"]``.
    ``cluster_key`` (results) defaults to an existing ``uns["exchange"]`` value.
    The input object is not modified.
    """
    if mode not in MODES:
        raise ExchangeError(f"mode must be one of {MODES}, got {mode!r}")
    problems = []
    if not adata.obs_names.is_unique:
        dups = adata.obs_names[adata.obs_names.duplicated()].unique()[:5].tolist()
        problems.append(f"obs_names (cell barcodes) are not unique, e.g. {dups}")
    if not adata.var_names.is_unique:
        dups = adata.var_names[adata.var_names.duplicated()].unique()[:5].tolist()
        problems.append(f"var_names (genes) are not unique, e.g. {dups}")
    if problems:
        raise ExchangeError(problems)

    prev = dict(adata.uns.get("exchange", {}) or {})
    if mode in ("slim", "results") and source_counts_checksum is None:
        source_counts_checksum = prev.get("source_counts_checksum") or prev.get("counts_checksum")
        if source_counts_checksum is None and "counts" in adata.layers:
            source_counts_checksum = counts_checksum(adata.layers["counts"])
        if source_counts_checksum is None:
            raise ExchangeError(
                f"{mode} mode needs source_counts_checksum (pass it, or provide "
                "layers['counts'] or uns['exchange'])"
            )
    if mode == "results":
        if cluster_key is None:
            cluster_key = prev.get("cluster_key")
        if cluster_key is None:
            raise ExchangeError("results mode needs cluster_key")
        if cluster_key not in adata.obs.columns:
            raise ExchangeError(f"cluster_key {cluster_key!r} is not an obs column")

    obs = _normalise_frame(adata.obs, "obs")
    obs.index = pd.Index(adata.obs_names.astype(str), dtype=object, name=None)
    if cluster_key is not None and not isinstance(obs[cluster_key].dtype, pd.CategoricalDtype):
        obs[cluster_key] = obs[cluster_key].astype(str).astype("category")

    var = adata.var.copy()
    var.index = pd.Index(adata.var_names.astype(str), dtype=object, name=None)
    if "gene_symbol_original" not in var.columns:
        var["gene_symbol_original"] = var.index.to_numpy(dtype=object)
    if "gene_id" not in var.columns:
        var["gene_id"] = ""
    var["gene_id"] = var["gene_id"].fillna("")
    if "highly_variable" in var.columns:
        var["highly_variable"] = var["highly_variable"].fillna(False).astype(bool)
    var = _normalise_frame(var, "var")

    obsm = {k: (np.asarray(v, dtype=np.float32) if not sp.issparse(v) else v) for k, v in adata.obsm.items()}
    obsp = {k: sp.csr_matrix(v) for k, v in adata.obsp.items()}
    uns = {k: v for k, v in adata.uns.items() if k != "exchange"}

    n_cells = adata.n_obs
    if mode == "full":
        if "counts" not in adata.layers:
            raise ExchangeError("full mode needs layers['counts']")
        counts = _int_counts(adata.layers["counts"])
        out = ad.AnnData(
            X=_float32(adata.X),
            obs=obs,
            var=var,
            obsm=obsm,
            obsp=obsp,
            varm={k: np.asarray(v, dtype=np.float32) for k, v in adata.varm.items()},
            layers={"counts": counts},
            uns=uns,
        )
    elif mode == "slim":
        if "highly_variable" in var.columns:
            mask = var["highly_variable"].to_numpy(dtype=bool)
        else:
            warnings.warn(
                "slim mode: var['highly_variable'] missing; sending all genes",
                UserWarning,
                stacklevel=2,
            )
            mask = np.ones(adata.n_vars, dtype=bool)
            var["highly_variable"] = True
        if not mask.any():
            raise ExchangeError("slim mode: no highly variable genes")
        X = adata.X[:, mask] if mask.size else adata.X
        out = ad.AnnData(X=_float32(X), obs=obs, var=var.iloc[mask].copy(), obsm=obsm, obsp=obsp, uns=uns)
    else:  # results
        out = ad.AnnData(
            X=sp.csr_matrix((n_cells, 0), dtype=np.float32),
            obs=obs,
            var=pd.DataFrame(index=pd.Index([], dtype=object)),
            obsm=obsm,
            obsp=obsp,
            uns=uns,
        )

    ex: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "run_id": str(run_id),
        "stage": str(stage),
        "writer": "python",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_hash": str(config_hash),
        "n_cells": np.int64(out.n_obs),
        "n_genes": np.int64(out.n_vars),
        "cells_sha256": _names_sha256(out.obs_names),
        "genes_sha256": _names_sha256(out.var_names),
        "packages": _packages(),
    }
    if mode == "full":
        ex["counts_checksum"] = counts_checksum(out.layers["counts"])
    else:
        ex["source_counts_checksum"] = str(source_counts_checksum)
    if mode == "results":
        ex["cluster_key"] = str(cluster_key)
    out.uns["exchange"] = ex

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Plain (non-nullable) string arrays, as the spec requires; strings with
    # missing values were turned into categoricals above.
    with ad.settings.override(allow_write_nullable_strings=False):
        out.write_h5ad(path, convert_strings_to_categoricals=False)


# --------------------------------------------------------------------------
# Reading and validation
# --------------------------------------------------------------------------


def _check_encoding(f: h5py.File) -> list[str]:
    problems = []
    attrs = f.attrs
    if attrs.get("encoding-type") != "anndata":
        problems.append("root group encoding-type is not 'anndata'")
    if attrs.get("encoding-version") != "0.1.0":
        problems.append("root group encoding-version is not '0.1.0'")
    for key in ("obs", "var"):
        if key not in f:
            problems.append(f"missing /{key}")
        elif f[key].attrs.get("encoding-type") != "dataframe":
            problems.append(f"/{key} encoding-type is not 'dataframe'")
    if "uns" not in f or "exchange" not in f["uns"]:
        problems.append("missing uns['exchange']")
    return problems


def _validate(adata: ad.AnnData, f: h5py.File | None) -> list[str]:
    problems: list[str] = []
    ex = adata.uns.get("exchange")
    if not isinstance(ex, dict):
        return ["uns['exchange'] missing or not a mapping"]

    sv = ex.get("schema_version")
    if not isinstance(sv, str):
        problems.append("uns['exchange']['schema_version'] missing")
    else:
        try:
            major = int(sv.split(".")[0])
        except ValueError:
            major = None
        if major != SUPPORTED_MAJOR:
            problems.append(f"unsupported schema_version {sv!r} (supported major: {SUPPORTED_MAJOR})")

    mode = ex.get("mode")
    if mode not in MODES:
        problems.append(f"uns['exchange']['mode'] is {mode!r}, expected one of {MODES}")
        required: tuple[str, ...] = _COMMON_KEYS
    else:
        required = _COMMON_KEYS + _MODE_KEYS[mode]
    missing = [k for k in required if k not in ex]
    if missing:
        problems.append(f"uns['exchange'] missing required keys: {missing}")
    for k in _STRING_KEYS:
        if k in ex and not isinstance(ex[k], str):
            problems.append(f"uns['exchange']['{k}'] must be a string")
    if ex.get("writer") not in (None, "R", "python"):
        problems.append(f"uns['exchange']['writer'] is {ex.get('writer')!r}, expected 'R' or 'python'")
    if "packages" in ex and not isinstance(ex["packages"], dict):
        problems.append("uns['exchange']['packages'] must be a mapping")
    if "created_utc" in ex and not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", str(ex["created_utc"])
    ):
        problems.append("uns['exchange']['created_utc'] must be YYYY-MM-DDTHH:MM:SSZ")

    for k, actual in (("n_cells", adata.n_obs), ("n_genes", adata.n_vars)):
        if k in ex:
            v = ex[k]
            if not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)):
                problems.append(f"uns['exchange']['{k}'] must be an integer")
            elif int(v) != actual:
                problems.append(f"uns['exchange']['{k}'] = {int(v)} but file has {actual}")
    if "cells_sha256" in ex and ex["cells_sha256"] != _names_sha256(adata.obs_names):
        problems.append("cells_sha256 does not match obs_names")
    if "genes_sha256" in ex and ex["genes_sha256"] != _names_sha256(adata.var_names):
        problems.append("genes_sha256 does not match var_names")

    if not adata.obs_names.is_unique:
        problems.append("obs_names (cell barcodes) are not unique")
    if not adata.var_names.is_unique:
        problems.append("var_names are not unique")

    n, g = adata.n_obs, adata.n_vars
    if adata.X is None:
        problems.append("X is missing")
    elif adata.X.shape != (n, g):
        problems.append(f"X shape {adata.X.shape} != ({n}, {g})")
    if mode == "results" and g != 0:
        problems.append(f"results mode X must have 0 columns, has {g}")
    for k, v in adata.layers.items():
        if v.shape != (n, g):
            problems.append(f"layers['{k}'] shape {v.shape} != ({n}, {g})")
    for k, v in adata.obsm.items():
        if v.shape[0] != n:
            problems.append(f"obsm['{k}'] has {v.shape[0]} rows, expected {n}")
    for k, v in adata.varm.items():
        if v.shape[0] != g:
            problems.append(f"varm['{k}'] has {v.shape[0]} rows, expected {g}")
    for k, v in adata.obsp.items():
        if v.shape != (n, n):
            problems.append(f"obsp['{k}'] shape {v.shape} != ({n}, {n})")

    if mode == "full":
        if "counts" not in adata.layers:
            problems.append("full mode file has no layers['counts']")
        elif "counts_checksum" in ex and counts_checksum(adata.layers["counts"]) != ex["counts_checksum"]:
            problems.append("counts_checksum does not match layers['counts']")
    if mode == "results":
        ck = ex.get("cluster_key")
        if isinstance(ck, str):
            if ck not in adata.obs.columns:
                problems.append(f"cluster_key {ck!r} is not an obs column")
            elif not isinstance(adata.obs[ck].dtype, pd.CategoricalDtype):
                problems.append(f"obs['{ck}'] must be categorical")
    return problems


def _read_with_problems(path: str | Path) -> tuple[ad.AnnData | None, list[str]]:
    path = Path(path)
    if not path.exists():
        return None, [f"file not found: {path}"]
    try:
        with h5py.File(path, "r") as f:
            problems = _check_encoding(f)
            # Barcodes/genes uniqueness straight from disk: anndata may
            # otherwise only warn on read.
            for key in ("obs", "var"):
                if key in f:
                    idx_name = f[key].attrs.get("_index", "_index")
                    if idx_name in f[key] and isinstance(f[key][idx_name], h5py.Dataset):
                        vals = f[key][idx_name][()]
                        if len(vals) != len(set(vals.tolist())):
                            what = "obs_names (cell barcodes)" if key == "obs" else "var_names"
                            problems.append(f"{what} are not unique")
    except OSError as e:
        return None, [f"not a readable HDF5 file: {e}"]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adata = ad.read_h5ad(path)
    except Exception as e:  # anndata refuses inconsistent files
        return None, problems + [f"anndata could not read the file: {e}"]
    for w in caught:
        problems.append(f"anndata warning on read: {w.message}")
    for p in _validate(adata, None):
        if p not in problems:
            problems.append(p)
    return adata, problems


def validate_exchange(path: str | Path) -> list[str]:
    """Check an exchange file against the spec; return problems (empty = valid)."""
    return _read_with_problems(path)[1]


def read_exchange(path: str | Path, *, validate: bool = True) -> ad.AnnData:
    """Read an exchange file. With ``validate=True`` raise ExchangeError on problems."""
    if not validate:
        return ad.read_h5ad(path)
    adata, problems = _read_with_problems(path)
    if problems or adata is None:
        raise ExchangeError(problems or ["could not read file"])
    return adata
