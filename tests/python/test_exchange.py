import hashlib
import struct
import warnings

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from cellwarden.exchange import (
    ExchangeError,
    counts_checksum,
    read_exchange,
    validate_exchange,
    write_exchange,
)

# [[0, 1, 0],
#  [2, 0, 3]]
DENSE = np.array([[0, 1, 0], [2, 0, 3]])


def _manual_hex():
    parts = [
        struct.pack("<2q", 2, 3),  # shape
        struct.pack("<3q", 0, 1, 3),  # indptr
        struct.pack("<3q", 1, 0, 2),  # indices
        struct.pack("<3d", 1.0, 2.0, 3.0),  # data
    ]
    return hashlib.sha256(b"".join(parts)).hexdigest()


def test_checksum_matches_manual():
    expected = _manual_hex()
    assert counts_checksum(sp.csr_matrix(DENSE)) == expected
    assert counts_checksum(DENSE) == expected
    assert counts_checksum(DENSE.astype(np.float32)) == expected


def test_checksum_invariant_to_storage():
    expected = _manual_hex()
    # unsorted column indices within row 1
    unsorted = sp.csr_matrix(
        (np.array([1, 3, 2]), np.array([1, 2, 0]), np.array([0, 1, 3])), shape=(2, 3)
    )
    assert not unsorted.has_sorted_indices
    assert counts_checksum(unsorted) == expected
    assert counts_checksum(sp.csc_matrix(DENSE)) == expected
    assert counts_checksum(sp.coo_matrix(DENSE)) == expected
    assert counts_checksum(sp.csr_array(DENSE.astype(np.int64))) == expected
    # explicit zero stored -> eliminated before hashing
    explicit0 = sp.csr_matrix(
        (np.array([1, 0, 2, 3]), np.array([1, 2, 0, 2]), np.array([0, 2, 4])), shape=(2, 3)
    )
    assert explicit0.nnz == 4
    assert counts_checksum(explicit0) == expected
    # input must not be mutated
    assert explicit0.nnz == 4 and not unsorted.has_sorted_indices


def test_checksum_sensitive_to_change():
    m = DENSE.copy()
    m[0, 0] = 1
    assert counts_checksum(m) != _manual_hex()
    assert counts_checksum(DENSE.T) != _manual_hex()


def _toy(n_cells=40, n_genes=25, seed=1):
    rng = np.random.default_rng(seed)
    counts = sp.random(n_cells, n_genes, density=0.3, format="csr", random_state=seed)
    counts.data = rng.integers(1, 50, size=counts.nnz).astype(np.float64)
    X = counts.copy()
    X.data = np.log1p(X.data).astype(np.float32)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                rng.choice(["T", "B", "NK"], n_cells), categories=["T", "NK", "B", "unused"]
            ),
            "ordered_cat": pd.Categorical(rng.choice(["lo", "hi"], n_cells), categories=["lo", "hi"], ordered=True),
            "sample": [f"S{i % 3}" for i in range(n_cells)],
            "is_doublet": rng.random(n_cells) < 0.1,
            "n_count": np.asarray(counts.sum(1)).ravel().astype(np.int32),
            "n_big": np.arange(n_cells, dtype=np.int64) * 10_000_000_000,
            "pct_mt": rng.random(n_cells) * 10,
        },
        index=[f"AAAC_{i:04d}-1" for i in range(n_cells)],
    )
    var = pd.DataFrame(
        {"highly_variable": np.arange(n_genes) % 2 == 0},
        index=[f"GENE_{i}_x" if i % 3 == 0 else f"G{i}" for i in range(n_genes)],
    )
    a = ad.AnnData(X=X, obs=obs, var=var, layers={"counts": counts})
    a.obsm["X_pca"] = rng.normal(size=(n_cells, 5))
    a.obsm["X_umap"] = rng.normal(size=(n_cells, 2))
    g = sp.random(n_cells, n_cells, density=0.1, format="csr", random_state=seed)
    a.obsp["connectivities"] = g
    a.obsp["distances"] = g
    a.varm["PCs"] = rng.normal(size=(n_genes, 5))
    return a


def _read_no_warnings(path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return ad.read_h5ad(path)


def _check_obs(orig, back):
    assert list(back.obs_names) == list(orig.obs_names)
    for col in orig.obs.columns:
        if col not in back.obs.columns:
            continue
        o, b = orig.obs[col], back.obs[col]
        if isinstance(o.dtype, pd.CategoricalDtype):
            assert isinstance(b.dtype, pd.CategoricalDtype)
            assert list(b.cat.categories) == list(o.cat.categories)
            assert b.cat.ordered == o.cat.ordered
            assert list(b.astype(str)) == list(o.astype(str))
        else:
            assert b.dtype == o.dtype, col
            np.testing.assert_array_equal(b.to_numpy(), o.to_numpy())


def test_round_trip_full(tmp_path):
    a = _toy()
    p = tmp_path / "full.h5ad"
    write_exchange(a, p, "full", run_id="r1", stage="S1", config_hash="abc")
    assert validate_exchange(p) == []
    b = _read_no_warnings(p)
    b2 = read_exchange(p)
    assert b2.shape == a.shape
    _check_obs(a, b)
    assert list(b.var_names) == list(a.var_names)
    assert list(b.var["gene_symbol_original"]) == list(a.var_names)
    assert list(b.var["gene_id"]) == [""] * a.n_vars
    assert b.var["highly_variable"].dtype == bool
    assert b.X.dtype == np.float32
    assert b.layers["counts"].dtype == np.int32
    np.testing.assert_array_equal(b.layers["counts"].toarray(), a.layers["counts"].toarray())
    np.testing.assert_allclose(b.X.toarray(), a.X.toarray())
    assert b.obsm["X_pca"].dtype == np.float32
    assert b.varm["PCs"].shape == (a.n_vars, 5)
    ex = b.uns["exchange"]
    assert ex["mode"] == "full" and ex["writer"] == "python" and ex["schema_version"] == "1.0"
    assert ex["run_id"] == "r1" and ex["stage"] == "S1" and ex["config_hash"] == "abc"
    assert ex["counts_checksum"] == counts_checksum(a.layers["counts"])
    assert int(ex["n_cells"]) == a.n_obs and int(ex["n_genes"]) == a.n_vars
    assert {"python", "anndata", "scanpy", "numpy"} <= set(ex["packages"])
    # on-disk encodings required by the spec
    with h5py.File(p) as f:
        assert f.attrs["encoding-type"] == "anndata"
        assert f["obs/sample"].attrs["encoding-type"] == "string-array"
        assert f["obs/_index"].attrs["encoding-type"] == "string-array"
        assert f["obs/cell_type"].attrs["encoding-type"] == "categorical"
        assert f["layers/counts"].attrs["encoding-type"] == "csr_matrix"
        assert f["uns/exchange"].attrs["encoding-type"] == "dict"


def test_large_counts_use_int64(tmp_path):
    a = _toy()
    a.layers["counts"] = a.layers["counts"].copy()
    a.layers["counts"].data[0] = 2**33
    p = tmp_path / "big.h5ad"
    write_exchange(a, p, "full")
    assert read_exchange(p).layers["counts"].dtype == np.int64


def test_round_trip_slim(tmp_path):
    a = _toy()
    p = tmp_path / "slim.h5ad"
    write_exchange(a, p, "slim", run_id="r1")
    assert validate_exchange(p) == []
    b = _read_no_warnings(p)
    hv = a.var["highly_variable"].to_numpy()
    assert b.shape == (a.n_obs, int(hv.sum()))
    assert list(b.var_names) == list(a.var_names[hv])
    assert "counts" not in b.layers
    assert "PCs" not in b.varm
    _check_obs(a, b)
    ex = b.uns["exchange"]
    assert ex["source_counts_checksum"] == counts_checksum(a.layers["counts"])
    assert "counts_checksum" not in ex


def test_round_trip_results(tmp_path):
    a = _toy()
    a.obs["leiden"] = pd.Categorical([str(i % 4) for i in range(a.n_obs)])
    p = tmp_path / "res.h5ad"
    write_exchange(a, p, "results", source_counts_checksum="deadbeef", cluster_key="leiden")
    assert validate_exchange(p) == []
    b = _read_no_warnings(p)
    assert b.shape == (a.n_obs, 0)
    assert b.X.shape == (a.n_obs, 0)
    _check_obs(a, b)
    assert b.obsm["X_umap"].shape == (a.n_obs, 2)
    assert b.obsp["connectivities"].shape == (a.n_obs, a.n_obs)
    ex = b.uns["exchange"]
    assert ex["cluster_key"] == "leiden" and ex["source_counts_checksum"] == "deadbeef"


def test_results_requires_cluster_key(tmp_path):
    a = _toy()
    with pytest.raises(ExchangeError):
        write_exchange(a, tmp_path / "r.h5ad", "results", source_counts_checksum="x")


def test_write_rejects_duplicates(tmp_path):
    a = _toy()
    names = list(a.obs_names)
    names[1] = names[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a.obs_names = names
    with pytest.raises(ExchangeError, match="not unique"):
        write_exchange(a, tmp_path / "d.h5ad", "full")
    a = _toy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a.var_names = ["G"] * a.n_vars
    with pytest.raises(ExchangeError, match="var_names"):
        write_exchange(a, tmp_path / "d.h5ad", "full")


def test_bad_mode(tmp_path):
    with pytest.raises(ExchangeError):
        write_exchange(_toy(), tmp_path / "x.h5ad", "fat")


def test_validate_catches_tampered_n_cells(tmp_path):
    p = tmp_path / "full.h5ad"
    write_exchange(_toy(), p, "full")
    with h5py.File(p, "r+") as f:
        ds = f["uns/exchange/n_cells"]
        attrs = dict(ds.attrs)
        del f["uns/exchange/n_cells"]
        f["uns/exchange"].create_dataset("n_cells", data=np.int64(999))
        f["uns/exchange/n_cells"].attrs.update(attrs)
    problems = validate_exchange(p)
    assert any("n_cells" in x for x in problems), problems
    with pytest.raises(ExchangeError) as ei:
        read_exchange(p)
    assert ei.value.problems == problems
    # validate=False still reads
    assert read_exchange(p, validate=False).n_obs == 40


def test_validate_catches_duplicate_barcodes(tmp_path):
    p = tmp_path / "full.h5ad"
    write_exchange(_toy(), p, "full")
    with h5py.File(p, "r+") as f:
        idx = f["obs/_index"]
        vals = idx[()]
        vals[1] = vals[0]
        idx[...] = vals
    problems = validate_exchange(p)
    assert any("not unique" in x for x in problems), problems
    assert any("cells_sha256" in x for x in problems), problems


def test_validate_catches_counts_tampering(tmp_path):
    p = tmp_path / "full.h5ad"
    write_exchange(_toy(), p, "full")
    with h5py.File(p, "r+") as f:
        d = f["layers/counts/data"]
        d[0] = d[0] + 1
    assert any("counts_checksum" in x for x in validate_exchange(p))


def test_validate_catches_schema_and_missing_keys(tmp_path):
    p = tmp_path / "slim.h5ad"
    write_exchange(_toy(), p, "slim")
    with h5py.File(p, "r+") as f:
        attrs = dict(f["uns/exchange/schema_version"].attrs)
        del f["uns/exchange/schema_version"]
        f["uns/exchange"].create_dataset("schema_version", data="2.0", dtype=h5py.string_dtype())
        f["uns/exchange/schema_version"].attrs.update(attrs)
        del f["uns/exchange/source_counts_checksum"]
    problems = validate_exchange(p)
    assert any("schema_version" in x for x in problems), problems
    assert any("source_counts_checksum" in x for x in problems), problems


def test_validate_missing_file(tmp_path):
    assert validate_exchange(tmp_path / "nope.h5ad")


def test_write_does_not_mutate_input(tmp_path):
    a = _toy()
    before = a.copy()
    write_exchange(a, tmp_path / "s.h5ad", "slim")
    assert a.shape == before.shape
    assert "exchange" not in a.uns
    assert "gene_symbol_original" not in a.var.columns
