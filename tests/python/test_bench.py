"""Unit tests for the benchmark harness in bench/ (docs/RUST_ENGINE.md section 8)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

BENCH = Path(__file__).resolve().parents[2] / "bench"
sys.path.insert(0, str(BENCH))

import common  # noqa: E402
import h5io  # noqa: E402
import parity  # noqa: E402
import prep  # noqa: E402
import report  # noqa: E402
import synth  # noqa: E402


def _orthogonal(n, rng):
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    return q * np.sign(np.diag(r))


# ---------------------------------------------------------------- parity.subspace_similarity


def test_subspace_identical():
    x = np.random.default_rng(0).normal(size=(200, 5))
    sim, sim_min = parity.subspace_similarity(x, x.copy(), 5)
    assert sim == pytest.approx(1.0, abs=1e-6)
    assert sim_min == pytest.approx(1.0, abs=1e-6)


def test_subspace_rotation_and_sign_flips():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 5))
    y = x @ _orthogonal(5, rng) * np.array([1, -1, 1, -1, -1])
    sim, sim_min = parity.subspace_similarity(x, y, 5)
    assert sim == pytest.approx(1.0, abs=1e-6)
    assert sim_min == pytest.approx(1.0, abs=1e-6)


def test_subspace_orthogonal():
    rng = np.random.default_rng(2)
    # Columns of one orthonormal basis, centred, split into two disjoint subspaces.
    z = rng.normal(size=(100, 6))
    z -= z.mean(axis=0)
    q, _ = np.linalg.qr(z)
    sim, sim_min = parity.subspace_similarity(q[:, :3], q[:, 3:], 3)
    assert sim == pytest.approx(0.0, abs=1e-6)
    assert sim_min == pytest.approx(0.0, abs=1e-6)


def test_subspace_partial_overlap():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(100, 4))
    z -= z.mean(axis=0)
    q, _ = np.linalg.qr(z)
    a = q[:, [0, 1]]
    b = q[:, [0, 2]]  # shares one direction of two
    sim, sim_min = parity.subspace_similarity(a, b, 2)
    assert sim == pytest.approx(0.5, abs=1e-6)
    assert sim_min == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < sim < 1.0


def test_subspace_n_capped_by_columns():
    x = np.random.default_rng(4).normal(size=(50, 3))
    sim, _ = parity.subspace_similarity(x, x, 10)
    assert sim == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------- parity.knn_overlap


def test_knn_overlap_identical_and_order_free():
    a = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    assert parity.knn_overlap(a, a) == 1.0
    assert parity.knn_overlap(a, a[:, ::-1]) == 1.0


def test_knn_overlap_disjoint():
    a = np.array([[1, 2], [2, 3]])
    b = np.array([[3, 4], [0, 4]])
    assert parity.knn_overlap(a, b) == 0.0


def test_knn_overlap_padding():
    a = np.array([[1, 2, -1], [0, -1, -1], [-1, -1, -1]])
    b = np.array([[2, 1, 3], [0, 2, 3], [-1, -1, -1]])
    # row 0: {1,2} vs {1,2,3} -> 2/2; row 1: {0} vs {0,2,3} -> 1/1; row 2: empty -> 0
    assert parity.knn_overlap(a, b) == pytest.approx(2 / 3)
    half = np.array([[1, 5, -1], [0, 4, -1], [-1, -1, -1]])
    # row 0: {1,5} & {1,2,3} -> 1/2; row 1: {0,4} & {0,2,3} -> 1/2
    assert parity.knn_overlap(half, b) == pytest.approx((0.5 + 0.5) / 3)


def test_knn_overlap_partial():
    a = np.array([[1, 2, 3, 4]])
    b = np.array([[1, 2, 7, 8]])
    assert parity.knn_overlap(a, b) == pytest.approx(0.5)


# ---------------------------------------------------------------- parity.remap_knn / compare


def _engine_outputs(n=60, p=6, k=5, seed=0):
    rng = np.random.default_rng(seed)
    barcodes = [f"CELL{i:04d}-1" for i in range(n)]
    pcs = rng.normal(size=(n, p)).astype(np.float32)
    d = ((pcs[:, None, :] - pcs[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    knn = np.argsort(d, axis=1)[:, :k].astype(np.int32)
    knn[0, -2:] = -1  # some padding
    labels = [str(i % 3) for i in range(n)]
    return barcodes, pcs, knn, labels


def _shuffled(barcodes, pcs, knn, labels, rng, relabel=True):
    n = len(barcodes)
    perm = rng.permutation(n)  # new row j holds old cell perm[j]
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    bc2 = [barcodes[i] for i in perm]
    rot = _orthogonal(pcs.shape[1], rng)
    pcs2 = (pcs[perm] @ rot).astype(np.float32)
    old = knn[perm]
    knn2 = np.where(old >= 0, inv[np.clip(old, 0, None)], -1).astype(np.int32)
    knn2 = knn2[:, ::-1]  # neighbour order within a row does not matter
    names = {"0": "x", "1": "y", "2": "z"} if relabel else {s: s for s in "012"}
    lab2 = [names[labels[i]] for i in perm]
    return bc2, pcs2, knn2, lab2


def test_remap_knn_identity():
    bc, _, knn, _ = _engine_outputs(n=10, k=3)
    idx = {b: i for i, b in enumerate(bc)}
    out = parity.remap_knn(knn, bc, idx, np.arange(10))
    np.testing.assert_array_equal(out, knn)


def test_remap_knn_permutation():
    rng = np.random.default_rng(5)
    bc, pcs, knn, lab = _engine_outputs(n=20, k=4)
    bc2, _, knn2, _ = _shuffled(bc, pcs, knn, lab, rng)
    ref_index = {b: i for i, b in enumerate(bc)}
    eng_index = {b: i for i, b in enumerate(bc2)}
    order = np.array([eng_index[b] for b in bc])
    out = parity.remap_knn(knn2, bc2, ref_index, order)
    np.testing.assert_array_equal(np.sort(out, axis=1), np.sort(knn, axis=1))


def test_compare_shuffled_engine(tmp_path):
    rng = np.random.default_rng(6)
    ref = _engine_outputs()
    eng = _shuffled(*ref, rng)
    common.write_outputs(tmp_path / "ref", *ref)
    common.write_outputs(tmp_path / "eng", *eng)
    truth = {b: f"t{int(l)}" for b, l in zip(ref[0], ref[3])}
    out = parity.compare(tmp_path / "ref", tmp_path / "eng", 6, truth)
    assert out["pc_similarity"] == pytest.approx(1.0, abs=1e-4)
    assert out["pc_similarity_min"] == pytest.approx(1.0, abs=1e-4)
    assert out["knn_overlap"] == 1.0
    assert out["ari"] == 1.0
    assert out["ari_truth"] == 1.0 and out["ari_truth_ref"] == 1.0
    assert out["n_clusters"] == out["n_clusters_ref"] == 3
    assert out["pass_all"] is True
    assert set(out["pass"]) == set(common.PARITY_THRESHOLDS)


def test_compare_detects_differences(tmp_path):
    rng = np.random.default_rng(7)
    ref = _engine_outputs()
    bc, pcs, knn, lab = ref
    other_pcs = rng.normal(size=pcs.shape).astype(np.float32)
    other_knn = np.roll(knn, 7, axis=0)
    common.write_outputs(tmp_path / "ref", *ref)
    common.write_outputs(tmp_path / "eng", bc, other_pcs, other_knn, [str(i % 5) for i in range(len(bc))])
    out = parity.compare(tmp_path / "ref", tmp_path / "eng", 6, None)
    assert out["pc_similarity"] < 0.9
    assert out["knn_overlap"] < 0.5
    assert out["ari"] < 0.9
    assert out["pass_all"] is False
    assert "ari_truth" not in out


@pytest.mark.parametrize("change", ["drop", "rename"])
def test_compare_mismatched_cells(tmp_path, change):
    bc, pcs, knn, lab = _engine_outputs(n=20, k=3)
    common.write_outputs(tmp_path / "ref", bc, pcs, knn, lab)
    if change == "drop":
        eng = (bc[:-1], pcs[:-1], np.where(knn[:-1] == 19, -1, knn[:-1]), lab[:-1])
    else:
        eng = (bc[:-1] + ["OTHER-1"], pcs, knn, lab)
    common.write_outputs(tmp_path / "eng", *eng)
    with pytest.raises(ValueError, match="cell sets differ"):
        parity.compare(tmp_path / "ref", tmp_path / "eng", 6, None)


# ---------------------------------------------------------------- common outputs


def test_write_read_outputs_roundtrip(tmp_path):
    bc = ["AAA-1", "CCC-1", "GGG-1"]
    pcs = np.arange(12, dtype=np.float64).reshape(3, 4) / 7
    knn = np.array([[1, 2], [0, -1], [1, 0]], dtype=np.int64)
    labels = [0, 1, 1]
    common.write_outputs(tmp_path / "o", bc, pcs, knn, labels)
    meta = json.loads((tmp_path / "o" / "meta.json").read_text())
    assert meta == {"n_cells": 3, "n_pcs": 4, "k": 2, "order": "row-major",
                    "knn_base": 0, "knn_self": "excluded"}
    assert (tmp_path / "o" / "pcs.f32").stat().st_size == 3 * 4 * 4
    assert (tmp_path / "o" / "knn.i32").stat().st_size == 3 * 2 * 4
    b2, p2, k2, l2 = common.read_outputs(tmp_path / "o")
    assert b2 == bc
    assert l2 == ["0", "1", "1"]
    assert p2.dtype == np.dtype("<f4") and p2.shape == (3, 4)
    assert k2.dtype == np.dtype("<i4") and k2.shape == (3, 2)
    np.testing.assert_allclose(p2, pcs.astype(np.float32))
    np.testing.assert_array_equal(k2, knn)


def test_write_outputs_fortran_input(tmp_path):
    pcs = np.asfortranarray(np.arange(6, dtype=np.float32).reshape(3, 2))
    common.write_outputs(tmp_path, ["a", "b", "c"], pcs, np.zeros((3, 1), np.int32), ["x"] * 3)
    _, p2, _, _ = common.read_outputs(tmp_path)
    np.testing.assert_array_equal(p2, pcs)


def test_knn_from_distances():
    # Row 0 stores itself explicitly; row 2 has only one neighbour; row 3 none.
    rows = [0, 0, 0, 0, 1, 1, 1, 2]
    cols = [0, 3, 1, 2, 0, 2, 3, 1]
    vals = [0.0, 0.5, 2.0, 1.0, 2.0, 0.3, 0.7, 0.9]
    d = sp.csr_matrix((vals, (rows, cols)), shape=(4, 4))
    out = common.knn_from_distances(d, 2)
    assert out.dtype == np.int32 and out.shape == (4, 2)
    np.testing.assert_array_equal(out, [[3, 2], [2, 3], [1, -1], [-1, -1]])
    out3 = common.knn_from_distances(d, 3)
    np.testing.assert_array_equal(out3[0], [3, 2, 1])
    np.testing.assert_array_equal(out3[1], [2, 3, 0])
    for i, row in enumerate(out3):
        assert i not in row


# ---------------------------------------------------------------- h5io


def _tiny_counts():
    m = sp.csr_matrix(np.array([[0, 3, 0, 1], [5, 0, 0, 0], [0, 0, 2, 7]], dtype=np.int32))
    return m, ["AAAC-1", "AAAG-1", "TTTC-1"], ["ENSG1", "ENSG2", "ENSG3", "ENSG4"], ["CD3E", "MT-CO1", "LYZ", "CD3E"]


def test_h5_roundtrip(tmp_path):
    m, bc, ids, names = _tiny_counts()
    f = tmp_path / "sub" / "matrix.h5"
    h5io.write_10x_h5(f, m, bc, ids, names)
    m2, bc2, ids2, names2 = h5io.read_counts(f)
    assert m2.shape == (3, 4)
    assert (m2 != m).nnz == 0
    assert bc2 == bc and ids2 == ids and names2 == names


def test_h5_read_by_scanpy(tmp_path):
    sc = pytest.importorskip("scanpy")
    m, bc, ids, names = _tiny_counts()
    f = tmp_path / "matrix.h5"
    h5io.write_10x_h5(f, m, bc, ids, names, genome="GRCh38")
    a = sc.read_10x_h5(str(f))
    assert a.shape == (3, 4)
    np.testing.assert_array_equal(np.asarray(a.X.todense()), m.toarray())
    assert a.obs_names.tolist() == bc
    assert a.var["gene_ids"].tolist() == ids
    assert a.var["genome"].tolist() == ["GRCh38"] * 4
    assert a.var_names.tolist() == names  # duplicates kept; scanpy's make_unique uses "-", R's "."


def test_read_counts_mtx_folder(tmp_path):
    import scipy.io

    m, bc, ids, names = _tiny_counts()
    scipy.io.mmwrite(str(tmp_path / "matrix.mtx"), m.T.tocoo())
    (tmp_path / "genes.tsv").write_text("".join(f"{i}\t{n}\n" for i, n in zip(ids, names)))
    (tmp_path / "barcodes.tsv").write_text("".join(b + "\n" for b in bc))
    m2, bc2, ids2, names2 = h5io.read_counts(tmp_path)
    assert (m2 != m).nnz == 0 and bc2 == bc and ids2 == ids and names2 == names


# ---------------------------------------------------------------- prep.make_unique


# Ground truth from R 4.x: print(make.unique(c(...)))
@pytest.mark.parametrize("names,expected", [
    (["A", "A", "B", "A", "A.1"], ["A", "A.2", "B", "A.3", "A.1"]),
    (["A", "A", "A.1", "A.1"], ["A", "A.2", "A.1", "A.1.1"]),
    (["A.1", "A", "A", "A.1", "A"], ["A.1", "A", "A.2", "A.1.1", "A.3"]),
    (["A", "A.1", "A", "A.1"], ["A", "A.1", "A.2", "A.1.1"]),
    (["X", "X.1", "X", "X.1.1", "X.1"], ["X", "X.1", "X.2", "X.1.1", "X.1.2"]),
    (["a", "b", "a", "a.2", "a"], ["a", "b", "a.1", "a.2", "a.3"]),
    (["g1", "g2"], ["g1", "g2"]),
    ([], []),
])
def test_make_unique_matches_r(names, expected):
    out = prep.make_unique(names)
    assert out == expected
    assert len(set(out)) == len(out)


# ---------------------------------------------------------------- prep.main


def _synth_h5(tmp_path, n_cells=300, n_genes=200):
    counts, bc, ids, names, labels = synth.generate(n_cells, n_genes, 3, seed=1)
    counts = counts.tolil()
    rare = n_genes - 50
    counts[:, rare] = 0
    counts[:2, rare] = 1  # one gene in only 2 cells -> dropped by min_cells=3
    counts[:20, 0] = 5000  # MT-SYN0 heavy in the first 20 cells -> fail mito QC
    counts = counts.tocsr()
    names[5] = names[6]  # a duplicate gene name for make_unique
    f = tmp_path / "in.h5"
    h5io.write_10x_h5(f, counts, bc, ids, names)
    lab = tmp_path / "labels.tsv"
    lab.write_text("barcode\tcell_type\n" + "".join(f"{b}\t{t}\n" for b, t in zip(bc, labels)))
    return f, lab, counts, bc, ids, names, labels


def _expected_qc(counts, names, min_cells, min_genes, max_genes, max_mt):
    """Seurat order, written independently: gene filter, then per-cell metrics on kept genes."""
    dense = counts.toarray()
    genes = (dense > 0).sum(axis=0) >= min_cells
    x = dense[:, genes]
    kept_names = [n for n, g in zip(names, genes) if g]
    nfeat = (x > 0).sum(axis=1)
    total = x.sum(axis=1)
    mt = np.array([n.startswith("MT-") for n in kept_names])
    pct = 100 * x[:, mt].sum(axis=1) / np.maximum(total, 1)
    cells = np.flatnonzero((nfeat > min_genes) & (nfeat < max_genes) & (pct < max_mt))
    return genes, cells


def test_prep_main(tmp_path, capsys):
    f, lab, counts, bc, ids, names, labels = _synth_h5(tmp_path)
    nfeat_raw = np.diff(counts.indptr)
    min_genes = int(np.quantile(nfeat_raw, 0.1))
    max_genes = int(np.quantile(nfeat_raw, 0.9))
    qc = dict(min_cells=3, min_genes=min_genes, max_genes=max_genes, max_mt=10.0)
    genes, cells = _expected_qc(counts, names, **qc)
    assert not genes[150] and genes[0] and genes.sum() < len(names)
    assert 50 < len(cells) < 300

    out = tmp_path / "data"
    rc = prep.main(["--input", str(f), "--out", str(out), "--sizes", "50,100,all,100000",
                    "--labels", str(lab), "--min-cells", "3", "--min-genes", str(min_genes),
                    "--max-genes", str(max_genes), "--max-mt", "10", "--seed", "3"])
    assert rc == 0
    assert "skip size 100000" in capsys.readouterr().out
    n_all = len(cells)
    kept_ids = [i for i, g in zip(ids, genes) if g]
    kept_names = prep.make_unique([n for n, g in zip(names, genes) if g])
    truth = dict(zip(bc, labels))

    sets = {}
    for n in (50, 100, n_all):
        d = out / f"n{n}"
        m, b, i, nm = h5io.read_counts(d / "matrix.h5")
        assert m.shape == (n, len(kept_ids))
        assert i == kept_ids and nm == kept_names
        assert len(set(nm)) == len(nm)
        rows = [bc.index(x) for x in b]
        assert rows == sorted(rows)  # original cell order kept
        assert set(rows) <= set(cells.tolist())
        assert (m != counts[rows][:, np.flatnonzero(genes)]).nnz == 0
        t = pd.read_csv(d / "cells.tsv", sep="\t", dtype=str)
        assert list(t.columns) == ["barcode", "cell_type"]
        assert t["barcode"].tolist() == b
        assert t["cell_type"].tolist() == [truth[x] for x in b]
        sets[n] = set(b)
    assert sets[50] < sets[100] < sets[n_all]
    assert sets[n_all] == {bc[c] for c in cells}
    assert not any(bc.index(x) < 20 for x in sets[n_all])  # mito-heavy cells removed

    meta = json.loads((out / "prep.json").read_text())
    assert meta["cells_in"] == 300 and meta["genes_in"] == 200
    assert meta["cells_qc"] == n_all and meta["genes_qc"] == int(genes.sum())
    assert meta["qc"] == {"min_cells": 3, "min_genes": min_genes, "max_genes": max_genes,
                          "max_mt": 10.0, "mt_prefix": "MT-"}
    assert meta["seed"] == 3 and meta["labels"] is True
    assert [s["cells"] for s in meta["sizes"]] == [50, 100, n_all]
    for s in meta["sizes"]:
        assert s["genes"] == int(genes.sum())
        assert Path(s["path"]) == out / f"n{s['cells']}" / "matrix.h5"
        assert s["nnz"] == h5io.read_counts(s["path"])[0].nnz


def test_prep_gene_filter_before_cell_filter(tmp_path):
    """A cell's feature count is taken after dropping rare genes, as in CreateSeuratObject."""
    # 6 cells x 5 genes. Genes 0-2 in all cells; gene 3 only in cell 0 (dropped by min_cells=2);
    # gene 4 in cells 1-5. Cell 0 has 4 raw features but only 3 after the gene filter.
    dense = np.zeros((6, 5), dtype=np.int32)
    dense[:, :3] = 1
    dense[0, 3] = 1
    dense[1:, 4] = 1
    f = tmp_path / "x.h5"
    bc = [f"C{i}" for i in range(6)]
    h5io.write_10x_h5(f, sp.csr_matrix(dense), bc, [f"G{i}" for i in range(5)], [f"G{i}" for i in range(5)])
    prep.main(["--input", str(f), "--out", str(tmp_path / "o"), "--min-cells", "2",
               "--min-genes", "3", "--max-genes", "10", "--max-mt", "100"])
    m, b, ids, _ = h5io.read_counts(tmp_path / "o" / "n5" / "matrix.h5")
    assert b == bc[1:]
    assert ids == ["G0", "G1", "G2", "G4"]


def test_prep_without_labels(tmp_path):
    f, _, *_ = _synth_h5(tmp_path, n_cells=80, n_genes=100)
    prep.main(["--input", str(f), "--out", str(tmp_path / "o"), "--sizes", "all",
               "--min-genes", "10", "--max-genes", "1000", "--max-mt", "100"])
    meta = json.loads((tmp_path / "o" / "prep.json").read_text())
    assert meta["labels"] is False and meta["cells_qc"] == 80
    t = pd.read_csv(tmp_path / "o" / "n80" / "cells.tsv", sep="\t", dtype=str)
    assert list(t.columns) == ["barcode"]


# ---------------------------------------------------------------- report.verdict


BIG = common.ADOPT_MIN_CELLS


@pytest.mark.parametrize("row,parity_ok,expected", [
    ({"engine": "seurat", "cells": BIG}, None, "baseline"),
    ({"engine": "seurat", "cells": 10}, False, "baseline"),
    ({"engine": "scfast-cpu", "cells": BIG, "speedup": 5.0}, None, "no parity data"),
    ({"engine": "scfast-cpu", "cells": BIG, "speedup": 5.0}, False, "fails parity"),
    ({"engine": "scfast-cpu", "cells": BIG - 1, "speedup": 5.0}, True,
     f"parity ok; too small to judge (< {BIG:,} cells)"),
    ({"engine": "scfast-cpu", "cells": BIG, "speedup": 1.5, "mem_ratio": 1.0}, True, "adopt"),
    ({"engine": "scfast-gpu", "cells": BIG, "speedup": 1.0, "mem_ratio": 2.0}, True, "adopt"),
    ({"engine": "scfast-cpu", "cells": BIG, "speedup": 1.49, "mem_ratio": 1.99}, True, "keep optional"),
    ({"engine": "scfast-cpu", "cells": BIG, "speedup": float("nan"), "mem_ratio": float("nan")}, True,
     "keep optional"),
    ({"engine": "scfast-cpu", "cells": BIG}, True, "keep optional"),
])
def test_verdict(row, parity_ok, expected):
    assert report.verdict(row, parity_ok) == expected
    assert report.verdict(pd.Series(row), parity_ok) == expected


# ---------------------------------------------------------------- hashing / multiplexed 10x files


def _write_multiplex_h5(path, n_cells=40, n_genes=30, n_ab=4):
    """A 10x v3 h5 whose features are genes followed by Antibody Capture features."""
    import h5py

    rng = np.random.default_rng(0)
    m = sp.csc_matrix(rng.poisson(1.0, size=(n_genes + n_ab, n_cells)).astype(np.int32))
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g["data"], g["indices"], g["indptr"] = m.data, m.indices.astype(np.int64), m.indptr.astype(np.int64)
        g["shape"] = np.array(m.shape, dtype=np.int32)
        g["barcodes"] = np.array([f"BC{i}-1".encode() for i in range(n_cells)])
        ft = g.create_group("features")
        ft["id"] = np.array([f"ENSG{i}".encode() for i in range(n_genes)] + [f"AB{i}".encode() for i in range(n_ab)])
        ft["name"] = np.array([f"G{i}".encode() for i in range(n_genes)] + [f"Hashtag{i}".encode() for i in range(n_ab)])
        ft["feature_type"] = np.array([b"Gene Expression"] * n_genes + [b"Antibody Capture"] * n_ab)
    return m


def test_read_counts_drops_antibody_features(tmp_path, capsys):
    m = _write_multiplex_h5(tmp_path / "hash.h5")
    counts, barcodes, ids, names = h5io.read_counts(tmp_path / "hash.h5")
    assert counts.shape == (40, 30) and len(ids) == len(names) == 30
    assert not any(n.startswith("Hashtag") for n in names)
    assert (counts.toarray() == m.toarray()[:30].T).all()  # genes kept, in order, cells x genes
    assert "dropping 4 others (Antibody Capture)" in capsys.readouterr().out


def test_pipeline_steps_writes_every_step(tmp_path):
    pytest.importorskip("scanpy")
    pytest.importorskip("scfast")
    pytest.importorskip("skmisc", reason="scanpy's seurat_v3 HVG flavour needs scikit-misc")
    import pipeline_steps

    counts, barcodes, ids, names, _ = synth.generate(1200, 600, 4, seed=1)
    h5 = tmp_path / "d.h5"
    h5io.write_10x_h5(h5, counts, barcodes, ids, names)
    out = tmp_path / "out"
    rc = pipeline_steps.main(["--input", str(h5), "--out", str(out), "--max-mt", "100", "--umap-epochs", "20"])
    assert rc == 0
    for step in ["01_qc", "02_normalize", "03_hvg", "04_pca", "05_neighbors", "06_cluster", "07_umap"]:
        assert any((out / step).iterdir()), step
    clusters = pd.read_csv(out / "06_cluster" / "clusters.csv")
    assert clusters["cluster"].nunique() >= 3
    assert (out / "summary.md").exists() and (out / "final.h5ad").exists()
