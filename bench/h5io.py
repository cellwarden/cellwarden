"""Read any supported input into cells x genes counts, and write 10x-format .h5 files.

The harness converts every input once (bench/prep.py) into Cell Ranger v3-style
``.h5`` files, so Seurat (``Read10X_h5``) and Scanpy (``read_10x_h5``) start from
byte-identical cells, genes and counts.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def _first(d: Path, names: list[str]) -> Path | None:
    for n in names:
        if (d / n).exists():
            return d / n
    return None


def read_counts(path: str | Path):
    """Return (counts CSR cells x genes, barcodes, gene_ids, gene_names)."""
    p = Path(path)
    if p.is_dir():
        mtx = _first(p, ["matrix.mtx.gz", "matrix.mtx"])
        feat = _first(p, ["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"])
        bc = _first(p, ["barcodes.tsv.gz", "barcodes.tsv"])
        if not (mtx and feat and bc):
            raise FileNotFoundError(f"{p} is not a 10x matrix folder (matrix.mtx, features/genes.tsv, barcodes.tsv)")
        m = scipy.io.mmread(str(mtx)).T.tocsr()  # genes x cells -> cells x genes
        f = pd.read_csv(feat, sep="\t", header=None, dtype=str)
        ids = f[0].tolist()
        names = (f[1] if f.shape[1] > 1 else f[0]).tolist()
        barcodes = pd.read_csv(bc, sep="\t", header=None, dtype=str)[0].tolist()
    elif p.suffix == ".h5":
        m, barcodes, ids, names = _read_10x_h5(p)
    elif p.suffix == ".h5ad":
        import anndata as ad

        a = ad.read_h5ad(p)
        if "counts" in a.layers:
            m = a.layers["counts"]
        elif a.raw is not None:
            m = a.raw.X
        else:
            m = a.X
        m = sp.csr_matrix(m)
        if m.nnz and not np.allclose(m.data[: min(m.nnz, 100_000)] % 1, 0):
            raise ValueError(f"{p}: matrix does not look like raw counts (non-integer values)")
        var = a.raw.var if (a.raw is not None and "counts" not in a.layers) else a.var
        barcodes = a.obs_names.tolist()
        ids = (var["gene_ids"] if "gene_ids" in var else var.index).astype(str).tolist()
        names = var.index.astype(str).tolist()
    else:
        raise ValueError(f"unsupported input {p} (10x folder, .h5 or .h5ad)")
    m = sp.csr_matrix(m)
    m.sum_duplicates()
    if m.shape != (len(barcodes), len(ids)):
        raise ValueError(f"shape {m.shape} does not match {len(barcodes)} barcodes x {len(ids)} genes")
    return m, barcodes, ids, names


def _read_10x_h5(path: Path):
    with h5py.File(path, "r") as f:
        g = f["matrix"] if "matrix" in f else f[list(f.keys())[0]]
        shape = tuple(g["shape"][()])  # genes x cells
        csc = sp.csc_matrix((g["data"][()], g["indices"][()], g["indptr"][()]), shape=shape)
        barcodes = [b.decode() for b in g["barcodes"][()]]
        keep = None
        if "features" in g:
            ids = [x.decode() for x in g["features/id"][()]]
            names = [x.decode() for x in g["features/name"][()]]
            # Multiplexed / CITE-seq / hashing files carry Antibody Capture, Multiplexing Capture
            # and CRISPR features next to the genes; only genes belong in the RNA workflow.
            if "feature_type" in g["features"]:
                ft = np.array([x.decode() for x in g["features/feature_type"][()]])
                if (ft != "Gene Expression").any():
                    keep = ft == "Gene Expression"
                    kinds = sorted(set(ft[~keep]))
                    print(f"{path.name}: keeping {int(keep.sum())} Gene Expression features, dropping "
                          f"{int((~keep).sum())} others ({', '.join(kinds)})")
        else:  # Cell Ranger v2
            ids = [x.decode() for x in g["genes"][()]]
            names = [x.decode() for x in g["gene_names"][()]]
    m = csc.T.tocsr()
    if keep is not None:
        m = m[:, np.flatnonzero(keep)]
        ids = [x for x, k in zip(ids, keep) if k]
        names = [x for x, k in zip(names, keep) if k]
    return m, barcodes, ids, names


def write_10x_h5(path: str | Path, counts_cells_x_genes, barcodes, gene_ids, gene_names,
                 genome: str = "unknown") -> None:
    """Write a Cell Ranger v3-format .h5 (genes x cells CSC, int32 counts)."""
    m = sp.csr_matrix(counts_cells_x_genes)
    gxc = m.T.tocsc()  # genes x cells CSC == cells x genes CSR arrays
    gxc.sort_indices()
    n_genes = len(gene_ids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("matrix")
        g.create_dataset("data", data=gxc.data.astype(np.int32), compression="gzip", compression_opts=1)
        g.create_dataset("indices", data=gxc.indices.astype(np.int64), compression="gzip", compression_opts=1)
        g.create_dataset("indptr", data=gxc.indptr.astype(np.int64))
        g.create_dataset("shape", data=np.array(gxc.shape, dtype=np.int32))
        g.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        ft = g.create_group("features")
        ft.create_dataset("id", data=np.array(gene_ids, dtype="S"))
        ft.create_dataset("name", data=np.array(gene_names, dtype="S"))
        ft.create_dataset("feature_type", data=np.array(["Gene Expression"] * n_genes, dtype="S"))
        ft.create_dataset("genome", data=np.array([genome] * n_genes, dtype="S"))
        ft.create_dataset("_all_tag_keys", data=np.array(["genome"], dtype="S"))
