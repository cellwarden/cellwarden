"""Shared QC and subsampling, so every engine starts from identical cells and genes.

Applies the QC from seurat_pbmc3k.R once, in Seurat's order (genes in at least
``--min-cells`` cells, then cells with at least ``--min-genes`` genes, then the
gene-count and mitochondrial thresholds), then writes one 10x ``.h5`` per
requested size. Smaller sizes are nested subsets of larger ones.

    python bench/prep.py --input pbmc3k/filtered_gene_bc_matrices/hg19 \
        --out bench/results/run1/data --sizes 1000,all

Output, per size: ``<out>/n<size>/matrix.h5`` and ``cells.tsv`` (barcode, and
``cell_type`` when ``--labels`` is given), plus ``<out>/prep.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h5io import read_counts, write_10x_h5  # noqa: E402


def make_unique(names: list[str]) -> list[str]:
    """R's make.unique: second 'X' becomes 'X.1', and so on."""
    seen: dict[str, int] = {}
    used = set(names)
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
            continue
        while True:
            seen[n] += 1
            cand = f"{n}.{seen[n]}"
            if cand not in used:
                used.add(cand)
                out.append(cand)
                break
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", required=True, help="10x folder, 10x .h5 or .h5ad with raw counts")
    p.add_argument("--out", required=True)
    p.add_argument("--sizes", default="all", help="comma-separated cell counts, or 'all'")
    p.add_argument("--labels", default=None, help="optional TSV with barcode and cell_type columns")
    p.add_argument("--min-cells", type=int, default=3)
    p.add_argument("--min-genes", type=int, default=200)
    p.add_argument("--max-genes", type=float, default=2500)
    p.add_argument("--max-mt", type=float, default=5.0)
    p.add_argument("--mt-prefix", default="MT-")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    counts, barcodes, ids, names = read_counts(a.input)
    n0, g0 = counts.shape
    # Seurat CreateSeuratObject(min.cells, min.features)
    counts.eliminate_zeros()  # stored entries are then exactly the detected genes, without a bool copy
    genes_keep = np.flatnonzero(counts.getnnz(axis=0) >= a.min_cells)
    counts = counts[:, genes_keep]
    ids = [ids[i] for i in genes_keep]
    names = make_unique([names[i] for i in genes_keep])
    n_genes = np.diff(counts.indptr)
    total = np.asarray(counts.sum(axis=1)).ravel()
    mt = np.array([n.upper().startswith(a.mt_prefix.upper()) for n in names])
    mt_counts = np.asarray(counts[:, np.flatnonzero(mt)].sum(axis=1)).ravel() if mt.any() else np.zeros(len(total))
    pct_mt = np.divide(100 * mt_counts, total, out=np.zeros_like(total, dtype=float), where=total > 0)
    # seurat_pbmc3k.R: min.features = 200 at creation, then subset nFeature > 200 & < 2500 & mt < 5
    q = [0.5, 0.9, 0.99]
    print(f"genes per cell p50/p90/p99: {np.quantile(n_genes, q).round().tolist()}; "
          f"% mitochondrial p50/p90/p99: {np.quantile(pct_mt, q).round(2).tolist()}")
    keep = (n_genes > a.min_genes) & (n_genes < a.max_genes) & (pct_mt < a.max_mt)
    cells = np.flatnonzero(keep)
    print(f"QC: {n0} -> {len(cells)} cells; {g0} -> {len(ids)} genes ({int(mt.sum())} mitochondrial)")

    labels = None
    if a.labels:
        lab = pd.read_csv(a.labels, sep="\t", dtype=str)
        col = "cell_type" if "cell_type" in lab.columns else lab.columns[1]
        labels = dict(zip(lab.iloc[:, 0], lab[col]))

    sizes = []
    for s in a.sizes.split(","):
        s = s.strip()
        n = len(cells) if s == "all" else int(s)
        if n > len(cells):
            print(f"skip size {n}: only {len(cells)} cells pass QC")
            continue
        sizes.append(n)
    rng = np.random.default_rng(a.seed)
    order = cells[rng.permutation(len(cells))]  # nested subsets: prefix of one permutation

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for n in sorted(set(sizes)):
        sel = np.sort(order[:n])  # keep original cell order inside each file
        d = out / f"n{n}"
        sub = counts[sel]
        write_10x_h5(d / "matrix.h5", sub, [barcodes[i] for i in sel], ids, names)
        tsv = pd.DataFrame({"barcode": [barcodes[i] for i in sel]})
        if labels is not None:
            tsv["cell_type"] = [labels.get(b, "NA") for b in tsv["barcode"]]
        tsv.to_csv(d / "cells.tsv", sep="\t", index=False)
        written.append({"cells": n, "genes": len(ids), "nnz": int(sub.nnz), "path": str(d / "matrix.h5")})
        print(f"wrote {d}/matrix.h5: {n} cells x {len(ids)} genes")

    (out / "prep.json").write_text(json.dumps({
        "input": str(a.input), "cells_in": n0, "genes_in": g0, "cells_qc": int(len(cells)),
        "genes_qc": len(ids), "qc": {"min_cells": a.min_cells, "min_genes": a.min_genes,
                                    "max_genes": a.max_genes, "max_mt": a.max_mt,
                                    "mt_prefix": a.mt_prefix},
        "seed": a.seed, "sizes": written, "labels": bool(labels),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
