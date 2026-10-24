#!/usr/bin/env python3
"""Generate the deterministic ``pbmc_like`` fixture used by the exchange tests.

The fixture is a small synthetic, PBMC-like counts matrix written in 10x
MatrixMarket form, plus ``expected.json`` with the values that every
implementation of docs/exchange-spec.md must reproduce:

* ``counts_checksum``  - spec section 6, computed here with an independent
  implementation (numpy + struct + hashlib; cellwarden is deliberately NOT imported).
* ``cells_sha256`` / ``genes_sha256_*`` - spec section 4.

Files written to ``tests/fixtures/pbmc_like/`` (genes x cells, like 10x):

    matrix.mtx     MatrixMarket coordinate integer general, genes x cells
    features.tsv   gene_id <TAB> symbol <TAB> feature_type   (no header)
    barcodes.tsv   one barcode per line                       (no header)
    cells.tsv      barcode, sample, cell_type                 (header)
    expected.json  checksums, shape, level orders, marker genes

Usage:
    python tests/fixtures/make_fixture.py            # (re)generate
    python tests/fixtures/make_fixture.py --check    # regenerate in a temp dir
                                                     # and compare byte-for-byte
    python tests/fixtures/make_fixture.py --verify   # recompute expected.json
                                                     # values from files on disk

The committed files are the source of truth for the tests; ``--check`` proves
the generator still reproduces them.
"""
from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

SEED = 20261017
N_CELLS = 500
N_GENES = 800
SAMPLES = ["donor_B", "donor_A"]          # deliberately non-alphabetical level order
CELL_TYPES = ["T", "B", "Mono", "NK"]       # deliberately non-alphabetical level order
GROUP_FRACTIONS = [0.40, 0.20, 0.25, 0.15]
OUT_DIR = Path(__file__).resolve().parent / "pbmc_like"

MARKERS = {
    "T": ["CD3D", "CD3E", "IL7R", "CD2", "LTB", "TRAC", "CD3G", "LEF1"],
    "B": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "HLA-DQA1", "TCL1A", "VPREB3"],
    "Mono": ["CD14", "LYZ", "S100A8", "S100A9", "FCN1", "CST3", "LST1", "TYROBP"],
    "NK": ["NKG7", "GNLY", "GZMB", "PRF1", "KLRD1", "GZMA", "CTSW", "FCGR3A"],
}
MT_GENES = ["MT-ND1", "MT-ND2", "MT-CO1", "MT-CO2", "MT-ATP8", "MT-ATP6", "MT-CO3",
            "MT-ND3", "MT-ND4L", "MT-ND4", "MT-ND5", "MT-ND6", "MT-CYB"]
# Underscores: Seurat rewrites these to dashes, so gene_symbol_original matters.
UNDERSCORE_GENES = ["LINC_00486", "RP11_34P13.7", "AC_093642.5", "TRBV20_1",
                    "IGHV3_23", "SNORD3A_HG", "CTD_2588J6.1", "HIST1_H2BC",
                    "KIR2DL_1", "U6_snRNA"]
# Dashes already (valid Seurat names) and look-alike-but-unique names.
DASH_GENES = ["HLA-DRA", "HLA-DRB1", "HLA-A", "HLA-B", "HLA-C", "RP11-34P13.3",
              "TBCE-AS1", "HSPA14-AS1", "LINC01238-AS1", "AP000254.8"]
LOOKALIKE_GENES = ["TBCE", "TBCE.1", "HSPA14", "HSPA14.1", "LINC01238", "LINC01238.1",
                   "HLA-A1", "Y_RNA", "Y_RNA.1", "Metazoa_SRP", "Metazoa_SRP.1"]
HOUSEKEEPING = ["ACTB", "GAPDH", "B2M", "MALAT1", "TMSB4X", "EEF1A1", "FTL", "FTH1"]


# --------------------------------------------------------------------------- data
def gene_symbols() -> list[str]:
    fixed = (HOUSEKEEPING + [g for v in MARKERS.values() for g in v] + MT_GENES
             + UNDERSCORE_GENES + DASH_GENES + LOOKALIKE_GENES)
    ribo = [f"RPL{i}" for i in range(3, 33)] + [f"RPS{i}" for i in range(2, 28)]
    fixed += ribo
    filler: list[str] = []
    i = 0
    while len(fixed) + len(filler) < N_GENES:
        stem = ("ZNF", "LOC", "SLC25A", "FAM", "KIAA", "TMEM", "C1orf", "CCDC")[i % 8]
        filler.append(f"{stem}{100 + i}")
        i += 1
    names = fixed + filler
    assert len(names) == N_GENES
    assert len(set(names)) == N_GENES, "symbols must be unique"
    # After Seurat's '_' -> '-' rewrite they must STILL be unique.
    assert len({n.replace("_", "-") for n in names}) == N_GENES, "sanitised collision"
    return names


def gene_ids(n: int) -> list[str]:
    return [f"ENSG{(100000 + 37 * i):011d}" for i in range(n)]


def barcodes(rng: np.random.Generator, sample_idx: np.ndarray) -> list[str]:
    alphabet = np.array(list("ACGT"))
    seen: set[str] = set()
    out: list[str] = []
    for s in sample_idx:
        while True:
            core = "".join(alphabet[rng.integers(0, 4, 16)])
            bc = f"{core}-{s + 1}"
            if bc not in seen:
                seen.add(bc)
                out.append(bc)
                break
    return out


def simulate():
    rng = np.random.default_rng(SEED)
    symbols = gene_symbols()
    ids = gene_ids(N_GENES)

    sizes = np.round(np.array(GROUP_FRACTIONS) * N_CELLS).astype(int)
    sizes[-1] = N_CELLS - sizes[:-1].sum()
    group = np.repeat(np.arange(len(CELL_TYPES)), sizes)
    rng.shuffle(group)
    sample = rng.integers(0, len(SAMPLES), N_CELLS)

    # Baseline mean expression (per-cell expected count at a 2,000 UMI library).
    base = rng.lognormal(mean=-2.6, sigma=1.3, size=N_GENES)
    sym_idx = {s: i for i, s in enumerate(symbols)}
    for g in HOUSEKEEPING:
        base[sym_idx[g]] = rng.uniform(8, 25)
    for g in MT_GENES:
        base[sym_idx[g]] = rng.uniform(1.0, 6.0)
    for g in symbols:
        if g.startswith(("RPL", "RPS")):
            base[sym_idx[g]] = rng.uniform(1.5, 6.0)
    for v in MARKERS.values():
        for g in v:
            base[sym_idx[g]] = rng.uniform(0.05, 0.2)

    mu = np.tile(base, (N_CELLS, 1))                    # cells x genes
    for k, ct in enumerate(CELL_TYPES):
        cols = [sym_idx[g] for g in MARKERS[ct]]
        mu[np.ix_(group == k, cols)] *= rng.uniform(15, 40, len(cols))
    # Mild batch effect on a handful of filler genes.
    batch_cols = rng.choice(np.arange(200, N_GENES), 20, replace=False)
    mu[np.ix_(sample == 1, batch_cols)] *= 2.0

    lib = rng.lognormal(mean=0.0, sigma=0.35, size=N_CELLS)
    mu *= lib[:, None]
    # Negative binomial via gamma-Poisson (dispersion theta = 5).
    theta = 5.0
    lam = rng.gamma(shape=theta, scale=mu / theta)
    counts = rng.poisson(lam).astype(np.int64)          # cells x genes
    # Every cell and gene must carry at least one count.
    zero_genes = np.where(counts.sum(0) == 0)[0]
    for j in zero_genes:
        counts[rng.integers(0, N_CELLS), j] = 1
    assert (counts.sum(1) > 0).all()

    bcs = barcodes(rng, sample)
    return counts, symbols, ids, bcs, sample, group


# ------------------------------------------------------------ independent checksum
def counts_checksum_from_coo(n_cells: int, n_genes: int, cell: np.ndarray,
                             gene: np.ndarray, value: np.ndarray) -> str:
    """Spec section 6 on the cells x genes CSR form, built from COO triplets.

    Independent of cellwarden and of scipy: sorts by (row, col), drops nothing
    (explicit zeros are not expected in a counts file), builds indptr by
    bincount, then hashes shape/indptr/indices/data as LE int64/int64/int64/f64.
    """
    cell = np.asarray(cell, dtype=np.int64)
    gene = np.asarray(gene, dtype=np.int64)
    value = np.asarray(value, dtype=np.float64)
    order = np.lexsort((gene, cell))                   # primary: cell, secondary: gene
    cell, gene, value = cell[order], gene[order], value[order]
    indptr = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(np.bincount(cell, minlength=n_cells), out=indptr[1:])
    h = hashlib.sha256()
    h.update(struct.pack("<qq", n_cells, n_genes))
    h.update(indptr.astype("<i8").tobytes())
    h.update(gene.astype("<i8").tobytes())
    h.update(value.astype("<f8").tobytes())
    return h.hexdigest()


def names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------- writing
def write_fixture(out: Path) -> dict:
    counts, symbols, ids, bcs, sample, group = simulate()
    out.mkdir(parents=True, exist_ok=True)

    cell_i, gene_j = np.nonzero(counts)                 # row-major: sorted by cell
    vals = counts[cell_i, gene_j]
    # MatrixMarket is genes x cells; write sorted by column (cell) then row (gene).
    with open(out / "matrix.mtx", "w", newline="\n") as fh:
        fh.write("%%MatrixMarket matrix coordinate integer general\n")
        fh.write("% cellwarden pbmc_like synthetic fixture; generated by make_fixture.py\n")
        fh.write(f"{N_GENES} {N_CELLS} {len(vals)}\n")
        fh.write("".join(f"{g + 1} {c + 1} {v}\n" for c, g, v in zip(cell_i, gene_j, vals)))
    with open(out / "features.tsv", "w", newline="\n") as fh:
        fh.write("".join(f"{i}\t{s}\tGene Expression\n" for i, s in zip(ids, symbols)))
    with open(out / "barcodes.tsv", "w", newline="\n") as fh:
        fh.write("".join(f"{b}\n" for b in bcs))
    with open(out / "cells.tsv", "w", newline="\n") as fh:
        fh.write("barcode\tsample\tcell_type\n")
        fh.write("".join(f"{b}\t{SAMPLES[s]}\t{CELL_TYPES[g]}\n"
                         for b, s, g in zip(bcs, sample, group)))

    expected = compute_expected(out)
    with open(out / "expected.json", "w", newline="\n") as fh:
        json.dump(expected, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return expected


# ------------------------------------------------------------------------ reading
def read_fixture(d: Path):
    """Parse the fixture from disk without scipy/cellwarden."""
    with open(d / "matrix.mtx") as fh:
        header = fh.readline()
        assert header.startswith("%%MatrixMarket matrix coordinate integer general")
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        n_genes, n_cells, nnz = map(int, line.split())
        trip = np.loadtxt(fh, dtype=np.int64, ndmin=2)
    assert trip.shape == (nnz, 3)
    feats = [ln.rstrip("\n").split("\t") for ln in open(d / "features.tsv")]
    bcs = [ln.rstrip("\n") for ln in open(d / "barcodes.tsv")]
    assert len(feats) == n_genes and len(bcs) == n_cells
    return n_genes, n_cells, trip, feats, bcs


def compute_expected(d: Path) -> dict:
    n_genes, n_cells, trip, feats, bcs = read_fixture(d)
    gene = trip[:, 0] - 1
    cell = trip[:, 1] - 1
    symbols = [f[1] for f in feats]
    seurat_names = [s.replace("_", "-") for s in symbols]
    checksum = counts_checksum_from_coo(n_cells, n_genes, cell, gene, trip[:, 2])
    # Self-test the checksum against a dense reference build (catches sorting bugs).
    dense = np.zeros((n_cells, n_genes), dtype=np.float64)
    dense[cell, gene] = trip[:, 2]
    r, c = np.nonzero(dense)
    assert counts_checksum_from_coo(n_cells, n_genes, r, c, dense[r, c]) == checksum

    meta = [ln.rstrip("\n").split("\t") for ln in open(d / "cells.tsv")][1:]
    hvg_hint = sorted({g for v in MARKERS.values() for g in v})
    return {
        "fixture": "pbmc_like",
        "seed": SEED,
        "generator": "tests/fixtures/make_fixture.py",
        "n_cells": n_cells,
        "n_genes": n_genes,
        "nnz": int(trip.shape[0]),
        "total_counts": int(trip[:, 2].sum()),
        "max_count": int(trip[:, 2].max()),
        "counts_checksum": checksum,
        "cells_sha256": names_sha256(bcs),
        "genes_sha256_original": names_sha256(symbols),
        "genes_sha256_seurat": names_sha256(seurat_names),
        "underscore_genes": [s for s in symbols if "_" in s],
        "mt_genes": [s for s in symbols if s.startswith("MT-")],
        "marker_genes": {k: v for k, v in MARKERS.items()},
        "marker_genes_flat": hvg_hint,
        "obs_levels": {"sample": SAMPLES, "cell_type": CELL_TYPES},
        "obs_counts": {
            "sample": {s: sum(m[1] == s for m in meta) for s in SAMPLES},
            "cell_type": {t: sum(m[2] == t for m in meta) for t in CELL_TYPES},
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true",
                   help="regenerate into a temp dir and compare with committed files")
    g.add_argument("--verify", action="store_true",
                   help="recompute expected.json values from the files on disk")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()

    files = ["matrix.mtx", "features.tsv", "barcodes.tsv", "cells.tsv", "expected.json"]
    if a.check:
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(Path(tmp))
            bad = [f for f in files if not filecmp.cmp(Path(tmp) / f, a.out / f, shallow=False)]
        if bad:
            print(f"FAIL: regenerated fixture differs in: {', '.join(bad)}", file=sys.stderr)
            return 1
        print("OK: fixture reproduces byte-for-byte")
        return 0
    if a.verify:
        on_disk = json.loads((a.out / "expected.json").read_text())
        fresh = compute_expected(a.out)
        diff = [k for k in fresh if fresh[k] != on_disk.get(k)]
        if diff:
            print(f"FAIL: expected.json disagrees with data files on: {diff}", file=sys.stderr)
            return 1
        print(f"OK: expected.json matches data (counts_checksum={fresh['counts_checksum']})")
        return 0
    exp = write_fixture(a.out)
    print(json.dumps({k: exp[k] for k in ("n_cells", "n_genes", "nnz", "counts_checksum")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
