#!/usr/bin/env python3
"""Build the two-sample, QC-stress input for the S0-S2 end-to-end test.

Starts from ``tests/fixtures/pbmc_like`` (500 cells x 800 genes, two samples,
four planted cell types) and writes ``tests/fixtures/stage_inputs/``:

    donor_A/                10x_mtx directory (Cell Ranger v3 layout, gzipped):
                              matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz
    donor_B.h5ad            plain AnnData (NOT an exchange file): raw counts in
                            layers["counts"] (int32), log-normalised floats in X,
                            gene order permuted and a few genes missing
    samples.csv             sample sheet with paths relative to this directory
                            (scpipe resolves sample paths against the folder
                            holding config.yaml, so this sheet suits a config
                            placed here; the e2e test uses
                            tests/integration/stage_configs/samples.csv)
    truth.tsv               cell_id (<sample_id>_<barcode>), sample, cell_type,
                            injected (none | lowq | doublet | platelet)
    manifest.json           counts per sample and per injected population,
                            gene bookkeeping, and SHA-256 of every file

What is injected, per sample (all deterministic, seeded):

* ``lowq``     ~5 % low-quality cells: a real cell thinned to ~25 % of its UMIs
               (few genes), then mitochondrial counts added to reach 35-60 %
               MT. They fail the hard mito limit, so rules must exclude them.
* ``doublet``  ~4 % synthetic doublets: the sum of two real cells of different
               planted types from the same sample.
* ``platelet`` 12 platelet-like cells: small libraries (~60-90 genes, ~150 UMIs)
               dominated by PPBP / PF4 / GNG11. They pass the hard limits but
               fall under the adaptive lower bounds, so S1c should turn them
               into ``review`` with reason ``PROTECTED_POP:Platelet``.

Gene rows added to both samples (absent from pbmc_like): PPBP, PF4, GNG11
(platelet markers), HBB, HBA1, HBA2 (haemoglobin, ambient level only) and the
look-alikes HBP1, HBEGF, which are NOT haemoglobin genes and must not count
towards pct_hb.

Usage:
    python tests/fixtures/make_stage_inputs.py           # (re)generate
    python tests/fixtures/make_stage_inputs.py --check   # regenerate in a temp
                                                         # dir and compare
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "pbmc_like"
OUT_DIR = HERE / "stage_inputs"
SEED = 20261018

LOWQ_FRAC = 0.05
DOUBLET_FRAC = 0.04
N_PLATELET = 12

# (symbol, Ensembl ID). IDs are the real GRCh38 ones; the pbmc_like IDs are
# synthetic (ENSG00000100000 + 37*i) and are checked not to collide.
ADDED_GENES = [
    ("PPBP", "ENSG00000163736"),
    ("PF4", "ENSG00000163737"),
    ("GNG11", "ENSG00000127920"),
    ("HBB", "ENSG00000244734"),
    ("HBA1", "ENSG00000206172"),
    ("HBA2", "ENSG00000188536"),
    ("HBP1", "ENSG00000105856"),
    ("HBEGF", "ENSG00000113070"),
]
PLATELET_MARKERS = {"PPBP": 26.0, "PF4": 16.0, "GNG11": 7.0}   # mean counts per platelet
AMBIENT = {"PPBP": 0.01, "PF4": 0.01, "GNG11": 0.02, "HBB": 0.03, "HBA1": 0.02,
           "HBA2": 0.02, "HBP1": 0.25, "HBEGF": 0.15}           # mean counts per other cell
N_DROPPED_FROM_B = 4          # low-expressed filler genes missing from donor_B
SAMPLE_ORDER = ["donor_A", "donor_B"]
FILES = ["donor_A/matrix.mtx.gz", "donor_A/features.tsv.gz", "donor_A/barcodes.tsv.gz",
         "donor_B.h5ad", "samples.csv", "truth.tsv", "manifest.json"]


# ----------------------------------------------------------------------- reading
def read_source(d: Path):
    with open(d / "matrix.mtx") as fh:
        assert fh.readline().startswith("%%MatrixMarket matrix coordinate integer general")
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        n_genes, n_cells, _ = map(int, line.split())
        trip = np.loadtxt(fh, dtype=np.int64, ndmin=2)
    counts = np.zeros((n_cells, n_genes), dtype=np.int64)          # cells x genes
    counts[trip[:, 1] - 1, trip[:, 0] - 1] = trip[:, 2]
    feats = [ln.rstrip("\n").split("\t") for ln in open(d / "features.tsv")]
    meta = [ln.rstrip("\n").split("\t") for ln in open(d / "cells.tsv")][1:]
    return counts, [f[0] for f in feats], [f[1] for f in feats], meta


# ---------------------------------------------------------------------- building
def new_barcodes(rng, n: int, suffix: str, taken: set[str]) -> list[str]:
    alphabet = np.array(list("ACGT"))
    out = []
    while len(out) < n:
        bc = "".join(alphabet[rng.integers(0, 4, 16)]) + suffix
        if bc not in taken:
            taken.add(bc)
            out.append(bc)
    return out


def thin(rng, row: np.ndarray, p: float) -> np.ndarray:
    return rng.binomial(row, p)


def build():
    rng = np.random.default_rng(SEED)
    counts, ids, symbols, meta = read_source(SRC_DIR)
    taken_ids = set(ids)
    assert not taken_ids & {g for _, g in ADDED_GENES}, "added gene ID collides"
    assert not set(symbols) & {s for s, _ in ADDED_GENES}, "added gene symbol already present"

    symbols = symbols + [s for s, _ in ADDED_GENES]
    ids = ids + [g for _, g in ADDED_GENES]
    n_genes = len(symbols)
    sym_idx = {s: i for i, s in enumerate(symbols)}
    mt_cols = np.array([i for i, s in enumerate(symbols) if s.startswith("MT-")])
    added_cols = np.array([sym_idx[s] for s, _ in ADDED_GENES])
    with open(SRC_DIR / "expected.json") as fh:
        markers = json.load(fh)["marker_genes"]
    marker_cols = np.array(sorted({sym_idx[g] for v in markers.values() for g in v}))

    # Original cells gain the new genes at ambient level.
    orig = np.zeros((counts.shape[0], n_genes), dtype=np.int64)
    orig[:, : counts.shape[1]] = counts
    for s, lam in AMBIENT.items():
        orig[:, sym_idx[s]] = rng.poisson(lam, counts.shape[0])

    # Baseline profile for platelets: the average cell with type markers, MT
    # and the added genes removed.
    base = orig.mean(0).astype(np.float64)
    base[marker_cols] = 0
    base[added_cols] = 0
    base[mt_cols] *= 0.4
    base /= base.sum()

    barcodes_all = [m[0] for m in meta]
    taken_bc = set(barcodes_all)
    samples: dict[str, dict] = {}
    for sample_id in SAMPLE_ORDER:
        idx = np.array([i for i, m in enumerate(meta) if m[1] == sample_id])
        suffix = barcodes_all[idx[0]][-2:]                     # "-1" / "-2"
        types = np.array([meta[i][2] for i in idx])
        rows, bcs, ctype, inj = [orig[idx]], [barcodes_all[i] for i in idx], list(types), ["none"] * len(idx)
        n = len(idx)

        n_lowq = int(round(LOWQ_FRAC * n))
        parents = rng.choice(len(idx), n_lowq, replace=False)
        lowq = np.zeros((n_lowq, n_genes), dtype=np.int64)
        for k, p in enumerate(parents):
            r = thin(rng, orig[idx[p]], rng.uniform(0.18, 0.30))
            r[mt_cols] = 0
            target_mt = rng.uniform(0.35, 0.60)
            n_mt = int(round(r.sum() * target_mt / (1 - target_mt)))
            w = orig[idx[p], mt_cols] + 1.0
            r[mt_cols] = rng.multinomial(n_mt, w / w.sum())
            lowq[k] = r
        rows.append(lowq)
        bcs += new_barcodes(rng, n_lowq, suffix, taken_bc)
        ctype += list(types[parents])
        inj += ["lowq"] * n_lowq

        n_dbl = int(round(DOUBLET_FRAC * n))
        dbl = np.zeros((n_dbl, n_genes), dtype=np.int64)
        dbl_types = []
        for k in range(n_dbl):
            a = rng.integers(0, n)
            b = rng.choice(np.where(types != types[a])[0])
            dbl[k] = orig[idx[a]] + orig[idx[b]]
            dbl_types.append("+".join(sorted((types[a], types[b]))))
        rows.append(dbl)
        bcs += new_barcodes(rng, n_dbl, suffix, taken_bc)
        ctype += dbl_types
        inj += ["doublet"] * n_dbl

        plt = np.zeros((N_PLATELET, n_genes), dtype=np.int64)
        for k in range(N_PLATELET):
            lib = rng.uniform(70, 110)
            r = rng.poisson(rng.gamma(4.0, base * lib / 4.0))
            for s, mu in PLATELET_MARKERS.items():
                r[sym_idx[s]] = rng.poisson(rng.gamma(5.0, mu / 5.0)) + 1
            plt[k] = r
        rows.append(plt)
        bcs += new_barcodes(rng, N_PLATELET, suffix, taken_bc)
        ctype += ["Platelet"] * N_PLATELET
        inj += ["platelet"] * N_PLATELET

        mat = np.vstack(rows)
        order = rng.permutation(mat.shape[0])                  # interleave injected cells
        samples[sample_id] = {
            "counts": mat[order],
            "barcodes": [bcs[i] for i in order],
            "cell_type": [ctype[i] for i in order],
            "injected": [inj[i] for i in order],
        }

    # donor_B: drop a few low-expressed filler genes and permute the gene order.
    filler = [i for i, s in enumerate(symbols)
              if s.startswith(("ZNF", "LOC", "SLC25A", "FAM", "KIAA", "TMEM", "C1orf", "CCDC"))]
    tot = samples["donor_B"]["counts"].sum(0)
    filler.sort(key=lambda i: (tot[i], i))
    dropped = sorted(filler[:N_DROPPED_FROM_B])
    keep_b = np.array([i for i in range(n_genes) if i not in set(dropped)])
    keep_b = keep_b[rng.permutation(len(keep_b))]
    return samples, symbols, ids, dropped, keep_b


# ----------------------------------------------------------------------- writing
def gz_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


def write_10x(d: Path, counts: np.ndarray, ids, symbols, barcodes) -> None:
    d.mkdir(parents=True, exist_ok=True)
    c, g = np.nonzero(counts)                                   # sorted by cell, then gene
    v = counts[c, g]
    lines = ["%%MatrixMarket matrix coordinate integer general",
             "%metadata_json: {\"software_version\": \"scpipe make_stage_inputs.py\"}",
             f"{counts.shape[1]} {counts.shape[0]} {len(v)}"]
    body = "".join(f"{gi + 1} {ci + 1} {vi}\n" for ci, gi, vi in zip(c, g, v))
    (d / "matrix.mtx.gz").write_bytes(gz_bytes("\n".join(lines) + "\n" + body))
    (d / "features.tsv.gz").write_bytes(
        gz_bytes("".join(f"{i}\t{s}\tGene Expression\n" for i, s in zip(ids, symbols))))
    (d / "barcodes.tsv.gz").write_bytes(gz_bytes("".join(f"{b}\n" for b in barcodes)))


def write_h5ad(path: Path, counts: np.ndarray, ids, symbols, barcodes) -> None:
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    cnt = sp.csr_matrix(counts.astype(np.int32))
    lib = np.asarray(cnt.sum(1)).ravel().astype(np.float64)
    x = sp.diags(1e4 / lib) @ cnt.astype(np.float64)
    x.data = np.log1p(x.data)
    a = ad.AnnData(
        X=sp.csr_matrix(x, dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(barcodes, name=None)),
        var=pd.DataFrame({"gene_ids": list(ids), "feature_types": ["Gene Expression"] * len(ids)},
                         index=pd.Index(symbols, name=None)),
    )
    a.layers["counts"] = cnt
    a.uns["note"] = "scpipe stage test input: raw counts in layers['counts'], X is log1p(CP10k)"
    a.write_h5ad(path, compression="gzip")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_all(out: Path) -> dict:
    samples, symbols, ids, dropped, keep_b = build()
    out.mkdir(parents=True, exist_ok=True)

    a = samples["donor_A"]
    write_10x(out / "donor_A", a["counts"], ids, symbols, a["barcodes"])
    b = samples["donor_B"]
    write_h5ad(out / "donor_B.h5ad", b["counts"][:, keep_b],
               [ids[i] for i in keep_b], [symbols[i] for i in keep_b], b["barcodes"])

    with open(out / "samples.csv", "w", newline="\n") as fh:
        # donor_A's format is left blank on purpose: S0 must auto-detect 10x_mtx.
        fh.write("sample_id,path,format,donor,condition,batch\n")
        fh.write("donor_A,donor_A,,D1,control,B1\n")
        fh.write("donor_B,donor_B.h5ad,h5ad,D2,control,B2\n")

    with open(out / "truth.tsv", "w", newline="\n") as fh:
        fh.write("cell_id\tsample\tcell_type\tinjected\n")
        for s in SAMPLE_ORDER:
            d = samples[s]
            fh.write("".join(f"{s}_{bc}\t{s}\t{ct}\t{inj}\n"
                             for bc, ct, inj in zip(d["barcodes"], d["cell_type"], d["injected"])))

    per_sample = {}
    for s in SAMPLE_ORDER:
        d = samples[s]
        c = d["counts"] if s == "donor_A" else d["counts"][:, keep_b]
        inj = np.array(d["injected"])
        ng = (c > 0).sum(1)
        nu = c.sum(1)
        per_sample[s] = {
            "format": "10x_mtx" if s == "donor_A" else "h5ad",
            "n_cells": int(c.shape[0]),
            "n_genes": int(c.shape[1]),
            "total_umis": int(c.sum()),
            "injected": {k: int((inj == k).sum()) for k in ("none", "lowq", "doublet", "platelet")},
            "median_genes": {k: float(np.median(ng[inj == k])) for k in ("none", "lowq", "doublet", "platelet")},
            "median_umis": {k: float(np.median(nu[inj == k])) for k in ("none", "lowq", "doublet", "platelet")},
        }
    manifest = {
        "generator": "tests/fixtures/make_stage_inputs.py",
        "source": "tests/fixtures/pbmc_like",
        "seed": SEED,
        "n_genes_union": len(symbols),
        "added_genes": [s for s, _ in ADDED_GENES],
        "platelet_markers": list(PLATELET_MARKERS),
        "genes_missing_from_donor_B": [symbols[i] for i in dropped],
        "donor_B_gene_order": "permuted",
        "samples": per_sample,
        "files": {},
    }
    manifest["files"] = {f: sha256(out / f) for f in FILES if f != "manifest.json"}
    with open(out / "manifest.json", "w", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest


# ------------------------------------------------------------------------ check
def h5ad_equal(p: Path, q: Path) -> bool:
    import anndata as ad
    a, b = ad.read_h5ad(p), ad.read_h5ad(q)
    return (list(a.obs_names) == list(b.obs_names) and list(a.var_names) == list(b.var_names)
            and a.var.equals(b.var)
            and (a.layers["counts"] != b.layers["counts"]).nnz == 0
            and np.array_equal(a.X.toarray(), b.X.toarray()))


def check(out: Path) -> int:
    bad, soft = [], []
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        write_all(t)
        for f in FILES:
            if not (out / f).exists():
                bad.append(f"{f} (missing)")
            elif (t / f).read_bytes() != (out / f).read_bytes():
                if f == "donor_B.h5ad" and h5ad_equal(t / f, out / f):
                    soft.append(f)              # HDF5 bytes differ, content identical
                elif f == "manifest.json":
                    fresh = json.loads((t / f).read_text())
                    old = json.loads((out / f).read_text())
                    fresh["files"].pop("donor_B.h5ad", None)
                    old["files"].pop("donor_B.h5ad", None)
                    if fresh != old or "donor_B.h5ad" not in soft:
                        bad.append(f)
                else:
                    bad.append(f)
    if bad:
        print(f"FAIL: regenerated stage inputs differ in: {', '.join(bad)}", file=sys.stderr)
        return 1
    note = f" ({', '.join(soft)} content-identical, HDF5 bytes differ)" if soft else ""
    print(f"OK: stage inputs reproduce{note}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="regenerate into a temp dir and compare with the committed files")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    if a.check:
        return check(a.out)
    m = write_all(a.out)
    print(json.dumps({s: {"n_cells": v["n_cells"], "injected": v["injected"]}
                      for s, v in m["samples"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
