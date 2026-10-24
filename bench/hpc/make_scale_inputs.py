#!/usr/bin/env python3
"""Synthetic pipeline inputs for the scale runs on the HPC (docs/scfast-integration.md, Phase 8).

One dataset per size, each a complete `cellwarden` input: a 10x ``.h5`` with the counts, a
``samples.csv`` sheet for it and the planted cell type of every cell (for ARI in
``summarise.py``). The counts come from ``bench/synth.py`` (negative binomial around per-type
profiles with marker genes, ``MT-`` genes for QC realism), so QC keeps nearly every cell and
the planted types cluster cleanly; the sizes are generated independently with the same seed.

    python bench/hpc/make_scale_inputs.py --sizes 50000,100000,200000,500000 --out bench/hpc/data

writes, per size N::

    <out>/n<N>/sample.h5     10x .h5, N cells x --genes genes
    <out>/n<N>/samples.csv   sample sheet (one sample, ``syn``, path relative to this folder)
    <out>/n<N>/truth.tsv     cell_id (syn_<barcode>, the S0 naming) and cell_type
    <out>/manifest.json      sizes, parameters, non-zeros and SHA-256 of every file

These are HPC-sized. Outside the HPC a size above ``MAX_LOCAL_CELLS`` (5,000) is refused
unless ``CELLWARDEN_ALLOW_LARGE=1`` is set, so a stray run on a laptop or in CI cannot fill the
disk; ``--sizes 2000`` and the like always work (that is what the unit test uses).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))

MAX_LOCAL_CELLS = 5_000
DEFAULT_SIZES = (50_000, 100_000, 200_000, 500_000)
SAMPLE_ID = "syn"


class RefusedError(RuntimeError):
    """A size above MAX_LOCAL_CELLS without CELLWARDEN_ALLOW_LARGE=1."""


def parse_sizes(text: str) -> list[int]:
    sizes = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n = int(float(tok))
        if n < 50:
            raise ValueError(f"size {tok!r} is too small (at least 50 cells)")
        sizes.append(n)
    if not sizes:
        raise ValueError("no sizes given")
    return sorted(set(sizes))


def check_allowed(sizes: list[int], env: dict | None = None) -> None:
    """Refuse HPC-sized inputs unless CELLWARDEN_ALLOW_LARGE=1 (RefusedError)."""
    env = os.environ if env is None else env
    big = [n for n in sizes if n > MAX_LOCAL_CELLS]
    if big and env.get("CELLWARDEN_ALLOW_LARGE") != "1":
        raise RefusedError(
            f"sizes {big} exceed {MAX_LOCAL_CELLS:,} cells; these inputs are meant for the HPC "
            "(docs/scfast-integration.md Phase 8). Set CELLWARDEN_ALLOW_LARGE=1 to generate them here."
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_dataset(n_cells: int, out: Path, n_genes: int, n_types: int, seed: int) -> dict:
    import synth  # noqa: E402  (bench/synth.py)
    from h5io import write_10x_h5  # noqa: E402

    counts, barcodes, ids, names, labels = synth.generate(n_cells, n_genes, n_types, seed)
    d = out / f"n{n_cells}"
    d.mkdir(parents=True, exist_ok=True)
    h5 = d / "sample.h5"
    write_10x_h5(h5, counts, barcodes, ids, names, genome="synthetic")
    truth = d / "truth.tsv"
    with open(truth, "w") as f:
        f.write("cell_id\tcell_type\n")
        for b, t in zip(barcodes, labels):
            f.write(f"{SAMPLE_ID}_{b}\t{t}\n")
    sheet = d / "samples.csv"
    sheet.write_text("sample_id,path,format,donor,condition,batch\n"
                     f"{SAMPLE_ID},sample.h5,10x_h5,D1,control,B1\n")
    return {
        "cells": n_cells, "genes": n_genes, "types": n_types, "nnz": int(counts.nnz), "dir": str(d),
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in (h5, truth, sheet)},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                   help="comma-separated cell counts (default: %(default)s)")
    p.add_argument("--genes", type=int, default=5_000)
    p.add_argument("--types", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output folder; one n<N>/ per size")
    a = p.parse_args(argv)
    try:
        sizes = parse_sizes(a.sizes)
        check_allowed(sizes)
    except (ValueError, RefusedError) as e:
        print(f"make_scale_inputs: {e}", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": a.seed, "genes": a.genes, "types": a.types, "sample_id": SAMPLE_ID, "datasets": []}
    for n in sizes:
        rec = write_dataset(n, out, a.genes, a.types, a.seed)
        manifest["datasets"].append(rec)
        print(f"wrote {rec['dir']}: {n:,} cells x {a.genes:,} genes, {rec['nnz']:,} non-zeros")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
