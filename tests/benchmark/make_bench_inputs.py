#!/usr/bin/env python3
"""Generate synthetic benchmark datasets for cellwarden (development sizes only).

During development we benchmark at **2,000 and 5,000 cells only**. Larger runs
(50k+) are done on the HPC before the first real analysis, not here.

Usage:
    python tests/benchmark/make_bench_inputs.py            # 2k and 5k, 2000 genes
    python tests/benchmark/make_bench_inputs.py --sizes 2000 --genes 1500

Writes, per size, `tests/benchmark/inputs/bench_<n>k/` holding one 10x MatrixMarket
sample (`matrix.mtx`, `features.tsv`, `barcodes.tsv`), `samples.csv` and `config.yaml`,
reusing the pbmc_like simulator from tests/fixtures/make_fixture.py with its size
constants overridden. Outputs are not committed (they are regenerated on demand).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fixtures"))
import make_fixture as mf  # noqa: E402

CONFIG = """\
project: bench_{n}k_{device}
run_dir: {run_dir}
seed: 42
samples: samples.csv
object_format: rds
resources: {{threads: {threads}, max_memory_gb: 16, gpu_mem_cap_gb: 20, nice: 10}}
doublets: {{method: scrublet}}
qc: {{review_action: keep}}
normalization: lognorm
n_hvg: 1000
device: {device}
integration: {{method: none, batch_key: sample_id}}
pcs: auto
n_neighbors: 20
clustering: {{resolution: 0.5}}
"""


def write_dataset(n_cells: int, n_genes: int, out: Path, threads: int) -> None:
    mf.N_CELLS = n_cells
    mf.N_GENES = n_genes
    counts, symbols, ids, bcs, sample, group = mf.simulate()
    d = out / "sample"
    d.mkdir(parents=True, exist_ok=True)
    _write_mtx(counts, d / "matrix.mtx")
    with open(d / "features.tsv", "w") as fh:
        for gid, sym in zip(ids, symbols):
            fh.write(f"{gid}\t{sym}\tGene Expression\n")
    with open(d / "barcodes.tsv", "w") as fh:
        fh.write("\n".join(bcs) + "\n")
    with open(out / "samples.csv", "w") as fh:
        fh.write("sample_id,path,format\nbench,sample,10x_mtx\n")
    k = n_cells // 1000
    for device in ("cpu", "gpu"):
        (out / f"config_{device}.yaml").write_text(
            CONFIG.format(n=k, device=device, threads=threads,
                          run_dir=str(out / f"run_{device}")))
    print(f"wrote {out}: {n_cells} cells x {n_genes} genes")


def _write_mtx(counts, path: Path) -> None:
    import numpy as np
    cell, gene = np.nonzero(counts)
    vals = counts[cell, gene]
    with open(path, "w") as fh:
        fh.write("%%MatrixMarket matrix coordinate integer general\n")
        fh.write(f"{counts.shape[1]} {counts.shape[0]} {len(vals)}\n")
        for g, c, v in zip(gene, cell, vals):
            fh.write(f"{g + 1} {c + 1} {v}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[2000, 5000])
    ap.add_argument("--genes", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=HERE / "inputs")
    a = ap.parse_args()
    for n in a.sizes:
        if n > 5000:
            sys.exit("Development benchmarks are limited to 2k and 5k cells; "
                     "run larger sizes on the HPC.")
        write_dataset(n, a.genes, a.out / f"bench_{n // 1000}k", a.threads)


if __name__ == "__main__":
    main()
