"""Generate a synthetic scRNA-seq dataset with known cell types, as a 10x .h5.

For scaling runs on any machine and for testing the harness itself. Counts are
negative binomial around per-type expression profiles with marker genes; the
true type of every cell goes to ``<out>.labels.tsv`` (barcode, cell_type).
Real data is harder to cluster, so use this for timing and plumbing, and real
data for parity.

    python bench/synth.py --cells 100000 --genes 5000 --types 10 --out data/synth_100k.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h5io import write_10x_h5  # noqa: E402


def generate(n_cells: int, n_genes: int, n_types: int, seed: int = 0, chunk: int = 20_000):
    rng = np.random.default_rng(seed)
    # Unequal type sizes, like blood: a few big populations and a long tail.
    w = rng.dirichlet(np.full(n_types, 0.8)) * 0.85 + 0.15 / n_types
    types = rng.choice(n_types, size=n_cells, p=w / w.sum())
    base = rng.lognormal(mean=-2.0, sigma=1.6, size=n_genes)          # baseline expression
    profiles = np.tile(base, (n_types, 1))
    n_markers = max(10, n_genes // 100)
    for t in range(n_types):
        markers = rng.choice(n_genes, size=n_markers, replace=False)
        profiles[t, markers] *= rng.uniform(4, 20, size=n_markers)      # up-regulated markers
    profiles /= profiles.sum(axis=1, keepdims=True)
    n_mt = min(13, n_genes // 50)                                       # MT- genes for QC realism
    names = [f"MT-SYN{i}" for i in range(n_mt)] + [f"GENE{i}" for i in range(n_genes - n_mt)]
    ids = [f"ENSGSYN{i:011d}" for i in range(n_genes)]
    blocks = []
    dispersion = 0.3
    for s in range(0, n_cells, chunk):
        t = types[s : s + chunk]
        lib = rng.lognormal(mean=np.log(3000), sigma=0.4, size=len(t))
        mu = profiles[t] * lib[:, None]
        lam = rng.gamma(shape=1 / dispersion, scale=mu * dispersion)  # NB as gamma-Poisson
        blocks.append(sp.csr_matrix(rng.poisson(lam).astype(np.int32)))
    counts = sp.vstack(blocks).tocsr()
    barcodes = [f"SYN{i:09d}-1" for i in range(n_cells)]
    return counts, barcodes, ids, names, [f"type{t}" for t in types]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cells", type=int, default=10_000)
    p.add_argument("--genes", type=int, default=5_000)
    p.add_argument("--types", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output .h5 path")
    a = p.parse_args(argv)
    counts, bc, ids, names, labels = generate(a.cells, a.genes, a.types, a.seed)
    write_10x_h5(a.out, counts, bc, ids, names, genome="synthetic")
    Path(a.out + ".labels.tsv").write_text(
        "barcode\tcell_type\n" + "".join(f"{b}\t{t}\n" for b, t in zip(bc, labels))
    )
    print(f"wrote {a.out}: {a.cells} cells x {a.genes} genes, {counts.nnz} non-zeros, {a.types} types")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
