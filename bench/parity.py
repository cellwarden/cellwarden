"""Parity of every engine against a reference engine (default: Seurat), per dataset size.

Compares the outputs each runner saved from its first measured repeat:

  pc_similarity      mean cosine of the principal angles between the top-N PC
                     subspaces (1.0 = same subspace; sign and rotation ignored)
  pc_similarity_min  the smallest of those cosines
  knn_overlap        mean fraction of shared neighbours per cell
  ari                adjusted Rand index between cluster labels
  ari_truth          ARI against known cell types, when prep.py had --labels

Thresholds come from docs/RUST_ENGINE.md section 7. Writes <run>/parity.jsonl.

    python bench/parity.py --run bench/results/run1 [--reference seurat --n-pcs 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def subspace_similarity(a: np.ndarray, b: np.ndarray, n: int) -> tuple[float, float]:
    """Cosines of the principal angles between the column spaces of a[:, :n] and b[:, :n]."""
    n = min(n, a.shape[1], b.shape[1])
    qa, _ = np.linalg.qr(a[:, :n] - a[:, :n].mean(axis=0), mode="reduced")
    qb, _ = np.linalg.qr(b[:, :n] - b[:, :n].mean(axis=0), mode="reduced")
    cos = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0.0, 1.0)
    return float(cos.mean()), float(cos.min())


def knn_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean |N_a(i) & N_b(i)| / k over cells; -1 entries are padding."""
    k = min(a.shape[1], b.shape[1])
    total = 0.0
    for ra, rb in zip(a[:, :k], b[:, :k]):
        sa, sb = set(ra[ra >= 0].tolist()), set(rb[rb >= 0].tolist())
        denom = max(min(len(sa), len(sb)), 1)
        total += len(sa & sb) / denom
    return total / len(a)


def ari(x, y) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(x, y))


def remap_knn(knn: np.ndarray, src_barcodes: list[str], dst_index: dict[str, int], order: np.ndarray) -> np.ndarray:
    """Re-express neighbour indices in the reference's cell order."""
    lut = np.array([dst_index.get(b, -1) for b in src_barcodes] + [-1], dtype=np.int64)
    k2 = knn[order].astype(np.int64)
    k2[k2 < 0] = len(src_barcodes)  # padding -> -1 through the last lut entry
    return lut[k2]


def compare(ref_dir: Path, eng_dir: Path, n_pcs: int, truth: dict[str, str] | None) -> dict:
    rb, rpcs, rknn, rlab = common.read_outputs(ref_dir)
    eb, epcs, eknn, elab = common.read_outputs(eng_dir)
    ref_index = {b: i for i, b in enumerate(rb)}
    eng_index = {b: i for i, b in enumerate(eb)}
    shared = [b for b in rb if b in eng_index]
    if len(shared) != len(rb) or len(shared) != len(eb):
        raise ValueError(f"cell sets differ: reference {len(rb)}, engine {len(eb)}, shared {len(shared)}")
    order = np.array([eng_index[b] for b in rb])  # engine rows in reference order
    sim, sim_min = subspace_similarity(rpcs, epcs[order], n_pcs)
    out = {
        "pc_similarity": round(sim, 5),
        "pc_similarity_min": round(sim_min, 5),
        "knn_overlap": round(knn_overlap(rknn, remap_knn(eknn, eb, ref_index, order)), 5),
        "ari": round(ari(rlab, [elab[i] for i in order]), 5),
        "n_clusters_ref": len(set(rlab)),
        "n_clusters": len(set(elab)),
    }
    if truth:
        out["ari_truth"] = round(ari([truth[b] for b in eb], elab), 5)
        out["ari_truth_ref"] = round(ari([truth[b] for b in rb], rlab), 5)
    out["pass"] = {m: out[m] >= t for m, t in common.PARITY_THRESHOLDS.items()}
    out["pass_all"] = all(out["pass"].values())
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--reference", default="seurat")
    p.add_argument("--n-pcs", type=int, default=10)
    a = p.parse_args(argv)
    run = Path(a.run)
    outputs = run / "outputs"
    records = []
    if not (outputs / a.reference).exists():
        print(f"no outputs for reference engine {a.reference!r}; nothing to compare")
    else:
        for ref_dir in sorted((outputs / a.reference).iterdir()):
            dataset = ref_dir.name
            truth = None
            cells = run / "data" / dataset / "cells.tsv"
            if cells.exists():
                import pandas as pd

                t = pd.read_csv(cells, sep="\t", dtype=str)
                if "cell_type" in t.columns:
                    truth = dict(zip(t["barcode"], t["cell_type"]))
            for eng in sorted(outputs.iterdir()):
                if eng.name == a.reference or not (eng / dataset).exists():
                    continue
                rec = {"engine": eng.name, "reference": a.reference, "dataset": dataset,
                       "cells": json.loads((ref_dir / "meta.json").read_text())["n_cells"],
                       "n_pcs_compared": a.n_pcs}
                try:
                    rec.update(compare(ref_dir, eng / dataset, a.n_pcs, truth))
                    rec["status"] = "ok"
                except Exception as e:
                    rec.update(status="error", error=f"{type(e).__name__}: {e}")
                records.append(rec)
                print(json.dumps(rec))
    (run / "parity.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
