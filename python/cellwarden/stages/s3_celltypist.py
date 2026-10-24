"""S3 (CellTypist half): reference-mapping votes per cell (docs/agents-spec.md section 4).

    python -m cellwarden.stages.s3_celltypist --run-dir DIR [--models-dir DIR] [--download]

Input: ``S3/exchange/counts.h5ad`` (full-mode exchange file written by the R
step ``run_s3`` only when ``annotation.celltypist`` is not ``off`` and
``celltypist`` is importable; X is the log1p-CP10k ``data`` layer CellTypist
expects, ``var['gene_symbol_original']`` the symbols).

Outputs:

* ``S3/celltypist.tsv.gz``: ``cell_id, predicted_label, majority_voting,
  conf_score`` (one row per cell), only when CellTypist ran;
* ``S3/celltypist_note.json``: always; ``{"status": "ok" | "skipped",
  "reason", "model", "models_dir", "n_cells", "celltypist_version", ...}``.

The stage never fails because CellTypist is missing: it is **skipped with a
note** when the package is not installed, the model is not available offline,
or the input file was not written. The input file is deleted after use.

Models: CellTypist downloads ``.pkl`` models on first use into
``CELLTYPIST_FOLDER``. The stage sets that to ``--models-dir`` /
``annotation.models_dir`` (default ``refs/celltypist`` next to ``config.yaml``)
so a shared cache can be pre-filled on a login node with internet access::

    python -m cellwarden.stages.s3_celltypist --models-dir refs/celltypist --download Immune_All_Low.pkl

``--download`` fetches the named model(s) into the cache and exits.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from . import StageError, log, stage_main, utc_now, write_json, write_tsv_gz

STAGE = "S3-celltypist"
DEFAULT_MODEL = "Immune_All_Low.pkl"
NOTE_FILE = "celltypist_note.json"
INPUT_FILE = Path("S3") / "exchange" / "counts.h5ad"


def annotation_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    a = dict(cfg.get("annotation") or {})
    a.setdefault("celltypist_model", DEFAULT_MODEL)
    a.setdefault("celltypist", "auto")
    a.setdefault("min_confidence", 0.6)
    a.setdefault("models_dir", None)
    return a


def models_dir_for(cfg: dict[str, Any], override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    a = annotation_cfg(cfg)
    if a.get("models_dir"):
        return Path(str(a["models_dir"])).expanduser().resolve()
    if os.environ.get("CELLWARDEN_CELLTYPIST_MODELS"):  # models baked into the container image
        return Path(os.environ["CELLWARDEN_CELLTYPIST_MODELS"])
    base = Path(cfg.get("config_dir") or cfg.get("run_dir") or ".")
    return (base / "refs" / "celltypist").resolve()


def _import_celltypist(models_dir: Path):
    os.environ["CELLTYPIST_FOLDER"] = str(models_dir)
    try:
        import celltypist  # noqa: WPS433

        return celltypist
    except Exception:  # ImportError or a broken install
        return None


def _model_path(models_dir: Path, model: str) -> Path:
    p = Path(model)
    if p.is_absolute() or p.exists():
        return p.resolve()
    return models_dir / "models" / model


def download_models(models: list[str], models_dir: Path) -> int:
    models_dir.mkdir(parents=True, exist_ok=True)
    ct = _import_celltypist(models_dir)
    if ct is None:
        print("celltypist is not installed (pip install celltypist)", file=sys.stderr)
        return 1
    for m in models:
        log(f"downloading {m} into {models_dir}")
        ct.models.download_models(model=m, force_update=False)
        log(f"  -> {_model_path(models_dir, m)}")
    return 0


def run(run_dir: Path, cfg: dict[str, Any], models_dir: str | None = None) -> dict[str, Any]:
    """Run CellTypist if possible; always writes the note. Returns the note."""
    s3 = run_dir / "S3"
    s3.mkdir(parents=True, exist_ok=True)
    a = annotation_cfg(cfg)
    mdir = models_dir_for(cfg, models_dir)
    model = str(a["celltypist_model"])
    note: dict[str, Any] = {"stage": STAGE, "created_utc": utc_now(), "status": "skipped", "reason": "",
                            "model": model, "models_dir": str(mdir), "n_cells": 0, "celltypist_version": None,
                            "output": None}
    inp = run_dir / INPUT_FILE
    out = s3 / "celltypist.tsv.gz"

    def skip(reason: str) -> dict[str, Any]:
        note["reason"] = reason
        log(f"celltypist skipped: {reason}")
        if out.exists():
            out.unlink()  # a stale table from an earlier run must not survive a skip
        if inp.exists():
            inp.unlink()
        write_json(note, s3 / NOTE_FILE)
        return note

    if a.get("celltypist") == "off":
        return skip("annotation.celltypist is off")
    ct = _import_celltypist(mdir)
    if ct is None:
        return skip("the celltypist package is not installed (pip install celltypist; models cache: "
                    f"{mdir})")
    note["celltypist_version"] = getattr(ct, "__version__", "unknown")
    if not inp.exists():
        return skip(f"input {inp} was not written by the R step (celltypist not importable from R's python?)")
    mpath = _model_path(mdir, model)
    if not mpath.exists():
        try:
            log(f"model {model} not in {mdir}; trying to download")
            mdir.mkdir(parents=True, exist_ok=True)
            ct.models.download_models(model=model, force_update=False)
        except Exception as e:  # offline
            return skip(f"model {model} is not in {mdir} and could not be downloaded ({e}); "
                        f"pre-download with: python -m cellwarden.stages.s3_celltypist --models-dir {mdir} --download {model}")
        if not mpath.exists():
            return skip(f"model {model} is still missing after download into {mdir}")

    import anndata as ad
    import numpy as np

    adata = ad.read_h5ad(inp)
    if "gene_symbol_original" in adata.var.columns:
        adata.var_names = adata.var["gene_symbol_original"].astype(str).values
    adata.var_names_make_unique()
    n = adata.n_obs
    if n < 2:
        return skip(f"only {n} cell(s)")
    # X is Seurat's log1p(CP10k) data layer, which is what CellTypist expects; recompute if it is not.
    x = adata.X
    mx = float(x.max()) if n else 0.0
    if mx > 30:
        import scanpy as sc

        log("X does not look log-normalised; normalising counts (CP10k, log1p)")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    log(f"celltypist {note['celltypist_version']}: {n} cells, model {mpath.name}")
    mdl = ct.models.Model.load(str(mpath))
    majority = n >= 50
    pred = ct.annotate(adata, model=mdl, majority_voting=majority)
    tab = pred.predicted_labels.copy()
    tab.index.name = "cell_id"
    tab = tab.reset_index()
    if "majority_voting" not in tab.columns:
        tab["majority_voting"] = tab["predicted_labels"]
    conf = pred.probability_matrix.max(axis=1).values if hasattr(pred, "probability_matrix") else np.full(n, np.nan)
    df = tab.rename(columns={"predicted_labels": "predicted_label"})[["cell_id", "predicted_label", "majority_voting"]]
    df["conf_score"] = np.round(conf, 5)
    write_tsv_gz(df, out, na_rep="")
    inp.unlink(missing_ok=True)
    note.update(status="ok", reason="", n_cells=int(n), output=str(out), majority_voting=majority,
                model_path=str(mpath), labels=sorted(df["majority_voting"].astype(str).unique().tolist()))
    write_json(note, s3 / NOTE_FILE)
    log(f"celltypist done: {n} cells, {df['majority_voting'].nunique()} majority labels")
    return note


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="python -m cellwarden.stages.s3_celltypist")
    p.add_argument("--run-dir", default=os.environ.get("CELLWARDEN_RUN_DIR"))
    p.add_argument("--models-dir", default=None, help="CellTypist model cache (default annotation.models_dir or refs/celltypist)")
    p.add_argument("--download", nargs="*", metavar="MODEL", help="download these models into --models-dir and exit")
    a = p.parse_args(argv)
    if a.download is not None:
        models = a.download or [DEFAULT_MODEL]
        mdir = Path(a.models_dir or "refs/celltypist").expanduser().resolve()
        return download_models(models, mdir)
    if not a.run_dir:
        p.error("--run-dir is required")
    rest = ["--run-dir", a.run_dir]
    return stage_main(STAGE, lambda rd, cfg: run(rd, cfg, a.models_dir), rest)


if __name__ == "__main__":
    raise SystemExit(main())
