"""Data files on the command line: ``cellwarden run a.h5 b.rds c.h5ad`` (docs/PIPELINE.md section 3).

Turns paths given on the command line into a sample sheet and a config file that the normal
``config.load_config`` path then runs unchanged, so fingerprints, ``--detach`` and container binds
work as for a hand-written config:

- :func:`detect_format` recognises 10x h5 (Cell Ranger v2 and v3), 10x mtx folders, h5ad (by
  content, whatever the extension), Seurat/SCE ``.rds``/``.qs2``/``.qs``, loom and csv/tsv/txt
  count tables. S0 (R) re-detects and is authoritative.
- :func:`collect_inputs` expands folders and a samples sheet into sample rows with stable,
  barcode-safe sample IDs.
- :func:`synthesize_config` writes ``<out>/config/input_samples.csv`` (absolute paths) and
  ``<out>/config/input_config.yaml`` (the user's config, if any, with its relative paths made
  absolute, plus ``samples`` and ``run_dir``).
"""

from __future__ import annotations

import csv
import datetime as dt
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .config import ConfigError

INPUT_CONFIG = "input_config.yaml"
INPUT_SAMPLES = "input_samples.csv"

# File names that say nothing about the sample: the ID comes from the folder instead.
_GENERIC_STEMS = {"filtered_feature_bc_matrix", "raw_feature_bc_matrix", "sample_filtered_feature_bc_matrix",
                  "matrix", "counts", "outs", "filtered_gene_bc_matrices", "raw_gene_bc_matrices", "data", "object"}
_TABLE_RE = re.compile(r"\.(csv|tsv|txt)(\.gz)?$", re.I)
_SUFFIX_RE = re.compile(r"(\.(h5ad|h5|hdf5|rds|qs2|qs|loom|mtx|csv|tsv|txt))?(\.gz)?$", re.I)
# Config keys holding paths that load_config resolves against the config file's folder.
_PATH_KEYS = (("samples",), ("qc_policy",), ("run_dir",), ("signatures",), ("scratch",),
              ("opentargets", "cache_dir"), ("annotation", "models_dir"))


def _h5_kind(path: Path) -> str | None:
    try:
        import h5py

        with h5py.File(path, "r") as f:
            keys = set(f.keys())
            if "matrix" in keys:
                return "10x_h5"
            if {"obs", "var"} <= keys:
                return "h5ad"
            # Cell Ranger v2: one group per genome holding the CSC arrays
            for k in keys:
                g = f[k]
                if isinstance(g, h5py.Group) and {"barcodes", "data", "indptr"} <= set(g.keys()):
                    return "10x_h5"
    except Exception:
        return None
    return None


def detect_format(path: str | Path) -> str | None:
    """Format of a sample path (``config.SAMPLE_FORMATS``), or ``None`` if unrecognised."""
    p = Path(path)
    name = p.name.lower()
    if p.is_dir():
        if any((p / f).exists() for f in ("matrix.mtx", "matrix.mtx.gz")):
            return "10x_mtx"
        return None
    if name in ("matrix.mtx", "matrix.mtx.gz"):
        return "10x_mtx"
    if name.endswith(".h5ad"):
        return "h5ad"
    if name.endswith((".h5", ".hdf5")):
        return _h5_kind(p)
    if name.endswith((".rds", ".qs2", ".qs")):
        return "seurat"
    if name.endswith(".loom"):
        return "loom"
    if _TABLE_RE.search(name):
        return "table"
    return None


def is_samples_sheet(path: Path) -> bool:
    """A csv/tsv whose header has ``sample_id`` and ``path`` (anything else is a count table)."""
    if not path.is_file() or not re.search(r"\.(csv|tsv)$", path.name, re.I):
        return False
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            first = fh.readline()
    except OSError:
        return False
    cols = {c.strip().strip('"') for c in re.split(r"[,\t]", first)}
    return {"sample_id", "path"} <= cols


def _sample_id(path: Path) -> str:
    """Barcode-safe ID from the file name, or from the folder for generic names."""
    p = path
    if p.name.lower() in ("matrix.mtx", "matrix.mtx.gz"):
        p = p.parent
    stem = _SUFFIX_RE.sub("", p.name)
    parts = [q for q in [stem] + [a.name for a in p.parents] if q]
    for cand in parts:
        if cand.lower() not in _GENERIC_STEMS:
            stem = cand
            break
    sid = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not sid or not sid[0].isalnum():
        sid = "S" + sid
    return sid[:60]


def _expand(path: Path) -> list[Path]:
    """One input path → the sample paths it stands for."""
    if not path.exists():
        raise ConfigError(f"input does not exist: {path}")
    if path.is_file():
        return [path]
    if detect_format(path) == "10x_mtx":
        return [path]
    # Cell Ranger output folders: prefer the filtered matrix (h5, then mtx folder)
    for base in (path, path / "outs"):
        for cand in ("filtered_feature_bc_matrix.h5", "sample_filtered_feature_bc_matrix.h5", "filtered_feature_bc_matrix"):
            c = base / cand
            if c.exists() and detect_format(c):
                return [c]
    found = []
    for child in sorted(path.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_file() and detect_format(child):
            found.append(child)
        elif child.is_dir() and detect_format(child) == "10x_mtx":
            found.append(child)
    if not found:
        raise ConfigError(f"no single-cell data found in folder {path} (looked for 10x h5/mtx, h5ad, rds/qs2, loom, csv/tsv)")
    return found


def collect_inputs(paths: Sequence[str | Path]) -> tuple[list[dict[str, str]], Path | None]:
    """Sample rows ``{sample_id, path, format}`` for command-line inputs.

    Returns ``(rows, sheet)``: when the only input is a samples sheet it is returned as ``sheet``
    (with no rows) and used as is.
    """
    items = [Path(p).expanduser().resolve() for p in paths]
    sheets = [p for p in items if is_samples_sheet(p)]
    if sheets:
        if len(items) > 1:
            raise ConfigError(f"a samples sheet ({sheets[0].name}) must be the only input; list the other files in it")
        return [], sheets[0]
    files: list[Path] = []
    for p in items:
        for f in _expand(p):
            if f not in files:
                files.append(f)
    rows, used = [], set()
    for f in files:
        fmt = detect_format(f)
        if fmt is None:
            raise ConfigError(f"cannot tell what kind of file {f} is; supported: 10x h5 or mtx folder, h5ad, "
                              "Seurat/SCE .rds/.qs2, loom, csv/tsv count table (or pass a samples sheet)")
        sid = base = _sample_id(f)
        n = 2
        while sid in used:
            sid = f"{base}_{n}"
            n += 1
        used.add(sid)
        rows.append({"sample_id": sid, "path": str(f), "format": fmt})
    return rows, None


def parse_meta(pairs: Sequence[str] | None) -> dict[str, str]:
    """``--meta condition=control`` → ``{"condition": "control"}``."""
    out: dict[str, str] = {}
    for kv in pairs or []:
        k, sep, v = kv.partition("=")
        k = k.strip()
        if not sep or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", k) or k in ("sample_id", "path", "format"):
            raise ConfigError(f"--meta {kv!r}: expected NAME=VALUE with a column name made of letters, digits and _")
        out[k] = v.strip()
    return out


def default_out_dir(rows: Sequence[Mapping[str, str]], name: str | None = None,
                    today: dt.date | None = None, cwd: Path | None = None) -> Path:
    """``./cellwarden_<name>_<YYYYMMDD>``; ``name`` defaults to the first sample ID (or ``run``)."""
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", name or (rows[0]["sample_id"] if rows else "run"))
    day = (today or dt.date.today()).strftime("%Y%m%d")
    return (cwd or Path.cwd()) / f"cellwarden_{label}_{day}"


def check_out_dir(out: Path) -> None:
    """Refuse a non-empty folder that is not an earlier cellwarden run (never overwrite user files)."""
    if out.exists() and not out.is_dir():
        raise ConfigError(f"--out {out} exists and is not a folder")
    if out.is_dir() and any(out.iterdir()):
        if not ((out / "state.json").exists() or (out / "config" / INPUT_CONFIG).exists()):
            raise ConfigError(f"--out {out} is not empty and is not an earlier cellwarden run; choose another folder")


def _absolutize(raw: dict[str, Any], base: Path) -> dict[str, Any]:
    out = dict(raw)
    for keys in _PATH_KEYS:
        d = out
        for k in keys[:-1]:
            if not isinstance(d.get(k), Mapping):
                d = None
                break
            d[k] = dict(d[k])
            d = d[k]
        if d is None:
            continue
        v = d.get(keys[-1])
        if isinstance(v, str) and v.strip() and not Path(v).expanduser().is_absolute() and "$" not in v:
            d[keys[-1]] = str((base / Path(v).expanduser()).resolve())
    site = out.get("site")
    if isinstance(site, str) and (site.endswith((".yaml", ".yml")) or "/" in site) and not Path(site).is_absolute():
        out["site"] = str((base / site).resolve())
    return out


def synthesize_config(out: Path, rows: Sequence[Mapping[str, str]] | None = None, sheet: Path | None = None,
                      config_path: Path | None = None, meta: Mapping[str, str] | None = None,
                      overrides: Mapping[str, Any] | None = None) -> Path:
    """Write the sample sheet and config for a command-line run; returns the config path."""
    cfg_dir = out / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    raw: dict[str, Any] = {}
    if config_path is not None:
        try:
            raw = yaml.safe_load(config_path.read_text()) or {}
        except (OSError, yaml.YAMLError) as e:
            raise ConfigError(f"cannot read --config {config_path}: {e}") from e
        if not isinstance(raw, dict):
            raise ConfigError(f"--config {config_path} is not a YAML mapping")
        raw = _absolutize(raw, config_path.resolve().parent)
    if sheet is not None:  # a sheet's relative paths are relative to the sheet itself
        dialect = "excel-tab" if sheet.name.lower().endswith(".tsv") else "excel"
        with open(sheet, encoding="utf-8-sig", newline="") as fh:
            rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(fh, dialect=dialect)]
        for r in rows:
            p = Path(r.get("path", "")).expanduser()
            if r.get("path") and not p.is_absolute():
                r["path"] = str((sheet.parent / p).resolve())
    rows = [dict(r) for r in rows or []]
    for r in rows:
        for k, v in (meta or {}).items():
            r.setdefault(k, v)
    first = ["sample_id", "path", "format"]
    seen = [k for r in rows for k in r]
    cols = first + [k for i, k in enumerate(seen) if k not in first and k not in seen[:i]]
    target = cfg_dir / INPUT_SAMPLES
    with open(target, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    raw["samples"] = str(target)
    raw["run_dir"] = str(out)
    for k, v in (overrides or {}).items():
        raw[k] = v
    path = cfg_dir / INPUT_CONFIG
    header = ("# Written by `cellwarden run <inputs>`: the config this run used (edit it and rerun with\n"
              f"#   cellwarden run --config {path}\n# to change settings).\n")
    path.write_text(header + yaml.safe_dump(raw, sort_keys=False))
    return path
