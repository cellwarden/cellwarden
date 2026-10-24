"""Run configuration: load, fill defaults, validate, and write the run's config folder.

Implements docs/stages-spec.md section 4 (``config.yaml``) and the
``<run-dir>/config/`` part of section 2.

* :func:`load_config` reads ``config.yaml``, ``samples.csv`` and
  ``qc_policy.yaml``, fills in defaults, resolves paths and validates
  everything, raising one :class:`ConfigError` that lists **every** problem.
* :func:`prepare_run_dir` writes ``config.yaml``, ``samples.csv`` and
  ``qc_policy.yaml`` (copies), ``config.resolved.yaml`` and ``hashes.json``.
* :func:`load_resolved` reads ``config.resolved.yaml`` back (what stages use).

Path rules:

* ``run_dir``, ``samples`` and ``qc_policy`` in ``config.yaml``: environment
  variables (``$USER``, ``${VAR}``) and ``~`` are expanded; relative paths are
  resolved against the directory containing ``config.yaml``. A ``--run-dir``
  override is resolved against the current directory.
* ``path`` values in ``samples.csv``: relative paths are also resolved against
  the directory containing ``config.yaml`` (spec section 4: "relative paths
  are resolved against the directory containing config.yaml"), whatever
  directory ``samples.csv`` itself is in; env vars and ``~`` are expanded.
  The absolute results are written to ``samples_resolved`` and that
  directory to ``config_dir`` in ``config.resolved.yaml`` (S0 in R resolves
  the copied sheet's paths against ``config_dir``).
* ``config.resolved.yaml`` points ``samples`` and ``qc_policy`` at the copies
  in ``<run-dir>/config/``; ``samples_source`` / ``qc_policy_source`` hold
  the user's originals. When ``qc_policy`` is not set, the built-in default
  policy (spec section 3.4) is written as the copy.

``config_hash`` = SHA-256 of the canonical JSON (sorted keys) of the resolved
analysis settings (every section, but not ``run_id``, ``run_dir``,
``repo_root`` or the ``*_source`` paths) plus the SHA-256 of the
``samples.csv`` and ``qc_policy.yaml`` contents.
"""

from __future__ import annotations

import copy
import csv
import getpass
import hashlib
import io
import json
import os
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import qc as qcmod

__all__ = [
    "ConfigError",
    "DEFAULTS",
    "SAMPLE_FORMATS",
    "ALLOWED",
    "REPO_ROOT",
    "load_config",
    "prepare_run_dir",
    "load_resolved",
    "get_key",
    "llm_settings",
    "sha256_file",
    "canonical_json",
    "detect_format",
]

REPO_ROOT = Path(os.environ.get("CELLWARDEN_REPO_ROOT") or Path(__file__).resolve().parents[2])

DEFAULTS: dict[str, Any] = {
    "project": "cellwarden",
    "run_dir": None,
    "seed": 42,
    "samples": None,
    "qc_policy": None,
    "object_format": "rds",
    "resources": {"threads": 8, "max_memory_gb": 64, "gpu_mem_cap_gb": 20, "nice": 10},
    "doublets": {"method": "scrublet"},
    "qc": {"review_action": "keep"},
    # S0: empty-droplet filtering of raw 10x matrices (auto: when the matrix looks raw).
    "ingest": {"empty_drops": "auto"},
    "normalization": "lognorm",
    "n_hvg": 2000,
    "engine": "auto",
    # GPU-plan engine (docs/scfast-integration.md "Pipeline wiring"): auto = scfast in process on CUDA
    # when scfastR was built with cuda and a device is visible, else rapids through the exchange.
    "engine_gpu": "auto",
    "device": "auto",
    # integration.engine (harmony): auto = scfast's Rust Harmony when scfastR/scfast provide it, else
    # R harmony / harmonypy / rapids; scfast or r force one. integration.harmony: harmony 2.x
    # parameters (null = the method's default; lambda null = automatic per-batch ridge).
    "integration": {"method": "none", "batch_key": "sample_id", "engine": "auto",
                    "harmony": {"theta": 2, "lambda": None, "sigma": 0.1, "nclust": None, "max_iter": 10}},
    "pcs": "auto",
    "n_neighbors": 20,
    "clustering": {"resolution": 0.5},
    # S3 evidence and consensus annotation (docs/agents-spec.md sections 4-5).
    # celltypist: auto (run when installed and a model is available, else skip with a note) | off;
    # models_dir: CellTypist model cache (null = refs/celltypist next to config.yaml);
    # min_confidence: consensus below it is low_confidence (the G3 agent may revise it).
    "annotation": {"celltypist_model": "Immune_All_Low.pkl", "celltypist": "auto", "min_confidence": 0.6,
                   "models_dir": None},
    # Optional path to a signature YAML replacing R/stages/data/signatures.yaml.
    "signatures": None,
    # S4 differential expression and Open Targets (docs/agents-spec.md section 6).
    # method findmarkers: Seurat::FindMarkers with `test` between two sets of cells; pseudobulk
    # (or deseq2 / limma_voom): counts summed per group x sample, DESeq2 or limma-voom.
    # contrasts: [{name, column (default condition_col), numerator, denominator (values; "rest" =
    # every other value), within (default group_by; "none" = all cells), covariates}]. With no
    # contrasts and markers auto, findmarkers finds the markers of each group_by group vs the rest.
    "dge": {
        "method": "findmarkers",
        "test": "MAST",
        "pseudobulk_method": "deseq2",
        "markers": "auto",
        # true: stop before S4, once the cell types are known, and ask for the comparisons
        # (cellwarden run -i, "after clustering and annotation"; cellwarden/dge_choice.py)
        "interactive": False,
        "sample_col": "sample_id",
        "condition_col": "condition",
        "group_by": "cell_type_broad",
        "contrasts": [],
        "min_pct": 0.1,
        "logfc_threshold": 0.25,
        "min_cells_per_side": 10,
        "max_cells_per_side": None,
        "min_cells_per_sample": 20,
        "min_replicates": 2,
        "padj": 0.05,
        "lfc": 0.5,
    },
    "opentargets": {
        "enabled": True,
        "max_genes_per_direction": 50,
        "cache_dir": "refs/opentargets",
        "offline": False,
        "timeout_s": 30,
        "retries": 3,
    },
    # Portability (docs/agents-spec.md section 8, DESIGN 7.1-7.2). `site:` names
    # a site file (conf/sites/<name>.yaml or a path) whose execution, resources,
    # container and scheduler sections are defaults; config.yaml overrides them.
    # S5 report and explorer (docs/agents-spec.md section 7)
    "report": {"explorer_max_cells": 100000, "expr_genes": 200, "expr_max_cells": 20000, "top_markers": 10},
    "site": None,
    "scratch": None,
    "execution": "direct",
    "container": {"runtime": "none", "image": None, "options": []},
    "scheduler": {"partition": None, "account": None, "queue": None, "time": "24:00:00", "cpus": None,
                  "mem_gb": None, "gpus": 0, "extra": []},
    # Gate modes (docs/agents-spec.md section 2 step 5): auto | review | escalate.
    "gates": {"G1": "auto", "G2": "review", "G3": "auto", "G4": "review"},
}

# `llm:` (DESIGN 4.8 + agents-spec section 3) is optional and deliberately NOT in
# DEFAULTS: when the section is absent the gates run rules-only and never call a
# model. When present it is validated by `_check_llm` and these defaults are
# applied by `llm_settings()` (used by the gates), not merged into the file.
LLM_DEFAULTS: dict[str, Any] = {
    # provider auto: Claude, then OpenAI, then NVIDIA, by which key is set; the named provider fills in
    # its model, key variable and endpoint (cellwarden.llm.provider.PRESETS).
    "default": {"provider": "auto"},
    "roles": {},
    "budget": {"currency": "GBP", "total_limit": 30, "per_run_limit": 5, "warn_at_pct": 80, "usd_to_gbp": 0.78,
               "ledger_path": "~/.cellwarden/ledger.json"},
    "replay": False,
}
from .llm.provider import PRESETS as LLM_PRESETS, PROVIDERS as LLM_PROVIDERS, auto_provider  # noqa: E402
# Per-role settings (llm.default, llm.roles.<role>); also accepted flat under llm: as a shorthand for llm.default.
LLM_ROLE_KEYS = ("provider", "model", "api_key_env", "base_url", "max_tokens", "timeout_s", "max_retries",
                 "temperature", "json_mode", "default_usage", "extra_body")
GATE_MODES = ("auto", "review", "escalate")

EXECUTION_MODES = ("direct", "slurm", "pbs", "lsf", "sge", "split")
SITE_SECTIONS = ("execution", "resources", "container", "scheduler", "scratch")

ALLOWED = {
    "execution": EXECUTION_MODES,
    "container.runtime": ("none", "apptainer", "singularity", "docker"),
    "object_format": ("rds", "qs2"),
    "doublets.method": ("scrublet", "scdblfinder", "none"),
    "qc.review_action": ("keep", "exclude"),
    "normalization": ("lognorm", "sct"),
    "ingest.empty_drops": ("auto", "on", "off"),
    "device": ("auto", "cpu", "gpu"),
    "engine": ("auto", "scfast", "seurat"),
    "engine_gpu": ("auto", "scfast", "rapids"),
    "integration.method": ("none", "harmony"),
    "integration.engine": ("auto", "scfast", "r"),
    "annotation.celltypist": ("auto", "off"),
    "dge.method": ("findmarkers", "pseudobulk", "deseq2", "limma_voom"),
    "dge.test": ("MAST", "wilcox", "wilcox_limma", "bimod", "t", "negbinom", "poisson", "LR", "DESeq2"),
    "dge.pseudobulk_method": ("deseq2", "limma_voom"),
}

_CONTRAST_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

SAMPLE_FORMATS = ("10x_h5", "10x_mtx", "h5ad", "seurat", "loom", "table")

# Keys the runner adds to config.resolved.yaml (never read from the user's file).
_DERIVED_KEYS = (
    "run_id",
    "config_hash",
    "repo_root",
    "config_file",
    "config_dir",
    "samples_source",
    "qc_policy_source",
    "samples_resolved",
    "sample_columns",
    "policy_hash",
    "cellwarden_version",
    "site_source",
    "llm_choice",
)
_HASH_EXCLUDE = ("run_id", "config_hash", "run_dir", "repo_root", "config_file", "config_dir", "samples_source",
                 "qc_policy_source", "samples", "qc_policy", "cellwarden_version",
                 # where/how the run executes does not change its results
                 "site", "site_source", "execution", "container", "scheduler", "scratch")


class ConfigError(ValueError):
    """The configuration is invalid. ``problems`` lists every problem found."""

    def __init__(self, problems: list[str] | str):
        if isinstance(problems, str):
            problems = [problems]
        self.problems = list(problems)
        super().__init__("invalid configuration:\n  - " + "\n  - ".join(self.problems))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_key(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """``get_key(cfg, "clustering.resolution")``; missing keys give ``default``."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


_ENV_RE = re.compile(r"\$(\w+)|\$\{([^}]+)\}")


def _expand(value: str, problems: list[str], what: str) -> str:
    env = dict(os.environ)
    if "USER" not in env:
        try:
            env["USER"] = getpass.getuser()
        except Exception:
            pass

    def sub(m: re.Match) -> str:
        name = m.group(1) or m.group(2)
        if name not in env:
            problems.append(f"{what}: environment variable ${name} is not set (in {value!r})")
            return m.group(0)
        return env[name]

    return os.path.expanduser(_ENV_RE.sub(sub, value))


def _resolve_path(value: str, base: Path, problems: list[str], what: str) -> Path:
    p = Path(_expand(value, problems, what))
    if not p.is_absolute():
        p = base / p
    return Path(os.path.normpath(p))


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def detect_format(path: Path) -> str | None:
    """Best-effort format detection (spec 3.1). ``None`` if unrecognised.

    S0 (R) is authoritative; this only lets the runner validate the sheet early. The rules live
    in :func:`cellwarden.inputs.detect_format`, shared with command-line inputs.
    """
    from .inputs import detect_format as _detect

    return _detect(path)


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------


def _merge_defaults(raw: Mapping[str, Any], problems: list[str], warnings: list[str]) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in raw.items():
        if k in _DERIVED_KEYS:
            warnings.append(f"key {k!r} is set by the runner; the value in config.yaml is ignored")
            continue
        if k == "llm":
            # Optional section (see LLM_DEFAULTS): kept as written, validated in _check_llm.
            warnings.append("optional key 'llm' is set: the gate agents will call an LLM (not used by S0-S2)")
            cfg[k] = v
            continue
        if k not in DEFAULTS:
            # Later build steps add sections (llm, gates, dge, ...); keep them, but say so.
            warnings.append(f"unknown top-level key {k!r} (kept as is, not used by S0-S2)")
            cfg[k] = v
            continue
        if isinstance(DEFAULTS[k], dict):
            if v is None:
                continue
            if not isinstance(v, Mapping):
                problems.append(f"{k} must be a mapping, got {type(v).__name__}")
                continue
            for kk, vv in v.items():
                if kk not in DEFAULTS[k]:
                    warnings.append(f"unknown key {k}.{kk} (kept as is, not used by S0-S2)")
                cfg[k][kk] = vv
        else:
            cfg[k] = v
    return cfg


def _check_values(cfg: dict[str, Any], problems: list[str]) -> None:
    def need(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    need(isinstance(cfg["project"], str) and bool(cfg["project"].strip()), f"project must be a non-empty string, got {cfg['project']!r}")
    need(_is_int(cfg["seed"]) and cfg["seed"] >= 0, f"seed must be a non-negative integer, got {cfg['seed']!r}")
    ing = cfg.get("ingest")
    if isinstance(ing, dict) and isinstance(ing.get("empty_drops"), bool):  # YAML reads a bare on/off as true/false
        ing["empty_drops"] = "on" if ing["empty_drops"] else "off"
    for key, allowed in ALLOWED.items():
        v = get_key(cfg, key)
        need(isinstance(v, str) and v in allowed, f"{key} must be one of {list(allowed)}, got {v!r}")
    r = cfg["resources"]
    need(_is_int(r["threads"]) and r["threads"] >= 1, f"resources.threads must be an integer >= 1, got {r['threads']!r}")
    for k in ("max_memory_gb", "gpu_mem_cap_gb"):
        need(_is_num(r[k]) and r[k] > 0, f"resources.{k} must be a positive number, got {r[k]!r}")
    need(_is_int(r["nice"]) and 0 <= r["nice"] <= 19, f"resources.nice must be an integer from 0 to 19, got {r['nice']!r}")
    need(_is_int(cfg["n_hvg"]) and cfg["n_hvg"] >= 10, f"n_hvg must be an integer >= 10, got {cfg['n_hvg']!r}")
    pcs = cfg["pcs"]
    need(pcs == "auto" or (_is_int(pcs) and 2 <= pcs <= 200), f"pcs must be 'auto' or an integer from 2 to 200, got {pcs!r}")
    need(_is_int(cfg["n_neighbors"]) and cfg["n_neighbors"] >= 2, f"n_neighbors must be an integer >= 2, got {cfg['n_neighbors']!r}")
    res = cfg["clustering"]["resolution"]
    need(_is_num(res) and res > 0, f"clustering.resolution must be a positive number, got {res!r}")
    bk = cfg["integration"]["batch_key"]
    need(isinstance(bk, str) and bool(bk.strip()), f"integration.batch_key must be a non-empty string, got {bk!r}")
    hp = cfg["integration"].get("harmony")
    if not isinstance(hp, Mapping):
        problems.append(f"integration.harmony must be a mapping (theta, lambda, sigma, nclust, max_iter), got {hp!r}")
    else:
        unknown = sorted(set(hp) - {"theta", "lambda", "sigma", "nclust", "max_iter"})
        need(not unknown, f"integration.harmony: unknown key(s) {unknown} (allowed: theta, lambda, sigma, nclust, max_iter)")
        for k in ("theta", "lambda", "sigma"):
            v = hp.get(k)
            need(v is None or (_is_num(v) and v >= 0 and (k != "sigma" or v > 0)),
                 f"integration.harmony.{k} must be a {'positive' if k == 'sigma' else 'non-negative'} number or null, got {v!r}")
        for k in ("nclust", "max_iter"):
            v = hp.get(k)
            need(v is None or (_is_int(v) and v >= 1), f"integration.harmony.{k} must be an integer >= 1 or null, got {v!r}")
    if cfg["engine"] == "scfast" and cfg["device"] == "gpu" and cfg["engine_gpu"] != "scfast":
        problems.append("engine: scfast has no GPU backend; use device: cpu or auto with engine: scfast "
                        "(or engine: seurat / auto for device: gpu; engine_gpu: scfast forces scfast on CUDA)")
    if cfg["engine"] == "seurat" and cfg["engine_gpu"] == "scfast":
        problems.append("engine_gpu: scfast needs engine: auto or scfast (engine: seurat keeps the rapids exchange on GPU)")
    if cfg["normalization"] == "sct" and cfg["device"] == "gpu":
        problems.append("normalization: sct runs on CPU only; use device: cpu or auto with sct")
    for g, m in (cfg.get("gates") or {}).items():
        need(isinstance(m, str) and m in GATE_MODES, f"gates.{g} must be one of {list(GATE_MODES)}, got {m!r}")
    _check_llm(cfg, problems)
    ann = cfg["annotation"]
    mc = ann.get("min_confidence")
    need(_is_num(mc) and 0 <= mc <= 1, f"annotation.min_confidence must be a number from 0 to 1, got {mc!r}")
    cm = ann.get("celltypist_model")
    need(isinstance(cm, str) and bool(cm.strip()), f"annotation.celltypist_model must be a non-empty string, got {cm!r}")
    md = ann.get("models_dir")
    need(md is None or (isinstance(md, str) and bool(md.strip())), f"annotation.models_dir must be a path or null, got {md!r}")
    sg = cfg["signatures"]
    need(sg is None or (isinstance(sg, str) and bool(sg.strip())), f"signatures must be a path or null, got {sg!r}")
    _check_dge(cfg, problems)
    _check_opentargets(cfg, problems)


def _check_dge(cfg: dict[str, Any], problems: list[str]) -> None:
    """Types and ranges of the ``dge`` section (sample-sheet checks are in :func:`_check_contrasts`)."""
    d = cfg["dge"]

    def need(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    for k in ("sample_col", "condition_col", "group_by"):
        need(isinstance(d[k], str) and bool(d[k].strip()), f"dge.{k} must be a non-empty string, got {d[k]!r}")
    for k in ("min_cells_per_sample", "min_replicates", "min_cells_per_side"):
        need(_is_int(d[k]) and d[k] >= 1, f"dge.{k} must be an integer >= 1, got {d[k]!r}")
    need(d["max_cells_per_side"] is None or (_is_int(d["max_cells_per_side"]) and d["max_cells_per_side"] >= 1),
         f"dge.max_cells_per_side must be an integer >= 1 or null, got {d['max_cells_per_side']!r}")
    need(_is_num(d["min_pct"]) and 0 <= d["min_pct"] <= 1, f"dge.min_pct must be a number in [0, 1], got {d['min_pct']!r}")
    need(_is_num(d["logfc_threshold"]) and d["logfc_threshold"] >= 0,
         f"dge.logfc_threshold must be a number >= 0, got {d['logfc_threshold']!r}")
    need(d["markers"] in ("auto", True, False), f"dge.markers must be auto, true or false, got {d['markers']!r}")
    need(isinstance(d["interactive"], bool), f"dge.interactive must be true or false, got {d['interactive']!r}")
    need(_is_num(d["padj"]) and 0 < d["padj"] <= 1, f"dge.padj must be a number in (0, 1], got {d['padj']!r}")
    need(_is_num(d["lfc"]) and d["lfc"] >= 0, f"dge.lfc must be a number >= 0, got {d['lfc']!r}")
    if d["contrasts"] is None:
        d["contrasts"] = []
    if not isinstance(d["contrasts"], list):
        problems.append(f"dge.contrasts must be a list of mappings, got {type(d['contrasts']).__name__}")
        return
    names: dict[str, int] = {}
    for i, c in enumerate(d["contrasts"]):
        where = f"dge.contrasts[{i}]"
        if not isinstance(c, Mapping):
            problems.append(f"{where} must be a mapping with name, numerator and denominator")
            continue
        v = c.get("name")
        need(isinstance(v, str) and bool(v.strip()), f"{where}.name must be a non-empty string, got {v!r}")
        for k in ("numerator", "denominator"):
            v = c.get(k)
            vals = v if isinstance(v, list) else [v]
            need(bool(vals) and all(isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip()
                                    for x in vals),
                 f"{where}.{k} must be a value or a list of values, got {v!r}")
        if c.get("numerator") == "rest":
            problems.append(f"{where}.numerator cannot be 'rest' (use it for the denominator)")
        for k in ("column", "within"):
            v = c.get(k)
            need(v is None or (isinstance(v, str) and bool(v.strip())), f"{where}.{k} must be a column name, got {v!r}")
        name = c.get("name")
        if isinstance(name, str) and name.strip():
            need(_CONTRAST_NAME_RE.fullmatch(name) is not None,
                 f"{where}.name {name!r} may only contain letters, digits, '.', '_' and '-' (it names output files)")
            if name in names:
                problems.append(f"dge.contrasts: name {name!r} is used twice (entries {names[name]} and {i})")
            names.setdefault(name, i)
        if isinstance(c.get("numerator"), str) and c.get("numerator") == c.get("denominator"):
            problems.append(f"{where}: numerator and denominator are both {c['numerator']!r}")
        cov = c.get("covariates")
        if cov is None:
            c["covariates"] = []
        elif not (isinstance(cov, list) and all(isinstance(x, str) and x.strip() for x in cov)):
            problems.append(f"{where}.covariates must be a list of column names, got {cov!r}")
        unknown = sorted(set(c) - {"name", "column", "numerator", "denominator", "within", "covariates"})
        if unknown:
            problems.append(f"{where}: unknown key(s) {unknown}")


def _check_opentargets(cfg: dict[str, Any], problems: list[str]) -> None:
    o = cfg["opentargets"]

    def need(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    for k in ("enabled", "offline"):
        need(isinstance(o[k], bool), f"opentargets.{k} must be true or false, got {o[k]!r}")
    need(_is_int(o["max_genes_per_direction"]) and o["max_genes_per_direction"] >= 1,
         f"opentargets.max_genes_per_direction must be an integer >= 1, got {o['max_genes_per_direction']!r}")
    need(isinstance(o["cache_dir"], str) and bool(o["cache_dir"].strip()),
         f"opentargets.cache_dir must be a non-empty path, got {o['cache_dir']!r}")
    need(_is_num(o["timeout_s"]) and o["timeout_s"] > 0, f"opentargets.timeout_s must be > 0, got {o['timeout_s']!r}")
    need(_is_int(o["retries"]) and o["retries"] >= 0, f"opentargets.retries must be an integer >= 0, got {o['retries']!r}")


def llm_settings(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    """The ``llm`` section with :data:`LLM_DEFAULTS` filled in, or ``None`` when unset (rules-only gates)."""
    raw = cfg.get("llm")
    if not raw:
        return None
    out = copy.deepcopy(LLM_DEFAULTS)
    for k, v in raw.items():
        if k in LLM_ROLE_KEYS:  # flat shorthand: llm: {provider: ..., model: ...} means llm.default
            out["default"][k] = v
        elif isinstance(out.get(k), dict) and isinstance(v, Mapping):
            out[k].update(v)
        else:
            out[k] = v
    return out


def _check_llm(cfg: dict[str, Any], problems: list[str]) -> None:
    """Validate the optional ``llm`` section (DESIGN 4.8, agents-spec section 3); the key itself is never read."""
    raw = cfg.get("llm")
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        problems.append(f"llm must be a mapping (default, roles, budget, replay), got {type(raw).__name__}")
        return
    unknown = sorted(set(raw) - {"default", "roles", "budget", "replay", "ledger_path", "script"} - set(LLM_ROLE_KEYS))
    if unknown:
        problems.append(f"llm: unknown key(s) {unknown} (allowed: default, roles, budget, replay, ledger_path)")
    flat = {k: raw[k] for k in LLM_ROLE_KEYS if k in raw}

    def check_role(where: str, s: Any, is_default: bool) -> None:
        if s is None:
            return
        if not isinstance(s, Mapping):
            problems.append(f"{where} must be a mapping (provider, model, api_key_env, ...)")
            return
        prov = s.get("provider")
        if prov is not None and (not isinstance(prov, str) or prov not in LLM_PROVIDERS):
            problems.append(f"{where}.provider must be one of {list(LLM_PROVIDERS)}, got {prov!r}")
        for k in ("model", "api_key_env", "base_url"):
            v = s.get(k)
            if v is not None and not (isinstance(v, str) and v.strip()):
                problems.append(f"{where}.{k} must be a non-empty string, got {v!r}")
        env = s.get("api_key_env")
        if isinstance(env, str) and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env):
            problems.append(f"{where}.api_key_env must be the NAME of an environment variable, got {env!r}")
        for k in ("max_tokens", "timeout_s", "max_retries"):
            v = s.get(k)
            if v is not None and not (_is_num(v) and v > 0):
                problems.append(f"{where}.{k} must be a positive number, got {v!r}")
        t = s.get("temperature")
        if t is not None and not (_is_num(t) and 0 <= t <= 2):
            problems.append(f"{where}.temperature must be a number from 0 to 2, got {t!r}")
        eb = s.get("extra_body")
        if eb is not None and not isinstance(eb, Mapping):
            problems.append(f"{where}.extra_body must be a mapping of extra request fields, got {type(eb).__name__}")

    if flat:
        check_role("llm", flat, True)
    check_role("llm.default", raw.get("default"), True)
    roles = raw.get("roles")
    if roles is not None:
        if not isinstance(roles, Mapping):
            problems.append("llm.roles must be a mapping of role name to settings")
        else:
            for r, s in roles.items():
                check_role(f"llm.roles.{r}", s, False)
    b = raw.get("budget")
    if b is not None:
        if not isinstance(b, Mapping):
            problems.append("llm.budget must be a mapping (total_limit, per_run_limit, warn_at_pct, usd_to_gbp, ledger_path)")
        else:
            for k in ("total_limit", "per_run_limit", "usd_to_gbp"):
                v = b.get(k)
                if v is not None and not (_is_num(v) and v > 0):
                    problems.append(f"llm.budget.{k} must be a positive number, got {v!r}")
            w = b.get("warn_at_pct")
            if w is not None and not (_is_num(w) and 0 < w <= 100):
                problems.append(f"llm.budget.warn_at_pct must be a number in (0, 100], got {w!r}")
            cur = b.get("currency")
            if cur is not None and cur != "GBP":
                problems.append(f"llm.budget.currency must be GBP (limits are in GBP), got {cur!r}")
            lp = b.get("ledger_path")
            if lp is not None and not (isinstance(lp, str) and lp.strip()):
                problems.append(f"llm.budget.ledger_path must be a path, got {lp!r}")
    rp = raw.get("replay")
    if rp is not None and not isinstance(rp, bool):
        problems.append(f"llm.replay must be true or false, got {rp!r}")


def _check_contrasts(cfg: dict[str, Any], header: list[str], records: list[dict[str, str]], problems: list[str]) -> None:
    """Contrast levels and covariates must exist in the sample sheet (agents-spec section 6, DESIGN 6.1)."""
    d = cfg["dge"]
    contrasts = d.get("contrasts") or []
    if not header or not isinstance(contrasts, list) or not contrasts:
        return
    cond = d["condition_col"]
    pseudobulk = d.get("method") != "findmarkers"
    scol = d["sample_col"]
    if isinstance(scol, str) and scol.strip() and scol != "sample_id" and scol not in header:
        problems.append(f"dge.sample_col {scol!r} is not a samples.csv column (columns: {header})")
    for i, c in enumerate(contrasts):
        if not isinstance(c, Mapping):
            continue
        where = f"dge.contrasts[{i}]"
        col = c.get("column") or cond
        if not isinstance(col, str) or not col.strip():
            continue
        if col not in header:
            # Cell-level columns (cell_type_broad, seurat_clusters, h5ad obs columns) only exist after the
            # earlier stages, so they are checked by S4. Pseudobulk compares samples: its column must be here.
            if pseudobulk:
                problems.append(f"{where}: column {col!r} is not a samples.csv column (columns: {header}); "
                                "pseudobulk compares samples, method findmarkers also accepts cell-level columns")
            continue
        levels = sorted({r.get(col, "") for r in records if r.get(col, "")})
        for k in ("numerator", "denominator"):
            v = c.get(k)
            for x in (v if isinstance(v, list) else [v]):
                x = str(x) if x is not None else ""
                if x.strip() and not (k == "denominator" and x == "rest") and x not in levels:
                    problems.append(f"{where}.{k} {x!r} is not a value of samples.csv column {col!r} (values: {levels})")
        for cv in c.get("covariates") or []:
            if isinstance(cv, str) and cv not in header and pseudobulk:
                problems.append(f"{where}.covariates: {cv!r} is not a samples.csv column (columns: {header})")
            elif cv == col:
                problems.append(f"{where}.covariates: {cv!r} is the contrast column itself")


def _check_portability(cfg: dict[str, Any], problems: list[str]) -> None:
    """Types of the ``container`` and ``scheduler`` sections (``execution`` is in ALLOWED)."""
    def need(cond: bool, msg: str) -> None:
        if not cond:
            problems.append(msg)

    c = cfg["container"]
    need(c.get("image") is None or (isinstance(c["image"], str) and bool(c["image"].strip())),
         f"container.image must be a path/URI or null, got {c.get('image')!r}")
    need(isinstance(c.get("options"), list) and all(isinstance(x, str) for x in c["options"]),
         f"container.options must be a list of strings, got {c.get('options')!r}")
    s = cfg["scheduler"]
    for k in ("partition", "account", "queue", "time"):
        need(s.get(k) is None or isinstance(s[k], str), f"scheduler.{k} must be a string or null, got {s.get(k)!r}")
    for k in ("cpus", "gpus"):
        need(s.get(k) is None or (_is_int(s[k]) and s[k] >= 0), f"scheduler.{k} must be an integer >= 0 or null, got {s.get(k)!r}")
    need(s.get("mem_gb") is None or (_is_num(s["mem_gb"]) and s["mem_gb"] > 0),
         f"scheduler.mem_gb must be a positive number or null, got {s.get('mem_gb')!r}")
    need(isinstance(s.get("extra"), list) and all(isinstance(x, str) for x in s["extra"]),
         f"scheduler.extra must be a list of strings (raw directive lines), got {s.get('extra')!r}")
    sc = cfg.get("scratch")
    need(sc is None or (isinstance(sc, str) and bool(sc.strip())), f"scratch must be a path or null, got {sc!r}")


def resolve_site_path(value: str, base: Path) -> Path:
    """``site: iridisx`` -> ``<repo>/conf/sites/iridisx.yaml``; a path is resolved against ``base``."""
    v = os.path.expanduser(os.path.expandvars(value))
    if "/" not in v and not v.endswith((".yaml", ".yml")):
        return REPO_ROOT / "conf" / "sites" / f"{v}.yaml"
    p = Path(v)
    if not p.is_absolute():
        p = base / p
    return Path(os.path.normpath(p))


def apply_site(raw: Mapping[str, Any], base: Path, problems: list[str], warnings: list[str]) -> tuple[dict[str, Any], str | None]:
    """Merge the site file named by ``raw['site']`` under ``raw`` (config.yaml wins per key).

    Only :data:`SITE_SECTIONS` are taken from the site file; the site file's
    own ``site`` key (its name) is kept for the record. Returns the merged raw
    mapping and the site file path (or None).
    """
    raw = dict(raw)
    site = raw.get("site")
    if site is None:
        return raw, None
    if not isinstance(site, str) or not site.strip():
        problems.append(f"site must be a site name or a path to a site YAML file, got {site!r}")
        return raw, None
    path = resolve_site_path(site, base)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as e:
        problems.append(f"site: cannot read {path}: {e}")
        return raw, None
    except yaml.YAMLError as e:
        problems.append(f"site: {path} is not valid YAML: {e}")
        return raw, None
    if not isinstance(data, Mapping):
        problems.append(f"site: {path} must hold a mapping")
        return raw, None
    for k, v in data.items():
        if k == "site":
            continue
        if k not in SITE_SECTIONS:
            warnings.append(f"site: key {k!r} in {path.name} is not a site setting (ignored; site files hold {list(SITE_SECTIONS)})")
            continue
        if isinstance(v, Mapping):
            merged = dict(v)
            user = raw.get(k)
            if isinstance(user, Mapping):
                merged.update(user)
            raw[k] = merged
        elif k not in raw:
            raw[k] = v
    return raw, str(path)


def _read_samples(path: Path, problems: list[str], warnings: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        problems.append(f"samples: cannot read {path}: {e}")
        return [], []
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        problems.append(f"samples: {path} is empty")
        return [], []
    header = [h.strip() for h in rows[0]]
    for col in ("sample_id", "path"):
        if col not in header:
            problems.append(f"samples: {path.name} lacks required column {col!r} (has {header})")
    dup_cols = sorted({h for h in header if header.count(h) > 1})
    if dup_cols:
        problems.append(f"samples: duplicated column name(s) {dup_cols}")
    if any(not h for h in header):
        problems.append("samples: empty column name in header")
    records = []
    for i, r in enumerate(rows[1:], start=2):
        if len(r) != len(header):
            problems.append(f"samples: line {i} has {len(r)} fields, header has {len(header)}")
            continue
        records.append({h: v.strip() for h, v in zip(header, r)})
    if not records:
        problems.append(f"samples: {path.name} has no sample rows")
    return header, records


def _check_samples(header, records, base: Path, problems: list[str], warnings: list[str]) -> list[dict[str, str]]:
    if "sample_id" not in header or "path" not in header:
        return []
    out = []
    seen: dict[str, int] = {}
    for i, rec in enumerate(records, start=2):
        sid = rec["sample_id"]
        if not sid:
            problems.append(f"samples: line {i}: sample_id is empty")
            continue
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", sid):
            problems.append(
                f"samples: line {i}: sample_id {sid!r} may only contain letters, digits, '.', '_' and '-' "
                "(it prefixes cell barcodes)"
            )
        if sid in seen:
            problems.append(f"samples: sample_id {sid!r} appears more than once (lines {seen[sid]} and {i})")
            continue
        seen[sid] = i
        raw_path = rec["path"]
        if not raw_path:
            problems.append(f"samples: line {i} ({sid}): path is empty")
            continue
        p = _resolve_path(raw_path, base, problems, f"samples line {i} path")
        fmt = rec.get("format", "") or ""
        if fmt and fmt not in SAMPLE_FORMATS:
            problems.append(f"samples: line {i} ({sid}): format {fmt!r} is not one of {list(SAMPLE_FORMATS)}")
        detected = None
        if not p.exists():
            problems.append(f"samples: line {i} ({sid}): path does not exist: {p}")
        else:
            detected = detect_format(p)
            if not fmt and detected is None:
                problems.append(
                    f"samples: line {i} ({sid}): cannot detect the format of {p}; set the format column "
                    f"to one of {list(SAMPLE_FORMATS)}"
                )
            elif fmt and detected and fmt != detected:
                warnings.append(f"samples: line {i} ({sid}): format {fmt!r} given, but the path looks like {detected!r}")
        out.append({"sample_id": sid, "path": str(p), "format": fmt, "format_detected": detected or ""})
    return out


def _resolve_llm(cfg: dict[str, Any], override: Mapping[str, Any] | None, warnings: list[str]) -> None:
    """Fix the run's LLM provider now, so it is recorded and part of the gates' fingerprints.

    No ``llm:`` section and no ``--llm``: use the first key found (Claude, OpenAI, NVIDIA), or
    rules-only gates when none is set. ``provider: auto`` behaves the same; ``--llm none`` turns
    the agents off. ``cfg["llm_choice"]`` says what was picked and why.
    """
    ov = {k: v for k, v in (override or {}).items() if v}
    if ov.get("provider") == "none":
        cfg["llm"], cfg["llm_choice"] = None, "none (--llm none): gates run rules-only"
        return
    raw = cfg.get("llm")
    keys = ", ".join("$" + LLM_PRESETS[p]["api_key_env"] for p in LLM_PRESETS)
    if raw is None and not ov:
        if auto_provider() is None:
            cfg["llm_choice"] = f"none: no API key set ({keys}); gates run rules-only"
            return
        raw = {}
    raw = copy.deepcopy(dict(raw or {}))
    d = dict(raw.get("default") or {})
    for k in LLM_ROLE_KEYS:  # fold the flat shorthand into default
        if k in raw and k not in d:
            d[k] = raw.pop(k)
    d.update(ov)
    prov = d.get("provider") or "auto"
    why = "set in the config" + (" / --llm" if ov else "")
    if prov == "auto":
        prov = auto_provider()
        if prov is None:
            cfg["llm"], cfg["llm_choice"] = None, f"none: provider auto but no API key set ({keys}); gates run rules-only"
            warnings.append(f"llm: provider auto found no API key ({keys}); the gates run rules-only")
            return
        why = f"auto: ${LLM_PRESETS[prov]['api_key_env']} is set"
    d["provider"] = prov
    raw["default"] = d
    cfg["llm"] = raw
    cfg["llm_choice"] = f"{prov} ({why})"


def load_config(config_path: str | Path, run_dir: str | Path | None = None,
                llm_override: Mapping[str, Any] | None = None,
                overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load and validate a run configuration; returns the resolved config dict.

    ``overrides`` (command-line options such as ``--integrate``) are merged into the file's
    settings, one level deep for sections (``{"integration": {"method": "harmony"}}``).

    Raises :class:`ConfigError` listing every problem. Non-fatal remarks are
    returned under ``cfg["_warnings"]`` (not written to the resolved file).
    """
    problems: list[str] = []
    warnings: list[str] = []
    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config_path = Path(os.path.normpath(config_path))
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"cannot read config file {config_path}: {e}") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"config file {config_path} is not valid YAML: {e}") from e
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"config file {config_path} must hold a mapping, got {type(raw).__name__}")
    raw = dict(raw)
    for k, v in (overrides or {}).items():
        raw[k] = {**raw[k], **v} if isinstance(v, Mapping) and isinstance(raw.get(k), Mapping) else v
    if isinstance(raw.get("gates"), Mapping):  # gates: {qc-review: auto} means gates: {G1: auto}
        from .stage_names import gate_id
        raw["gates"] = {gate_id(str(g)): m for g, m in raw["gates"].items()}
    base = config_path.parent
    raw, site_source = apply_site(raw, base, problems, warnings)
    cfg = _merge_defaults(raw, problems, warnings)
    if site_source:
        cfg["site_source"] = site_source
    _check_values(cfg, problems)
    _check_portability(cfg, problems)

    # run_dir
    if run_dir is not None:
        rd = _resolve_path(str(run_dir), Path.cwd(), problems, "--run-dir")
    elif isinstance(cfg["run_dir"], str) and cfg["run_dir"].strip():
        rd = _resolve_path(cfg["run_dir"], base, problems, "run_dir")
    else:
        rd = None
        problems.append("run_dir is not set (set it in config.yaml or pass --run-dir)")
    cfg["run_dir"] = str(rd) if rd is not None else None

    # samples
    samples_path = None
    if not isinstance(cfg["samples"], str) or not cfg["samples"].strip():
        problems.append("samples is not set (path to samples.csv)")
    else:
        samples_path = _resolve_path(cfg["samples"], base, problems, "samples")
        if not samples_path.is_file():
            problems.append(f"samples: file not found: {samples_path}")
            samples_path = None
    header: list[str] = []
    resolved_samples: list[dict[str, str]] = []
    if samples_path is not None:
        header, records = _read_samples(samples_path, problems, warnings)
        resolved_samples = _check_samples(header, records, base, problems, warnings)
    bk = cfg["integration"]["batch_key"]
    if cfg["integration"]["method"] == "harmony" and header and isinstance(bk, str) and bk not in header:
        problems.append(
            f"integration.batch_key {bk!r} is not a samples.csv column (columns: {header})"
        )
    if cfg["integration"]["method"] == "harmony" and resolved_samples and len(resolved_samples) < 2:
        warnings.append("integration.method is harmony but there is only one sample")
    if samples_path is not None:
        _check_contrasts(cfg, header, records, problems)
    # Open Targets cache: shared across runs, so it lives outside the run dir (relative to config.yaml).
    oc = cfg["opentargets"].get("cache_dir")
    if isinstance(oc, str) and oc.strip():
        cfg["opentargets"]["cache_dir"] = str(_resolve_path(oc, base, problems, "opentargets.cache_dir"))
    # S3: signature override must exist; the CellTypist model cache is shared across runs (relative to config.yaml).
    sg = cfg.get("signatures")
    if isinstance(sg, str) and sg.strip():
        sp = _resolve_path(sg, base, problems, "signatures")
        if not sp.is_file():
            problems.append(f"signatures: file not found: {sp}")
        cfg["signatures"] = str(sp)
    md = (cfg.get("annotation") or {}).get("models_dir")
    if isinstance(md, str) and md.strip():
        cfg["annotation"]["models_dir"] = str(_resolve_path(md, base, problems, "annotation.models_dir"))

    # qc_policy
    policy_path = None
    policy = None
    if cfg["qc_policy"] is not None:
        if not isinstance(cfg["qc_policy"], str) or not cfg["qc_policy"].strip():
            problems.append(f"qc_policy must be a path or null, got {cfg['qc_policy']!r}")
        else:
            policy_path = _resolve_path(cfg["qc_policy"], base, problems, "qc_policy")
            if not policy_path.is_file():
                problems.append(f"qc_policy: file not found: {policy_path}")
                policy_path = None
    try:
        policy = qcmod.load_policy(policy_path) if policy_path else qcmod.policy_from_dict(None)
    except qcmod.QCError as e:
        problems.extend(f"qc_policy: {p}" for p in e.problems)

    if problems:
        raise ConfigError(problems)

    from . import __version__

    cfg["config_file"] = str(config_path)
    cfg["config_dir"] = str(base)
    cfg["samples_source"] = str(samples_path)
    cfg["qc_policy_source"] = str(policy_path) if policy_path else "builtin-default"
    cfg["samples"] = str(Path(rd) / "config" / "samples.csv")
    cfg["qc_policy"] = str(Path(rd) / "config" / "qc_policy.yaml")
    cfg["samples_resolved"] = resolved_samples
    cfg["sample_columns"] = header
    cfg["policy_hash"] = policy.hash
    cfg["repo_root"] = str(REPO_ROOT)
    cfg["cellwarden_version"] = __version__
    _resolve_llm(cfg, llm_override, warnings)
    cfg["_warnings"] = warnings
    cfg["_policy"] = policy
    return cfg


def _policy_yaml(policy: qcmod.Policy) -> str:
    return "# Built-in default QC policy (docs/stages-spec.md section 3.4)\n" + yaml.safe_dump(
        policy.to_dict(), sort_keys=False
    )


def _copy(src: Path, dst: Path) -> None:
    if dst.exists() and src.resolve() == dst.resolve():
        return
    shutil.copyfile(src, dst)


def _new_run_id(project: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", project).strip("-") or "run"
    return f"{slug}-{stamp}-{secrets.token_hex(3)}"


def config_hash(cfg: Mapping[str, Any], samples_sha: str, policy_sha: str) -> str:
    body = {k: v for k, v in cfg.items() if not k.startswith("_") and k not in _HASH_EXCLUDE}
    body["samples_sha256"] = samples_sha
    body["qc_policy_sha256"] = policy_sha
    return sha256_bytes(canonical_json(body).encode("utf-8"))


def prepare_run_dir(cfg: dict[str, Any]) -> dict[str, Any]:
    """Create ``<run-dir>/config/`` and write the copies, resolved config and hashes.

    ``run_id`` is kept from an existing ``config.resolved.yaml`` in the same
    run directory (so it is stable across reruns), otherwise newly generated.
    Returns the resolved config as written (without ``_``-prefixed keys).
    """
    rd = Path(cfg["run_dir"])
    cdir = rd / "config"
    try:
        cdir.mkdir(parents=True, exist_ok=True)
        (rd / "logs").mkdir(exist_ok=True)
    except OSError as e:
        raise ConfigError(f"cannot create run directory {rd}: {e}") from e

    run_id = None
    prev = cdir / "config.resolved.yaml"
    if prev.exists():
        try:
            run_id = (yaml.safe_load(prev.read_text()) or {}).get("run_id")
        except Exception:
            run_id = None
    run_id = run_id or _new_run_id(cfg["project"])

    _copy(Path(cfg["config_file"]), cdir / "config.yaml")
    _copy(Path(cfg["samples_source"]), cdir / "samples.csv")
    if cfg["qc_policy_source"] == "builtin-default":
        (cdir / "qc_policy.yaml").write_text(_policy_yaml(cfg["_policy"]), encoding="utf-8")
    else:
        _copy(Path(cfg["qc_policy_source"]), cdir / "qc_policy.yaml")

    samples_sha = sha256_file(cdir / "samples.csv")
    policy_sha = sha256_file(cdir / "qc_policy.yaml")
    out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    out["run_id"] = run_id
    out["config_hash"] = config_hash(out, samples_sha, policy_sha)

    header = (
        "# Written by cellwarden: defaults filled in, paths absolute, validated.\n"
        "# Stages read this file, never the user's config.yaml.\n"
    )
    tmp = cdir / "config.resolved.yaml.tmp"
    tmp.write_text(header + yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8")
    os.replace(tmp, cdir / "config.resolved.yaml")

    hashes = {
        name: sha256_file(cdir / name)
        for name in ("config.yaml", "samples.csv", "qc_policy.yaml", "config.resolved.yaml")
    }
    hashes["config_hash"] = out["config_hash"]
    hashes["policy_hash"] = out["policy_hash"]
    (cdir / "hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return out


def load_resolved(run_dir: str | Path) -> dict[str, Any]:
    """Read ``<run-dir>/config/config.resolved.yaml``."""
    p = Path(run_dir) / "config" / "config.resolved.yaml"
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"{p} not found or unreadable ({e}); run 'cellwarden run --config ...' first") from e
    if not isinstance(cfg, dict) or "run_id" not in cfg:
        raise ConfigError(f"{p} is not a resolved cellwarden config")
    return cfg
