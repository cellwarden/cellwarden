"""Rules-based QC engine (docs/stages-spec.md section 3.4).

Pure, deterministic functions on a per-cell metrics table (one row per cell,
the columns of ``S1/qc_metrics.tsv.gz``). S1c calls them; the QC agent in
build step 3 reuses them unchanged.

Public API::

    load_policy(path) -> Policy
    compute_thresholds(metrics, policy) -> Thresholds          # per sample
    simulate_filter(metrics, thresholds, policy) -> SimulationReport
    check_protected(metrics, thresholds, policy) -> list[ProtectedLoss]
    decide(metrics, thresholds, policy) -> Decisions

Conventions (documented choices where the spec leaves room):

* **Metric columns.** ``log_genes`` = log10(``n_genes``), ``log_umis`` =
  log10(``n_umis``) (log10 of 0 is -inf), ``mito_pct`` = ``pct_mito``
  (percent, linear scale, upper bound only).
* **MAD** is the *scaled* MAD, ``1.4826 * median(|x - median(x)|)``, so it
  estimates the standard deviation for normal data (as scater/`isOutlier`).
  Bounds are ``median +/- n_mads * MAD``. Medians and MADs are computed per
  sample over all of the sample's cells (ignoring NaN), before any filtering.
  If a metric's MAD is 0 (e.g. no mitochondrial reads at all) that adaptive
  bound is not applied, and the thresholds record why.
* **Small samples.** A sample with fewer than ``min_cells_for_mad`` cells
  gets hard limits only (no adaptive bounds).
* **Comparisons are strict:** a cell fails ``min_genes: 200`` only with
  fewer than 200 genes, and ``max_mito_pct: 25`` only above 25 %. A missing
  (NaN) value fails the corresponding hard limit; adaptive rules ignore it.
* **Rule 4/5 basis.** An *adaptive-only exclusion* is a cell flagged by at
  least one adaptive rule and by no hard limit and no doublet call. Only
  those can be turned into ``review``.
* **Review triggers** (rule 5) are evaluated on the cells that would be
  excluded after rules 1-4 (protected-population cells already moved to
  ``review`` do not count as lost). Both triggers use the same baseline, so
  their order does not change which ones fire. "More than" is strict.
* **Reason codes** list every rule that fired, in rule order:
  hard (``LOW_GENES``, ``LOW_UMIS``, ``HIGH_MT``), adaptive
  (``ADAPT_LOW_GENES``, ``ADAPT_HIGH_GENES``, ``ADAPT_LOW_UMIS``,
  ``ADAPT_HIGH_UMIS``, ``ADAPT_HIGH_MT``), ``DOUBLET``,
  ``PROTECTED_POP:<name>`` (one per qualifying population, policy order),
  ``SAMPLE_LOSS``, ``CLUSTER_LOSS``. A ``review`` cell keeps the adaptive
  codes that excluded it, followed by the code(s) that rescued it.
* **Defaults.** Missing policy keys take the spec values; an omitted
  ``protected_populations`` list means none (see :func:`policy_from_dict`).
* **Policy hash** = SHA-256 of the canonical JSON (sorted keys, no
  whitespace) of the parsed policy with defaults filled in.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

__all__ = [
    "MAD_SCALE",
    "QCError",
    "ProtectedPopulation",
    "Policy",
    "Bound",
    "SampleThresholds",
    "Thresholds",
    "SimulationReport",
    "ProtectedLoss",
    "Decisions",
    "load_policy",
    "policy_from_dict",
    "compute_thresholds",
    "simulate_filter",
    "check_protected",
    "decide",
    "read_metrics",
    "HARD_CODES",
    "ADAPT_CODES",
]

MAD_SCALE = 1.4826

HARD_CODES = ("LOW_GENES", "LOW_UMIS", "HIGH_MT")
ADAPT_CODES = ("ADAPT_LOW_GENES", "ADAPT_HIGH_GENES", "ADAPT_LOW_UMIS", "ADAPT_HIGH_UMIS", "ADAPT_HIGH_MT")
DOUBLET = "DOUBLET"
PROTECTED_PREFIX = "PROTECTED_POP:"
SAMPLE_LOSS = "SAMPLE_LOSS"
CLUSTER_LOSS = "CLUSTER_LOSS"
ADAPTIVE_METRICS = ("log_genes", "log_umis", "mito_pct")
REQUIRED_COLUMNS = ("cell_id", "sample_id", "n_genes", "n_umis", "pct_mito")

DEFAULT_POLICY: dict[str, Any] = {
    "hard_limits": {"min_genes": 200, "max_mito_pct": 25, "min_umis": 400},
    "adaptive": {"n_mads": 3, "metrics": ["log_genes", "log_umis", "mito_pct"], "min_cells_for_mad": 100},
    "doublets": {"exclude_called": True},
    "protected_populations": [
        {"name": "Platelet", "markers": ["PPBP", "PF4", "GNG11"], "min_fraction": 0.01},
        {"name": "Erythrocyte", "markers": ["HBB", "HBA1", "HBA2"], "min_fraction": 0.05},
    ],
    "review_triggers": {"max_cells_removed_per_sample_pct": 30, "max_cluster_removed_pct": 50},
}
DEFAULT_MIN_FRACTION = 0.01


class QCError(ValueError):
    """Invalid policy or metrics table. ``problems`` lists every problem found."""

    def __init__(self, problems: list[str] | str):
        if isinstance(problems, str):
            problems = [problems]
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtectedPopulation:
    name: str
    markers: tuple[str, ...]
    min_fraction: float

    @property
    def column(self) -> str:
        return f"protected_{self.name}"


@dataclass(frozen=True)
class Policy:
    min_genes: float | None
    min_umis: float | None
    max_mito_pct: float | None
    n_mads: float
    adaptive_metrics: tuple[str, ...]
    min_cells_for_mad: int
    exclude_doublets: bool
    protected_populations: tuple[ProtectedPopulation, ...]
    max_cells_removed_per_sample_pct: float
    max_cluster_removed_pct: float
    parsed: Mapping[str, Any] = field(repr=False, compare=False, hash=False)
    hash: str = ""
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.parsed))


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _num(x: float) -> int | float:
    """Canonical number: integral floats become ints, so 25 and 25.0 hash alike."""
    f = float(x)
    return int(f) if f.is_integer() else f


def policy_from_dict(raw: Mapping[str, Any] | None, source: str | None = None) -> Policy:
    """Validate a parsed policy mapping, fill defaults and build a :class:`Policy`.

    ``None`` or an empty mapping gives the built-in default policy (the spec
    example, section 3.4). Otherwise missing sections and keys take the spec
    defaults, except ``protected_populations``: a list is replaced, never
    merged, so leaving it out means no protected populations (S1a in R reads
    the policy file the same way). A key set to ``null`` in ``hard_limits``
    disables that limit. Unknown keys are
    errors, so a typo never silently disables a rule. All problems are
    reported together in one :class:`QCError`.
    """
    raw = {} if raw is None else raw
    problems: list[str] = []
    if not isinstance(raw, Mapping):
        raise QCError(f"policy must be a mapping, got {type(raw).__name__}")
    known_sections = set(DEFAULT_POLICY)
    for k in raw:
        if k not in known_sections:
            problems.append(f"unknown policy section {k!r} (allowed: {sorted(known_sections)})")

    def section(name: str) -> dict[str, Any]:
        val = raw.get(name, None)
        if val is None:
            return dict(DEFAULT_POLICY[name])
        if not isinstance(val, Mapping):
            problems.append(f"{name} must be a mapping")
            return dict(DEFAULT_POLICY[name])
        for k in val:
            if k not in DEFAULT_POLICY[name]:
                problems.append(f"unknown key {name}.{k} (allowed: {sorted(DEFAULT_POLICY[name])})")
        out = dict(DEFAULT_POLICY[name])
        out.update({k: v for k, v in val.items() if k in DEFAULT_POLICY[name]})
        return out

    hard = section("hard_limits")
    for k in ("min_genes", "min_umis", "max_mito_pct"):
        v = hard[k]
        if v is None:
            continue
        if not _is_number(v) or float(v) < 0:
            problems.append(f"hard_limits.{k} must be a non-negative number or null, got {v!r}")
            hard[k] = None
        else:
            hard[k] = _num(v)
    if hard["max_mito_pct"] is not None and float(hard["max_mito_pct"]) > 100:
        problems.append(f"hard_limits.max_mito_pct must be <= 100, got {hard['max_mito_pct']!r}")

    adaptive = section("adaptive")
    if not _is_number(adaptive["n_mads"]) or float(adaptive["n_mads"]) <= 0:
        problems.append(f"adaptive.n_mads must be a positive number, got {adaptive['n_mads']!r}")
        adaptive["n_mads"] = DEFAULT_POLICY["adaptive"]["n_mads"]
    adaptive["n_mads"] = _num(adaptive["n_mads"])
    mets = adaptive["metrics"]
    if mets is None:
        mets = []
    if not isinstance(mets, (list, tuple)) or not all(isinstance(m, str) for m in mets):
        problems.append(f"adaptive.metrics must be a list of names, got {mets!r}")
        mets = []
    bad = [m for m in mets if m not in ADAPTIVE_METRICS]
    if bad:
        problems.append(f"adaptive.metrics has unknown metric(s) {bad} (allowed: {list(ADAPTIVE_METRICS)})")
    if len(set(mets)) != len(mets):
        problems.append("adaptive.metrics has duplicates")
    # Canonical order (the order of the rules), so list order never changes the hash.
    adaptive["metrics"] = [m for m in ADAPTIVE_METRICS if m in mets]
    mc = adaptive["min_cells_for_mad"]
    if isinstance(mc, bool) or not isinstance(mc, int) or mc < 1:
        if _is_number(mc) and float(mc).is_integer() and float(mc) >= 1:
            adaptive["min_cells_for_mad"] = int(mc)
        else:
            problems.append(f"adaptive.min_cells_for_mad must be a positive integer, got {mc!r}")
            adaptive["min_cells_for_mad"] = DEFAULT_POLICY["adaptive"]["min_cells_for_mad"]

    doublets = section("doublets")
    if not isinstance(doublets["exclude_called"], bool):
        problems.append(f"doublets.exclude_called must be true or false, got {doublets['exclude_called']!r}")
        doublets["exclude_called"] = True

    triggers = section("review_triggers")
    for k in ("max_cells_removed_per_sample_pct", "max_cluster_removed_pct"):
        v = triggers[k]
        if not _is_number(v) or not (0 <= float(v) <= 100):
            problems.append(f"review_triggers.{k} must be a number between 0 and 100, got {v!r}")
            triggers[k] = DEFAULT_POLICY["review_triggers"][k]
        triggers[k] = _num(triggers[k])

    # A list is replaced, not merged: an omitted (or null) protected_populations
    # means none, as in S1a (R). The built-in default policy, used when the
    # config sets no qc_policy at all, does list the spec's two populations.
    pops_raw = raw.get("protected_populations", None) if raw else DEFAULT_POLICY["protected_populations"]
    if pops_raw is None:
        pops_raw = []
    pops: list[dict[str, Any]] = []
    if not isinstance(pops_raw, (list, tuple)):
        problems.append("protected_populations must be a list")
        pops_raw = []
    seen: set[str] = set()
    for i, p in enumerate(pops_raw):
        where = f"protected_populations[{i}]"
        if not isinstance(p, Mapping):
            problems.append(f"{where} must be a mapping with name, markers, min_fraction")
            continue
        for k in p:
            if k not in ("name", "markers", "min_fraction"):
                problems.append(f"unknown key {where}.{k} (allowed: ['markers', 'min_fraction', 'name'])")
        name = p.get("name")
        if not isinstance(name, str) or not name.strip():
            problems.append(f"{where}.name must be a non-empty string")
            continue
        if name in seen:
            problems.append(f"protected population name {name!r} is duplicated")
        seen.add(name)
        markers = p.get("markers")
        if not isinstance(markers, (list, tuple)) or not markers or not all(isinstance(m, str) and m for m in markers):
            problems.append(f"{where}.markers must be a non-empty list of gene names")
            markers = []
        mf = p.get("min_fraction", DEFAULT_MIN_FRACTION)
        if not _is_number(mf) or not (0 < float(mf) <= 1):
            problems.append(f"{where}.min_fraction must be a number in (0, 1], got {mf!r}")
            mf = DEFAULT_MIN_FRACTION
        pops.append({"name": name, "markers": list(markers), "min_fraction": _num(mf)})

    if problems:
        raise QCError(problems)

    parsed = {
        "hard_limits": hard,
        "adaptive": adaptive,
        "doublets": doublets,
        "protected_populations": pops,
        "review_triggers": triggers,
    }
    h = hashlib.sha256(_canonical_json(parsed).encode("utf-8")).hexdigest()
    return Policy(
        min_genes=hard["min_genes"],
        min_umis=hard["min_umis"],
        max_mito_pct=hard["max_mito_pct"],
        n_mads=float(adaptive["n_mads"]),
        adaptive_metrics=tuple(adaptive["metrics"]),
        min_cells_for_mad=int(adaptive["min_cells_for_mad"]),
        exclude_doublets=bool(doublets["exclude_called"]),
        protected_populations=tuple(
            ProtectedPopulation(p["name"], tuple(p["markers"]), float(p["min_fraction"])) for p in pops
        ),
        max_cells_removed_per_sample_pct=float(triggers["max_cells_removed_per_sample_pct"]),
        max_cluster_removed_pct=float(triggers["max_cluster_removed_pct"]),
        parsed=parsed,
        hash=h,
        source=source,
    )


def load_policy(path: str | Path) -> Policy:
    """Read ``qc_policy.yaml`` and return a validated :class:`Policy`."""
    import yaml

    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise QCError(f"cannot read QC policy {path}: {e}") from e
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise QCError(f"QC policy {path} is not valid YAML: {e}") from e
    return policy_from_dict(raw, source=str(path))


# --------------------------------------------------------------------------
# Metrics table
# --------------------------------------------------------------------------

_TRUE = {"true", "t", "1", "yes"}
_FALSE = {"false", "f", "0", "no", "", "na", "nan", "none"}


def _as_bool(s: pd.Series) -> np.ndarray:
    """Bool array from a bool/number/string column (TRUE/FALSE/True/1/...); missing -> False."""
    if pd.api.types.is_bool_dtype(s.dtype):
        return s.fillna(False).to_numpy(dtype=bool)
    if pd.api.types.is_numeric_dtype(s.dtype):
        return (s.fillna(0).to_numpy() != 0)
    vals = s.astype(object).where(s.notna(), "").map(lambda v: str(v).strip().lower())
    bad = sorted(set(vals) - _TRUE - _FALSE)
    if bad:
        raise QCError(f"column {s.name!r} has non-boolean values, e.g. {bad[:5]}")
    return vals.isin(_TRUE).to_numpy(dtype=bool)


def read_metrics(path: str | Path) -> pd.DataFrame:
    """Read a ``qc_metrics.tsv(.gz)`` table with ids as strings and metrics as floats."""
    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"cell_id": str, "sample_id": str, "pre_cluster": str},
        keep_default_na=True,
        na_values=["NA", "NaN", ""],
    )
    return df


def _validate_metrics(metrics: pd.DataFrame, policy: Policy) -> None:
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in metrics.columns]
    if missing:
        problems.append(f"metrics table lacks required column(s) {missing}")
    for p in policy.protected_populations:
        if p.column not in metrics.columns:
            problems.append(
                f"metrics table lacks {p.column!r} for protected population {p.name!r} "
                "(S1a must be run with the same qc_policy)"
            )
    if "cell_id" in metrics.columns and not metrics["cell_id"].is_unique:
        dups = metrics["cell_id"][metrics["cell_id"].duplicated()].unique()[:5].tolist()
        problems.append(f"cell_id values are not unique, e.g. {dups}")
    if "sample_id" in metrics.columns and metrics["sample_id"].isna().any():
        problems.append("sample_id has missing values")
    if problems:
        raise QCError(problems)


def _metric_values(metrics: pd.DataFrame, name: str) -> np.ndarray:
    if name == "log_genes":
        x = pd.to_numeric(metrics["n_genes"], errors="coerce").to_numpy(dtype=float)
    elif name == "log_umis":
        x = pd.to_numeric(metrics["n_umis"], errors="coerce").to_numpy(dtype=float)
    else:
        return pd.to_numeric(metrics["pct_mito"], errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(x)  # log10(0) = -inf; negative/NaN -> NaN


def _samples(metrics: pd.DataFrame) -> np.ndarray:
    return metrics["sample_id"].astype(str).to_numpy()


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------


@dataclass
class Bound:
    """Adaptive bound for one metric of one sample (on the metric's own scale)."""

    median: float | None
    mad: float | None
    lower: float | None
    upper: float | None
    applied: bool
    note: str = ""


@dataclass
class SampleThresholds:
    sample_id: str
    n_cells: int
    adaptive_applied: bool
    note: str
    bounds: dict[str, Bound]  # keys: log_genes, log_umis, mito_pct (only applied ones are used)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "sample_id": self.sample_id,
            "n_cells": self.n_cells,
            "adaptive_applied": self.adaptive_applied,
            "note": self.note,
            "bounds": {k: asdict(v) for k, v in sorted(self.bounds.items())},
        }
        # Natural-scale equivalents for readers of the summary.
        nat: dict[str, Any] = {}
        for k, unit in (("log_genes", "genes"), ("log_umis", "umis")):
            b = self.bounds.get(k)
            if b is not None and b.applied:
                nat[f"min_{unit}"] = None if b.lower is None else float(10 ** b.lower)
                nat[f"max_{unit}"] = None if b.upper is None else float(10 ** b.upper)
        b = self.bounds.get("mito_pct")
        if b is not None and b.applied:
            nat["max_mito_pct"] = b.upper
        d["adaptive_natural_scale"] = nat
        return d


@dataclass
class Thresholds:
    """Hard limits (global) and adaptive bounds (per sample)."""

    min_genes: float | None
    min_umis: float | None
    max_mito_pct: float | None
    n_mads: float
    per_sample: dict[str, SampleThresholds]
    policy_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_limits": {"min_genes": self.min_genes, "min_umis": self.min_umis, "max_mito_pct": self.max_mito_pct},
            "n_mads": self.n_mads,
            "mad_scale": MAD_SCALE,
            "policy_hash": self.policy_hash,
            "per_sample": {s: t.to_dict() for s, t in sorted(self.per_sample.items())},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Thresholds":
        """Rebuild thresholds (e.g. ones proposed by the QC agent) from :meth:`to_dict` output."""
        hl = d["hard_limits"]
        per = {}
        for s, t in d["per_sample"].items():
            bounds = {}
            for k, b in t["bounds"].items():
                bounds[k] = Bound(
                    median=b.get("median"),
                    mad=b.get("mad"),
                    lower=b.get("lower"),
                    upper=b.get("upper"),
                    applied=bool(b.get("applied", True)),
                    note=b.get("note", ""),
                )
            per[str(s)] = SampleThresholds(
                sample_id=str(s),
                n_cells=int(t.get("n_cells", 0)),
                adaptive_applied=bool(t.get("adaptive_applied", True)),
                note=t.get("note", ""),
                bounds=bounds,
            )
        return cls(
            min_genes=hl.get("min_genes"),
            min_umis=hl.get("min_umis"),
            max_mito_pct=hl.get("max_mito_pct"),
            n_mads=float(d.get("n_mads", 3)),
            per_sample=per,
            policy_hash=str(d.get("policy_hash", "")),
        )


def _robust(x: np.ndarray) -> tuple[float | None, float | None]:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return None, None
    med = float(np.median(x))
    if not math.isfinite(med):
        return None, None
    dev = np.abs(x - med)
    mad = float(np.median(dev)) * MAD_SCALE
    return med, (mad if math.isfinite(mad) else None)


def compute_thresholds(metrics: pd.DataFrame, policy: Policy) -> Thresholds:
    """Per-sample adaptive bounds (median +/- n_mads x scaled MAD) plus the hard limits."""
    _validate_metrics(metrics, policy)
    samples = _samples(metrics)
    values = {m: _metric_values(metrics, m) for m in ADAPTIVE_METRICS}
    per: dict[str, SampleThresholds] = {}
    for s in sorted(set(samples)):
        idx = samples == s
        n = int(idx.sum())
        bounds: dict[str, Bound] = {}
        if n < policy.min_cells_for_mad:
            note = f"{n} cells < min_cells_for_mad ({policy.min_cells_for_mad}): hard limits only"
            per[s] = SampleThresholds(s, n, False, note, bounds)
            continue
        for m in policy.adaptive_metrics:
            med, mad = _robust(values[m][idx])
            if med is None or mad is None:
                bounds[m] = Bound(med, mad, None, None, False, "no finite values")
                continue
            if mad == 0:
                bounds[m] = Bound(med, mad, None, None, False, "MAD is 0: bound not applied")
                continue
            lo = med - policy.n_mads * mad
            hi = med + policy.n_mads * mad
            if m == "mito_pct":
                lo = None  # upper bound only
            bounds[m] = Bound(med, mad, lo, hi, True)
        applied = any(b.applied for b in bounds.values())
        per[s] = SampleThresholds(s, n, applied, "" if applied else "no adaptive bound applicable", bounds)
    return Thresholds(
        min_genes=policy.min_genes,
        min_umis=policy.min_umis,
        max_mito_pct=policy.max_mito_pct,
        n_mads=policy.n_mads,
        per_sample=per,
        policy_hash=policy.hash,
    )


# --------------------------------------------------------------------------
# Rule evaluation (shared by simulate_filter, check_protected and decide)
# --------------------------------------------------------------------------


@dataclass
class _Eval:
    flags: dict[str, np.ndarray]  # code -> bool per row (hard, adaptive, DOUBLET)
    hard_any: np.ndarray
    adapt_any: np.ndarray
    doublet: np.ndarray
    excluded_rules: np.ndarray  # rules 1-3
    adapt_only: np.ndarray
    protected: dict[str, np.ndarray]  # population name -> qualifies (fraction >= min_fraction)
    protected_any: np.ndarray
    protect_review: np.ndarray  # rule 4
    excluded: np.ndarray  # after rule 4 (basis of the review triggers)
    samples: np.ndarray
    clusters: np.ndarray | None


def _evaluate(metrics: pd.DataFrame, thresholds: Thresholds, policy: Policy) -> _Eval:
    _validate_metrics(metrics, policy)
    n = len(metrics)
    samples = _samples(metrics)
    genes = pd.to_numeric(metrics["n_genes"], errors="coerce").to_numpy(dtype=float)
    umis = pd.to_numeric(metrics["n_umis"], errors="coerce").to_numpy(dtype=float)
    mito = pd.to_numeric(metrics["pct_mito"], errors="coerce").to_numpy(dtype=float)

    flags: dict[str, np.ndarray] = {}
    # 1. Hard limits; NaN fails.
    flags["LOW_GENES"] = (
        (np.isnan(genes) | (genes < thresholds.min_genes)) if thresholds.min_genes is not None else np.zeros(n, bool)
    )
    flags["LOW_UMIS"] = (
        (np.isnan(umis) | (umis < thresholds.min_umis)) if thresholds.min_umis is not None else np.zeros(n, bool)
    )
    flags["HIGH_MT"] = (
        (np.isnan(mito) | (mito > thresholds.max_mito_pct))
        if thresholds.max_mito_pct is not None
        else np.zeros(n, bool)
    )

    # 2. Adaptive bounds, per sample; NaN never fails an adaptive rule.
    vals = {m: _metric_values(metrics, m) for m in ADAPTIVE_METRICS}
    lo = {m: np.full(n, -np.inf) for m in ADAPTIVE_METRICS}
    hi = {m: np.full(n, np.inf) for m in ADAPTIVE_METRICS}
    for s, t in thresholds.per_sample.items():
        idx = samples == s
        if not idx.any() or not t.adaptive_applied:
            continue
        for m, b in t.bounds.items():
            if m not in ADAPTIVE_METRICS or not b.applied:
                continue
            if b.lower is not None and m != "mito_pct":
                lo[m][idx] = b.lower
            if b.upper is not None:
                hi[m][idx] = b.upper
    with np.errstate(invalid="ignore"):
        flags["ADAPT_LOW_GENES"] = vals["log_genes"] < lo["log_genes"]
        flags["ADAPT_HIGH_GENES"] = vals["log_genes"] > hi["log_genes"]
        flags["ADAPT_LOW_UMIS"] = vals["log_umis"] < lo["log_umis"]
        flags["ADAPT_HIGH_UMIS"] = vals["log_umis"] > hi["log_umis"]
        flags["ADAPT_HIGH_MT"] = vals["mito_pct"] > hi["mito_pct"]

    # 3. Doublets.
    if policy.exclude_doublets and "doublet_call" in metrics.columns:
        doublet = _as_bool(metrics["doublet_call"])
    else:
        doublet = np.zeros(n, bool)
    flags[DOUBLET] = doublet

    hard_any = flags["LOW_GENES"] | flags["LOW_UMIS"] | flags["HIGH_MT"]
    adapt_any = np.zeros(n, bool)
    for c in ADAPT_CODES:
        adapt_any |= flags[c]
    excluded_rules = hard_any | adapt_any | doublet
    adapt_only = adapt_any & ~hard_any & ~doublet

    # 4. Protected populations.
    protected: dict[str, np.ndarray] = {}
    protected_any = np.zeros(n, bool)
    for p in policy.protected_populations:
        frac = pd.to_numeric(metrics[p.column], errors="coerce").to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            q = frac >= p.min_fraction
        protected[p.name] = q
        protected_any |= q
    protect_review = adapt_only & protected_any
    excluded = excluded_rules & ~protect_review

    clusters = None
    if "pre_cluster" in metrics.columns:
        pc = metrics["pre_cluster"]
        clusters = pc.astype(object).where(pc.notna(), None).map(lambda v: None if v is None else str(v)).to_numpy()
        clusters = np.array([c if c not in (None, "", "nan", "NA") else None for c in clusters], dtype=object)
    return _Eval(
        flags, hard_any, adapt_any, doublet, excluded_rules, adapt_only, protected, protected_any,
        protect_review, excluded, samples, clusters,
    )


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


@dataclass
class SimulationReport:
    """Cells that would be removed, per sample and per pre-cluster, without applying anything.

    ``per_sample`` columns: n_cells, one column per rule code (cells flagged by
    that rule), n_fail_hard, n_fail_adaptive_only, n_fail_doublet,
    n_excluded_rules (rules 1-3), n_protected_review (rule 4), n_excluded
    (after rule 4) and pct_excluded. ``per_cluster`` (None without a
    ``pre_cluster`` column): n_cells, n_excluded, pct_excluded. The trigger
    lists name the samples / clusters whose pct_excluded exceeds the policy
    limits.
    """

    per_sample: pd.DataFrame
    per_cluster: pd.DataFrame | None
    sample_loss_triggered: list[str]
    cluster_loss_triggered: list[str]
    max_cells_removed_per_sample_pct: float
    max_cluster_removed_pct: float

    def to_dict(self) -> dict[str, Any]:
        def frame(df: pd.DataFrame | None) -> dict[str, Any] | None:
            if df is None:
                return None
            return {str(k): {c: _py(v) for c, v in row.items()} for k, row in df.iterrows()}

        return {
            "per_sample": frame(self.per_sample),
            "per_cluster": frame(self.per_cluster),
            "sample_loss_triggered": list(self.sample_loss_triggered),
            "cluster_loss_triggered": list(self.cluster_loss_triggered),
            "max_cells_removed_per_sample_pct": self.max_cells_removed_per_sample_pct,
            "max_cluster_removed_pct": self.max_cluster_removed_pct,
        }


def _py(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _pct(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, 100.0 * num / np.maximum(den, 1), 0.0)


def _simulate(ev: _Eval, policy: Policy) -> SimulationReport:
    order = sorted(set(ev.samples))
    rows = []
    for s in order:
        idx = ev.samples == s
        r: dict[str, Any] = {"sample_id": s, "n_cells": int(idx.sum())}
        for c in HARD_CODES + ADAPT_CODES + (DOUBLET,):
            r[c] = int(ev.flags[c][idx].sum())
        r["n_fail_hard"] = int(ev.hard_any[idx].sum())
        r["n_fail_adaptive_only"] = int(ev.adapt_only[idx].sum())
        r["n_fail_doublet"] = int((ev.doublet & ~ev.hard_any)[idx].sum())
        r["n_excluded_rules"] = int(ev.excluded_rules[idx].sum())
        r["n_protected_review"] = int(ev.protect_review[idx].sum())
        r["n_excluded"] = int(ev.excluded[idx].sum())
        rows.append(r)
    per_sample = pd.DataFrame(rows).set_index("sample_id") if rows else pd.DataFrame()
    if rows:
        per_sample["pct_excluded"] = _pct(per_sample["n_excluded"].to_numpy(), per_sample["n_cells"].to_numpy())
    s_trig = [str(s) for s in per_sample.index if per_sample.loc[s, "pct_excluded"] > policy.max_cells_removed_per_sample_pct] if rows else []

    per_cluster = None
    c_trig: list[str] = []
    if ev.clusters is not None:
        valid = np.array([c is not None for c in ev.clusters], dtype=bool)
        labels = ev.clusters[valid].astype(str)
        exc = ev.excluded[valid]
        crow = []
        for c in _natural_sort(set(labels)):
            idx = labels == c
            crow.append({"pre_cluster": c, "n_cells": int(idx.sum()), "n_excluded": int(exc[idx].sum())})
        per_cluster = pd.DataFrame(crow, columns=["pre_cluster", "n_cells", "n_excluded"]).set_index("pre_cluster")
        per_cluster["pct_excluded"] = _pct(per_cluster["n_excluded"].to_numpy(), per_cluster["n_cells"].to_numpy())
        c_trig = [str(c) for c in per_cluster.index if per_cluster.loc[c, "pct_excluded"] > policy.max_cluster_removed_pct]
    return SimulationReport(
        per_sample, per_cluster, s_trig, c_trig,
        policy.max_cells_removed_per_sample_pct, policy.max_cluster_removed_pct,
    )


def _natural_sort(items) -> list[str]:
    def key(s: str):
        return (0, int(s), s) if s.lstrip("-").isdigit() else (1, 0, s)

    return sorted(items, key=key)


def simulate_filter(metrics: pd.DataFrame, thresholds: Thresholds, policy: Policy) -> SimulationReport:
    """Cells removed per sample and per pre-cluster under ``thresholds``, without deciding anything."""
    return _simulate(_evaluate(metrics, thresholds, policy), policy)


# --------------------------------------------------------------------------
# Protected populations
# --------------------------------------------------------------------------


@dataclass
class ProtectedLoss:
    """How a protected population fares under the thresholds.

    n_protected: cells with ``protected_<name>`` >= min_fraction.
    n_flagged: of those, cells failing any rule 1-3 (what would be lost
    without protection). n_rescued: adaptive-only failures moved to
    ``review`` by rule 4. n_lost: flagged by a hard limit or doublet call, so
    excluded even though protected. ``would_be_lost`` is true when every
    protected cell would be removed without rule 4.
    """

    name: str
    min_fraction: float
    n_protected: int
    n_flagged: int
    n_rescued: int
    n_lost: int
    pct_flagged: float
    pct_lost: float
    would_be_lost: bool
    per_sample: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _protected(ev: _Eval, policy: Policy) -> list[ProtectedLoss]:
    out = []
    for p in policy.protected_populations:
        q = ev.protected[p.name]
        flagged = q & ev.excluded_rules
        rescued = q & ev.protect_review
        lost = flagged & ~rescued
        per = {}
        for s in sorted(set(ev.samples)):
            idx = ev.samples == s
            per[s] = {
                "n_protected": int((q & idx).sum()),
                "n_flagged": int((flagged & idx).sum()),
                "n_rescued": int((rescued & idx).sum()),
                "n_lost": int((lost & idx).sum()),
            }
        n_prot = int(q.sum())
        n_flag = int(flagged.sum())
        n_lost = int(lost.sum())
        out.append(
            ProtectedLoss(
                name=p.name,
                min_fraction=p.min_fraction,
                n_protected=n_prot,
                n_flagged=n_flag,
                n_rescued=int(rescued.sum()),
                n_lost=n_lost,
                pct_flagged=float(100.0 * n_flag / n_prot) if n_prot else 0.0,
                pct_lost=float(100.0 * n_lost / n_prot) if n_prot else 0.0,
                would_be_lost=bool(n_prot > 0 and n_flag == n_prot),
                per_sample=per,
            )
        )
    return out


def check_protected(metrics: pd.DataFrame, thresholds: Thresholds, policy: Policy) -> list[ProtectedLoss]:
    """One :class:`ProtectedLoss` per protected population (policy order)."""
    return _protected(_evaluate(metrics, thresholds, policy), policy)


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------


@dataclass
class Decisions:
    """Per-cell decisions plus everything needed for ``qc_summary.json``.

    ``table`` columns (row order = metrics row order): cell_id, sample_id,
    decision (keep / exclude / review), reasons (``;``-separated codes, empty
    for keep), policy_hash, decided_by.
    """

    table: pd.DataFrame
    policy_hash: str
    thresholds: Thresholds
    simulation: SimulationReport
    protected: list[ProtectedLoss]
    triggered: list[dict[str, Any]]

    def counts_per_sample(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        t = self.table
        for s in sorted(t["sample_id"].astype(str).unique()):
            d = t.loc[t["sample_id"].astype(str) == s, "decision"].value_counts()
            out[s] = {k: int(d.get(k, 0)) for k in ("keep", "exclude", "review")}
        return out

    def counts_total(self) -> dict[str, int]:
        d = self.table["decision"].value_counts()
        return {k: int(d.get(k, 0)) for k in ("keep", "exclude", "review")}

    def counts_per_reason(self, by_sample: bool = False) -> dict[str, Any]:
        t = self.table
        codes = t["reasons"].str.split(";")
        ex = pd.DataFrame({"sample_id": t["sample_id"].astype(str), "code": codes}).explode("code")
        ex = ex[ex["code"].notna() & (ex["code"] != "")]
        if not by_sample:
            vc = ex["code"].value_counts()
            return {k: int(vc[k]) for k in sorted(vc.index)}
        out: dict[str, dict[str, int]] = {}
        for s in sorted(t["sample_id"].astype(str).unique()):
            vc = ex.loc[ex["sample_id"] == s, "code"].value_counts()
            out[s] = {k: int(vc[k]) for k in sorted(vc.index)}
        return out

    def summary(self) -> dict[str, Any]:
        """The ``qc_summary.json`` content (minus run provenance)."""
        return {
            "policy_hash": self.policy_hash,
            "decided_by": "rules",
            "n_cells": int(len(self.table)),
            "counts": self.counts_total(),
            "counts_per_sample": self.counts_per_sample(),
            "counts_per_reason": self.counts_per_reason(),
            "counts_per_reason_per_sample": self.counts_per_reason(by_sample=True),
            "thresholds": self.thresholds.to_dict(),
            "protected_populations": [p.to_dict() for p in self.protected],
            "triggered_review_rules": self.triggered,
            "simulation": self.simulation.to_dict(),
        }


def decide(metrics: pd.DataFrame, thresholds: Thresholds, policy: Policy) -> Decisions:
    """Apply rules 1-6 of spec section 3.4 and return keep / exclude / review per cell."""
    ev = _evaluate(metrics, thresholds, policy)
    sim = _simulate(ev, policy)
    prot = _protected(ev, policy)
    n = len(metrics)

    s_trig = set(sim.sample_loss_triggered)
    c_trig = set(sim.cluster_loss_triggered)
    sample_review = ev.adapt_only & np.isin(ev.samples, list(s_trig)) if s_trig else np.zeros(n, bool)
    if c_trig and ev.clusters is not None:
        cluster_review = ev.adapt_only & np.array([c in c_trig for c in ev.clusters], dtype=bool)
    else:
        cluster_review = np.zeros(n, bool)
    review = ev.protect_review | sample_review | cluster_review

    decision = np.full(n, "keep", dtype=object)
    decision[ev.excluded_rules] = "exclude"
    decision[review] = "review"

    reasons = np.full(n, "", dtype=object)
    for code in HARD_CODES + ADAPT_CODES + (DOUBLET,):
        f = ev.flags[code]
        if f.any():
            reasons[f] = reasons[f] + (code + ";")
    for p in policy.protected_populations:
        f = ev.protect_review & ev.protected[p.name]
        if f.any():
            reasons[f] = reasons[f] + (PROTECTED_PREFIX + p.name + ";")
    for code, f in ((SAMPLE_LOSS, sample_review), (CLUSTER_LOSS, cluster_review)):
        if f.any():
            reasons[f] = reasons[f] + (code + ";")
    reasons = np.array([r[:-1] if r else "" for r in reasons], dtype=object)

    table = pd.DataFrame(
        {
            "cell_id": metrics["cell_id"].astype(str).to_numpy(),
            "sample_id": metrics["sample_id"].astype(str).to_numpy(),
            "decision": decision,
            "reasons": reasons,
            "policy_hash": policy.hash,
            "decided_by": "rules",
        }
    )

    triggered: list[dict[str, Any]] = []
    for s in sim.sample_loss_triggered:
        triggered.append(
            {
                "rule": SAMPLE_LOSS,
                "sample_id": s,
                "pct_excluded": float(sim.per_sample.loc[s, "pct_excluded"]),
                "limit_pct": policy.max_cells_removed_per_sample_pct,
                "n_cells_to_review": int((sample_review & (ev.samples == s)).sum()),
            }
        )
    for c in sim.cluster_loss_triggered:
        triggered.append(
            {
                "rule": CLUSTER_LOSS,
                "pre_cluster": c,
                "pct_excluded": float(sim.per_cluster.loc[c, "pct_excluded"]),
                "limit_pct": policy.max_cluster_removed_pct,
                "n_cells_to_review": int(
                    (cluster_review & np.array([x == c for x in ev.clusters], dtype=bool)).sum()
                ),
            }
        )
    return Decisions(table, policy.hash, thresholds, sim, prot, triggered)
