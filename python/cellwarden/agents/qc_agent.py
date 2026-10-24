"""QC worker agent for gate G1 and the code around its proposal (DESIGN 4.1).

Two parts:

:class:`QCContext`
    Everything the gate and the agent need in memory: the per-cell metrics
    table, the policy, the baseline thresholds and decisions from the rules
    engine (:mod:`cellwarden.qc`, called unchanged). It turns a *proposal*
    (natural-scale thresholds, see below) into a :class:`~cellwarden.qc.Thresholds`
    object, simulates it, validates it against the policy, and applies it
    with :func:`cellwarden.qc.decide`. Nothing here writes files.

:class:`QCWorker`
    The agent: prompt ``prompts/qc_worker.md`` plus the five tools of DESIGN
    4.1 (``get_qc_summary``, ``simulate_filter``, ``check_protected``,
    ``flag_for_review``, ``propose_decision``), each a thin wrapper over the
    context that validates its inputs. ``propose_decision`` is terminal; an
    invalid proposal is returned to the model as an error with the
    violations, and after :data:`MAX_INVALID_PROPOSALS` rejections the tool
    refuses further attempts, so the gate falls back to the baseline.

Proposal shape (``proposal.json`` -> ``proposal`` for G1)::

    {"thresholds": {"hard_limits": {"min_genes", "min_umis", "max_mito_pct"},   # all optional
                    "n_mads": float,
                    "per_sample": {sample_id: {"min_genes", "max_genes", "min_umis",
                                               "max_umis", "max_mito_pct"}}},
     "review_cells": [cell_id, ...],
     "summary": {"per_sample": {sample_id: {"keep", "exclude", "review"}}, "total": {...}}}

Thresholds are on the natural scale (genes, UMIs, percent). Omitted keys keep
the baseline value; an empty ``thresholds`` object is the baseline itself.

Policy checks (``policy_check.violations``; the spec says "outside the hard
limits" and "violates review_triggers", resolved as follows):

* ``hard_limits`` may only be as strict as or stricter than the policy's.
* ``n_mads`` in ``[1, 10]``.
* Per-sample bounds must name known samples, be ordered (min < max) and sit
  within the (proposed) hard limits.
* The proposal must not trigger a ``SAMPLE_LOSS`` / ``CLUSTER_LOSS`` review
  rule that the baseline did not already trigger.
* ``review_cells`` must be known, unique cell ids (at most
  :data:`MAX_REVIEW_CELLS`).

Applying a decision: the rules engine runs with the proposal's thresholds
(``decide``), then every cell in ``review_cells`` becomes ``review`` with the
reason code ``AGENT_REVIEW`` appended; ``decided_by`` is ``agent`` or
``human``; ``policy_hash`` stays the run's policy hash.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .. import qc
from .base import Agent, Tool, wrap_data

__all__ = ["MAX_REVIEW_CELLS", "AGENT_REVIEW", "QCContext", "QCWorker"]


MAX_REVIEW_CELLS = 100_000
AGENT_REVIEW = "AGENT_REVIEW"
HARD_KEYS = ("min_genes", "min_umis", "max_mito_pct")
SAMPLE_KEYS = ("min_genes", "max_genes", "min_umis", "max_umis", "max_mito_pct")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _num(x: Any, what: str, problems: list[str], lo: float = 0.0, hi: float | None = None) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        problems.append(f"{what} must be a finite number, got {x!r}")
        return None
    v = float(x)
    if v < lo or (hi is not None and v > hi):
        problems.append(f"{what} must be between {lo:g} and {'inf' if hi is None else f'{hi:g}'}, got {v:g}")
        return None
    return v


@dataclass
class QCContext:
    """Metrics, policy and baseline for one run; see the module docstring."""

    metrics: pd.DataFrame
    policy: qc.Policy
    baseline_thresholds: qc.Thresholds = field(init=False)
    baseline: qc.Decisions = field(init=False)

    def __post_init__(self) -> None:
        self.baseline_thresholds = qc.compute_thresholds(self.metrics, self.policy)
        self.baseline = qc.decide(self.metrics, self.baseline_thresholds, self.policy)
        self._cells = set(self.metrics["cell_id"].astype(str))
        self._samples = sorted(self.metrics["sample_id"].astype(str).unique())
        self._baseline_triggers = {(t["rule"], t.get("sample_id") or t.get("pre_cluster")) for t in self.baseline.triggered}

    @property
    def samples(self) -> list[str]:
        return list(self._samples)

    # -- bundle ------------------------------------------------------------

    def metric_distributions(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        m = self.metrics
        sids = m["sample_id"].astype(str)
        for s in self._samples:
            sub = m[sids == s]
            d: dict[str, Any] = {"n_cells": int(len(sub))}
            for col in ("n_genes", "n_umis", "pct_mito"):
                x = pd.to_numeric(sub[col], errors="coerce").dropna()
                d[col] = {f"q{int(q * 100):02d}": (round(float(x.quantile(q)), 3) if len(x) else None) for q in QUANTILES}
            if "doublet_call" in sub.columns:
                try:
                    d["doublet_rate_pct"] = round(100.0 * float(qc._as_bool(sub["doublet_call"]).mean()), 2) if len(sub) else 0.0
                except qc.QCError:
                    d["doublet_rate_pct"] = None
            if "pre_cluster" in sub.columns:
                vc = sub["pre_cluster"].astype(str).value_counts()
                d["pre_clusters"] = {str(k): int(v) for k, v in vc.items() if k not in ("nan", "None", "")}
            out[s] = d
        return out

    def bundle(self) -> dict[str, Any]:
        """Aggregated evidence for the gate: no per-cell data, no free text from the sample sheet."""
        b = self.baseline
        return {
            "gate": "G1",
            "n_cells": int(len(self.metrics)),
            "samples": self.samples,
            "policy": self.policy.to_dict(),
            "policy_hash": self.policy.hash,
            "baseline": {
                "thresholds": self.baseline_thresholds.to_dict(),
                "counts": b.counts_total(),
                "counts_per_sample": b.counts_per_sample(),
                "counts_per_reason_per_sample": b.counts_per_reason(by_sample=True),
                "protected_populations": [p.to_dict() for p in b.protected],
                "triggered_review_rules": b.triggered,
                "simulation": b.simulation.to_dict(),
            },
            "metrics": self.metric_distributions(),
        }

    # -- proposal -> thresholds --------------------------------------------

    def parse_thresholds(self, raw: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
        """Normalise a proposal's ``thresholds`` and list shape/policy violations."""
        problems: list[str] = []
        raw = raw or {}
        if not isinstance(raw, Mapping):
            return {}, ["thresholds must be an object"]
        for k in raw:
            if k not in ("hard_limits", "n_mads", "per_sample"):
                problems.append(f"unknown thresholds key {k!r} (allowed: hard_limits, n_mads, per_sample)")
        pol = self.policy
        hard: dict[str, float | None] = {"min_genes": pol.min_genes, "min_umis": pol.min_umis, "max_mito_pct": pol.max_mito_pct}
        hl = raw.get("hard_limits") or {}
        if not isinstance(hl, Mapping):
            problems.append("hard_limits must be an object")
            hl = {}
        for k, v in hl.items():
            if k not in HARD_KEYS:
                problems.append(f"unknown hard_limits key {k!r}")
                continue
            val = _num(v, f"hard_limits.{k}", problems, 0, 100 if k == "max_mito_pct" else None)
            if val is None:
                continue
            base = hard[k]
            if k == "max_mito_pct":
                if base is not None and val > base:
                    problems.append(f"hard_limits.max_mito_pct {val:g} is above the policy's hard limit {base:g}")
            elif base is not None and val < base:
                problems.append(f"hard_limits.{k} {val:g} is below the policy's hard limit {base:g}")
            hard[k] = val
        n_mads = pol.n_mads
        if "n_mads" in raw and raw["n_mads"] is not None:
            v = _num(raw["n_mads"], "n_mads", problems, 1.0, 10.0)
            if v is not None:
                n_mads = v
        per: dict[str, dict[str, float]] = {}
        ps = raw.get("per_sample") or {}
        if not isinstance(ps, Mapping):
            problems.append("per_sample must be an object keyed by sample_id")
            ps = {}
        for sid, ov in ps.items():
            sid = str(sid)
            if sid not in self._samples:
                problems.append(f"per_sample: unknown sample {sid!r} (samples: {self._samples})")
                continue
            if not isinstance(ov, Mapping):
                problems.append(f"per_sample.{sid} must be an object")
                continue
            d: dict[str, float] = {}
            for k, v in ov.items():
                if k not in SAMPLE_KEYS:
                    problems.append(f"per_sample.{sid}: unknown key {k!r} (allowed: {list(SAMPLE_KEYS)})")
                    continue
                if v is None:
                    continue
                val = _num(v, f"per_sample.{sid}.{k}", problems, 0, 100 if k == "max_mito_pct" else None)
                if val is not None:
                    d[k] = val
            if "min_genes" in d and hard["min_genes"] is not None and d["min_genes"] < hard["min_genes"]:
                problems.append(f"per_sample.{sid}.min_genes {d['min_genes']:g} is below the hard limit {hard['min_genes']:g}")
            if "min_umis" in d and hard["min_umis"] is not None and d["min_umis"] < hard["min_umis"]:
                problems.append(f"per_sample.{sid}.min_umis {d['min_umis']:g} is below the hard limit {hard['min_umis']:g}")
            if "max_mito_pct" in d and hard["max_mito_pct"] is not None and d["max_mito_pct"] > hard["max_mito_pct"]:
                problems.append(f"per_sample.{sid}.max_mito_pct {d['max_mito_pct']:g} is above the hard limit {hard['max_mito_pct']:g}")
            for lo_k, hi_k in (("min_genes", "max_genes"), ("min_umis", "max_umis")):
                if lo_k in d and hi_k in d and d[lo_k] >= d[hi_k]:
                    problems.append(f"per_sample.{sid}: {lo_k} {d[lo_k]:g} must be below {hi_k} {d[hi_k]:g}")
            if d:
                per[sid] = d
        norm = {"hard_limits": {k: hard[k] for k in HARD_KEYS}, "n_mads": n_mads, "per_sample": per}
        return norm, problems

    def thresholds_for(self, norm: Mapping[str, Any]) -> qc.Thresholds:
        """Build engine thresholds from a normalised proposal (see :meth:`parse_thresholds`)."""
        parsed = self.policy.to_dict()
        parsed["hard_limits"] = dict(norm["hard_limits"])
        parsed["adaptive"] = dict(parsed["adaptive"], n_mads=norm["n_mads"])
        pol2 = qc.policy_from_dict(parsed)
        thr = qc.compute_thresholds(self.metrics, pol2)
        for sid, ov in (norm.get("per_sample") or {}).items():
            t = thr.per_sample[sid]
            for metric, lo_k, hi_k, log in (("log_genes", "min_genes", "max_genes", True),
                                            ("log_umis", "min_umis", "max_umis", True),
                                            ("mito_pct", None, "max_mito_pct", False)):
                if not ((lo_k and lo_k in ov) or (hi_k in ov)):
                    continue
                b = t.bounds.get(metric) or qc.Bound(None, None, None, None, False, "")
                lo, hi = b.lower, b.upper
                if lo_k and lo_k in ov:
                    lo = math.log10(ov[lo_k]) if log and ov[lo_k] > 0 else (-math.inf if log else ov[lo_k])
                if hi_k in ov:
                    hi = math.log10(ov[hi_k]) if log and ov[hi_k] > 0 else ov[hi_k]
                t.bounds[metric] = qc.Bound(b.median, b.mad, lo, hi, True, "set by the QC agent proposal")
            t.adaptive_applied = True
            t.note = (t.note + "; " if t.note else "") + "per-sample bounds from the proposal"
        thr.policy_hash = self.policy.hash
        return thr

    def simulate(self, thresholds: Mapping[str, Any] | None) -> dict[str, Any]:
        """Simulation of candidate thresholds: losses per sample / cluster, protected losses, policy check."""
        norm, problems = self.parse_thresholds(thresholds)
        thr = self.thresholds_for(norm)
        sim = qc.simulate_filter(self.metrics, thr, self.policy)
        prot = qc.check_protected(self.metrics, thr, self.policy)
        dec = qc.decide(self.metrics, thr, self.policy)
        new_triggers = self._new_triggers(dec)
        violations = problems + [f"new review trigger {r} on {w} (not triggered by the baseline)" for r, w in new_triggers]
        return {
            "thresholds": norm,
            "effective_thresholds": thr.to_dict(),
            "counts": dec.counts_total(),
            "counts_per_sample": dec.counts_per_sample(),
            "baseline_counts_per_sample": self.baseline.counts_per_sample(),
            "simulation": sim.to_dict(),
            "protected_populations": [p.to_dict() for p in prot],
            "triggered_review_rules": dec.triggered,
            "policy_check": {"ok": not violations, "violations": violations},
        }

    def _new_triggers(self, dec: qc.Decisions) -> list[tuple[str, str]]:
        out = []
        for t in dec.triggered:
            key = (t["rule"], t.get("sample_id") or t.get("pre_cluster"))
            if key not in self._baseline_triggers:
                out.append(key)
        return out

    def validate_review_cells(self, cells: Any) -> tuple[list[str], list[str]]:
        problems: list[str] = []
        if cells is None:
            return [], []
        if not isinstance(cells, (list, tuple)):
            return [], ["review_cells must be a list of cell ids"]
        ids = [str(c) for c in cells]
        if len(ids) > MAX_REVIEW_CELLS:
            problems.append(f"review_cells lists {len(ids)} cells (limit {MAX_REVIEW_CELLS})")
        unknown = [c for c in ids if c not in self._cells]
        if unknown:
            problems.append(f"review_cells: {len(unknown)} unknown cell id(s), e.g. {unknown[:3]}")
        return sorted(set(ids)), problems

    def validate(self, proposal: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Normalise a full proposal and return ``(normalised, violations)``."""
        if not isinstance(proposal, Mapping):
            return {}, ["proposal must be an object with thresholds and review_cells"]
        norm, problems = self.parse_thresholds(proposal.get("thresholds"))
        cells, p2 = self.validate_review_cells(proposal.get("review_cells"))
        problems += p2
        for k in proposal:
            if k not in ("thresholds", "review_cells", "summary"):
                problems.append(f"unknown proposal key {k!r}")
        out = {"thresholds": norm, "review_cells": cells, "summary": {}}
        if not problems:
            dec = self.apply(out)
            for r, w in self._new_triggers(dec):
                problems.append(f"new review trigger {r} on {w} (not triggered by the baseline)")
            out["summary"] = self.summary(dec)
        return out, problems

    # -- apply --------------------------------------------------------------

    def apply(self, norm: Mapping[str, Any], decided_by: str = "agent") -> qc.Decisions:
        """Run the rules engine with the proposal's thresholds and add the review flags."""
        thr = self.thresholds_for(norm["thresholds"])
        dec = qc.decide(self.metrics, thr, self.policy)
        cells = set(norm.get("review_cells") or [])
        if cells:
            t = dec.table
            mask = t["cell_id"].astype(str).isin(cells).to_numpy()
            t.loc[mask, "decision"] = "review"
            r = t.loc[mask, "reasons"].astype(str)
            t.loc[mask, "reasons"] = np.where(r == "", AGENT_REVIEW, r + ";" + AGENT_REVIEW)
        dec.table["decided_by"] = decided_by
        dec.table["policy_hash"] = self.policy.hash
        return dec

    @staticmethod
    def summary(dec: qc.Decisions) -> dict[str, Any]:
        return {"per_sample": dec.counts_per_sample(), "total": dec.counts_total()}

    def baseline_proposal(self) -> dict[str, Any]:
        """The baseline as a proposal: empty thresholds (= policy), no review flags."""
        norm, _ = self.parse_thresholds({})
        return {"thresholds": norm, "review_cells": [], "summary": self.summary(self.baseline)}

    def is_baseline(self, norm: Mapping[str, Any]) -> bool:
        base = self.baseline_proposal()
        return (norm.get("thresholds") == base["thresholds"]) and not norm.get("review_cells")


# --------------------------------------------------------------------------
# The worker agent
# --------------------------------------------------------------------------

_THRESHOLDS_SCHEMA = {
    "type": "object",
    "description": "Candidate thresholds, natural scale; omitted keys keep the baseline.",
    "properties": {
        "hard_limits": {"type": "object", "properties": {k: {"type": "number"} for k in HARD_KEYS}},
        "n_mads": {"type": "number"},
        "per_sample": {"type": "object",
                       "additionalProperties": {"type": "object", "properties": {k: {"type": "number"} for k in SAMPLE_KEYS}}},
    },
}


class QCWorker(Agent):
    role = "qc_worker"
    gate = "G1"

    def __init__(self, ctx: QCContext, provider, ledger=None, **kw):
        self.ctx = ctx
        self.flags: list[dict[str, Any]] = []
        self.flagged: set[str] = set()
        self.proposal: dict[str, Any] | None = None
        self.rationale: str = ""
        super().__init__(provider, ledger, tools=self._tools(), **kw)

    def user_message(self, bundle: Mapping[str, Any]) -> str:
        return ("Evidence bundle for gate G1 (QC). Inspect it, simulate if needed, then call propose_decision.\n\n"
                + wrap_data(json.dumps(bundle, indent=1, default=str)))

    def _tools(self) -> list[Tool]:
        return [
            Tool("get_qc_summary", "The evidence bundle (policy, baseline thresholds and counts, simulated losses, "
                 "metric distributions), or one sample's part of it.",
                 {"type": "object", "properties": {"sample_id": {"type": "string"}}}, self.get_qc_summary),
            Tool("simulate_filter", "Cells removed per sample and per pre-cluster under candidate thresholds, with "
                 "the policy check. Applies nothing.",
                 {"type": "object", "properties": {"thresholds": _THRESHOLDS_SCHEMA}, "required": ["thresholds"]},
                 self.simulate_filter),
            Tool("check_protected", "Whether a protected population would be lost under candidate thresholds.",
                 {"type": "object", "properties": {"thresholds": _THRESHOLDS_SCHEMA}, "required": ["thresholds"]},
                 self.check_protected),
            Tool("flag_for_review", "Send cells (by id) or a whole pre-cluster (optionally within one sample) to the "
                 "human review list, with a reason.",
                 {"type": "object", "properties": {"reason": {"type": "string"}, "cell_ids": {"type": "array", "items": {"type": "string"}},
                                                   "pre_cluster": {"type": "string"}, "sample_id": {"type": "string"}},
                  "required": ["reason"]}, self.flag_for_review),
            Tool("propose_decision", "Final answer: the thresholds to apply and a short rationale. Rejected (with the "
                 "violations) if outside the policy's hard limits or if it newly triggers a review rule.",
                 {"type": "object", "properties": {"thresholds": _THRESHOLDS_SCHEMA, "rationale": {"type": "string"}},
                  "required": ["thresholds", "rationale"]}, self.propose_decision, terminal=True),
        ]

    # -- tools --------------------------------------------------------------

    def get_qc_summary(self, sample_id: str | None = None) -> dict[str, Any]:
        b = self.ctx.bundle()
        if sample_id is None:
            return b
        sid = str(sample_id)
        if sid not in self.ctx.samples:
            raise ValueError(f"unknown sample {sid!r}; samples: {self.ctx.samples}")
        base = b["baseline"]
        return {
            "sample_id": sid,
            "thresholds": base["thresholds"]["per_sample"].get(sid),
            "hard_limits": base["thresholds"]["hard_limits"],
            "counts": base["counts_per_sample"].get(sid),
            "counts_per_reason": base["counts_per_reason_per_sample"].get(sid),
            "simulation": base["simulation"]["per_sample"].get(sid),
            "metrics": b["metrics"].get(sid),
        }

    def simulate_filter(self, thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
        r = self.ctx.simulate(thresholds or {})
        r.pop("effective_thresholds", None)
        return r

    def check_protected(self, thresholds: Mapping[str, Any] | None = None) -> dict[str, Any]:
        r = self.ctx.simulate(thresholds or {})
        return {"protected_populations": r["protected_populations"], "policy_check": r["policy_check"]}

    def flag_for_review(self, reason: str, cell_ids: list[str] | None = None, pre_cluster: str | None = None,
                        sample_id: str | None = None) -> dict[str, Any]:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        m = self.ctx.metrics
        ids: list[str] = []
        if cell_ids:
            ids, problems = self.ctx.validate_review_cells(cell_ids)
            if problems:
                raise ValueError("; ".join(problems))
        if pre_cluster is not None:
            if "pre_cluster" not in m.columns:
                raise ValueError("the metrics table has no pre_cluster column")
            mask = m["pre_cluster"].astype(str) == str(pre_cluster)
            if sample_id is not None:
                mask &= m["sample_id"].astype(str) == str(sample_id)
            if not mask.any():
                raise ValueError(f"no cells in pre_cluster {pre_cluster!r}" + (f" of sample {sample_id!r}" if sample_id else ""))
            ids += m.loc[mask, "cell_id"].astype(str).tolist()
        if not ids:
            raise ValueError("give cell_ids or a pre_cluster")
        new = [c for c in ids if c not in self.flagged]
        if len(self.flagged) + len(new) > MAX_REVIEW_CELLS:
            raise ValueError(f"more than {MAX_REVIEW_CELLS} cells would be flagged for review")
        self.flagged.update(new)
        self.flags.append({"reason": reason.strip(), "n_cells": len(new), "pre_cluster": pre_cluster, "sample_id": sample_id})
        return {"flagged_now": len(new), "flagged_total": len(self.flagged)}

    def propose_decision(self, thresholds: Mapping[str, Any] | None = None, rationale: str = "") -> dict[str, Any]:
        """The proposal tool: ``accepted`` with the normalised proposal, or the violations (one retry)."""
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be a non-empty string")
        norm, violations = self.ctx.validate({"thresholds": thresholds or {}, "review_cells": sorted(self.flagged)})
        if violations:
            return {"accepted": False, "violations": violations, "proposal": norm, "rationale": rationale.strip(),
                    "hint": "fix the violations and call propose_decision again (one retry)"}
        self.proposal = norm
        self.rationale = rationale.strip()
        return {"accepted": True, "proposal": norm, "rationale": self.rationale,
                "summary": norm["summary"], "n_review_cells": len(norm["review_cells"])}
