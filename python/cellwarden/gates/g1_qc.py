"""Gate G1: QC decisions (DESIGN 4.1, agents-spec section 2).

    python -m cellwarden.gates.g1_qc --run-dir DIR

Runs after S1c and before S2. Inputs: ``S1/qc_metrics.tsv.gz``,
``S1/qc_decisions.tsv.gz`` (S1c's rules-only decisions, the *baseline*),
``S1/qc_summary.json`` and ``config/qc_policy.yaml``. Outputs: the gate
files in ``gates/G1/`` and, when the decision differs from the baseline, a
rewritten ``S1/qc_decisions.tsv.gz``.

Baseline bookkeeping
--------------------
S1c's file is kept as ``S1/qc_decisions.baseline.tsv.gz``: when
``qc_decisions.tsv.gz`` holds only ``decided_by == rules`` rows (a fresh S1c
output) it is copied there; otherwise it is a previous G1 result and the
existing baseline copy is used. The baseline in memory is recomputed with
the rules engine (:mod:`cellwarden.qc`, unchanged) from the metrics and the
policy, which is exactly what S1c did; a mismatch with the file is logged.

Applying a decision
-------------------
Only ever through the rules engine: :meth:`QCContext.apply` runs
``qc.decide`` with the accepted thresholds and then marks the accepted
``review_cells`` as ``review`` (``AGENT_REVIEW``). ``decided_by`` in the
table is ``agent`` or ``human``; ``policy_hash`` stays the run's. A decision
equal to the baseline restores the baseline copy (no new table is computed)
so a rerun is byte-identical. ``gates/G1/qc_summary.json`` holds the applied
decision's counts (S1c's ``S1/qc_summary.json`` is left untouched: it is an
S1c output with a recorded checksum).

The proposal / decision content shape is documented in
:mod:`cellwarden.agents.qc_agent`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from .. import qc
from ..agents.qc_agent import QCContext, QCWorker
from ..stages import read_tsv_str, utc_now, write_json, write_tsv_gz
from .protocol import GateError, gate_main
from .report import pre, table

__all__ = ["GATE", "WORKER_ROLE", "G1Gate", "make_gate", "main"]

GATE = "G1"
WORKER_ROLE = "qc_worker"
BASELINE_NAME = "qc_decisions.baseline.tsv.gz"


class G1Gate:
    """The gate object for G1 (interface in :mod:`cellwarden.gates.protocol`)."""

    gate_id = GATE
    worker_role = WORKER_ROLE

    def __init__(self, run_dir: Path, cfg: Mapping[str, Any]):
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        s1 = self.run_dir / "S1"
        self.metrics_path = s1 / "qc_metrics.tsv.gz"
        self.decisions_path = s1 / "qc_decisions.tsv.gz"
        self.baseline_path = s1 / BASELINE_NAME
        policy_path = Path(cfg.get("qc_policy") or self.run_dir / "config" / "qc_policy.yaml")
        for p in (self.metrics_path, self.decisions_path):
            if not p.exists():
                raise GateError(f"input {p} not found (S1a-S1c must run first)")
        try:
            self.policy_obj = qc.load_policy(policy_path)
            metrics = qc.read_metrics(self.metrics_path)
            self.ctx = QCContext(metrics, self.policy_obj)
        except qc.QCError as e:
            raise GateError(f"QC policy or metrics: {e}") from e
        self.notes: list[str] = []
        self._keep_baseline_copy()

    # -- baseline file ------------------------------------------------------------

    def _keep_baseline_copy(self) -> None:
        current = read_tsv_str(self.decisions_path)
        if "decided_by" in current.columns and (current["decided_by"] == "rules").all():
            if not self.baseline_path.exists() or self.baseline_path.read_bytes() != self.decisions_path.read_bytes():
                shutil.copyfile(self.decisions_path, self.baseline_path)
        elif not self.baseline_path.exists():
            raise GateError(f"{self.decisions_path} was not written by the rules engine and no {BASELINE_NAME} exists")
        base = read_tsv_str(self.baseline_path)
        mine = self.ctx.baseline.table
        if len(base) != len(mine) or (base["cell_id"].to_numpy() != mine["cell_id"].to_numpy()).any() or \
                (base["decision"].to_numpy() != mine["decision"].to_numpy()).any():
            self.notes.append("the recomputed rules baseline differs from S1/qc_decisions.baseline.tsv.gz "
                              "(policy or metrics changed since S1c?); the recomputed one is used")

    # -- gate interface ---------------------------------------------------------------

    def build_bundle(self) -> dict[str, Any]:
        b = self.ctx.bundle()
        b["run_id"] = self.cfg.get("run_id")
        b["policy_file"] = str(self.cfg.get("qc_policy"))
        return b

    def baseline(self, bundle: Mapping[str, Any]) -> dict[str, Any]:
        return self.ctx.baseline_proposal()

    def policy_check(self, proposal: Mapping[str, Any], bundle: Mapping[str, Any]) -> tuple[bool, list[str]]:
        _, violations = self.ctx.validate(proposal)
        return (not violations), violations

    def normalise(self, proposal: Mapping[str, Any], bundle: Mapping[str, Any]) -> dict[str, Any]:
        norm, violations = self.ctx.validate(proposal)
        if violations:
            raise GateError("invalid G1 proposal: " + "; ".join(violations))
        return norm

    def policy(self, bundle: Mapping[str, Any]) -> dict[str, Any]:
        return self.policy_obj_dict()

    def policy_obj_dict(self) -> dict[str, Any]:
        d = self.policy_obj.to_dict()
        d["proposal_rules"] = {
            "hard_limits": "may only be as strict as or stricter than the policy's",
            "n_mads": "between 1 and 10",
            "per_sample": "bounds replace the adaptive bounds for that sample and must sit within the hard limits",
            "review_triggers": "a proposal must not trigger a review rule the baseline did not trigger",
        }
        return d

    def make_worker(self, provider, ledger=None, **kw) -> QCWorker:
        return QCWorker(self.ctx, provider, ledger, **kw)

    def apply(self, decision: Mapping[str, Any], bundle: Mapping[str, Any]) -> None:
        content = decision.get("decision") or {}
        norm, violations = self.ctx.validate(content)
        if violations:
            raise GateError("decision violates the policy: " + "; ".join(violations))
        by = str(decision.get("decided_by", "rules"))
        gate_dir = self.run_dir / "gates" / GATE
        if self.ctx.is_baseline(norm):
            if self.decisions_path.read_bytes() != self.baseline_path.read_bytes():
                shutil.copyfile(self.baseline_path, self.decisions_path)
            dec = self.ctx.baseline
            applied_by = "rules"
        else:
            applied_by = "human" if by.startswith("human") else "agent"
            dec = self.ctx.apply(norm, decided_by=applied_by)
            write_tsv_gz(dec.table, self.decisions_path, na_rep="")
        summary = {"gate": GATE, "created_utc": utc_now(), "run_id": self.cfg.get("run_id"), "decided_by": by,
                   "table_decided_by": applied_by, "baseline_file": f"S1/{BASELINE_NAME}",
                   "thresholds": norm["thresholds"], "n_review_cells": len(norm.get("review_cells") or []),
                   "counts": dec.counts_total(), "counts_per_sample": dec.counts_per_sample(),
                   "counts_per_reason": dec.counts_per_reason(), "policy_hash": self.policy_obj.hash}
        write_json(summary, gate_dir / "qc_summary.json")

    def report_sections(self, bundle, proposal, review, decision) -> list[tuple[str, str]]:
        base = bundle.get("baseline", {}).get("counts_per_sample") or {}
        content = (decision or {}).get("decision") or proposal.get("proposal") or {}
        prop = (content.get("summary") or {}).get("per_sample") or {}
        rows = []
        for s in sorted(set(base) | set(prop)):
            b, p = base.get(s, {}), prop.get(s, {})
            rows.append({"sample": s, "baseline keep": b.get("keep"), "baseline exclude": b.get("exclude"),
                         "baseline review": b.get("review"), "proposed keep": p.get("keep"),
                         "proposed exclude": p.get("exclude"), "proposed review": p.get("review")})
        thr = content.get("thresholds") or {}
        sections = [("QC decisions per sample (baseline vs proposal)", table(rows)),
                    ("Proposed thresholds", pre(thr) + f"<p>{len(content.get('review_cells') or [])} cell(s) flagged for review</p>")]
        trig = bundle.get("baseline", {}).get("triggered_review_rules") or []
        if trig:
            sections.append(("Review rules triggered by the baseline", table(trig)))
        if self.notes:
            sections.append(("Baseline notes", "<ul>" + "".join(f"<li>{n}</li>" for n in self.notes) + "</ul>"))
        return sections


def make_gate(run_dir: Path, cfg: Mapping[str, Any]) -> G1Gate:
    return G1Gate(run_dir, cfg)


def main(argv: list[str] | None = None) -> int:
    return gate_main(GATE, argv)


if __name__ == "__main__":
    raise SystemExit(main())
