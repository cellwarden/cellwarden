"""Consensus cell-type annotation from S3 evidence (docs/agents-spec.md section 5).

Per cluster, three sources vote for entries of the marker reference
(``markers_human.yaml``; fine labels grouped under broad labels, with CL IDs):

* **signatures** - lineage sets from ``R/stages/data/signatures.yaml`` scored
  per cluster (mean ``AddModuleScore``). Each set's mean score is z-scored
  across clusters; support = clip(z / 2, 0, 1), zero when the mean score is
  <= 0 (below the control genes), times a coverage factor
  ``min(1, n_genes_found / max(3, ceil(n_genes_set / 2)))`` so sets with only
  a couple of genes in the data carry little weight, times the ratio of the
  set's score to the cluster's best lineage score (z-scores are scale free, so
  this keeps a set that is barely above its background from matching the
  cluster's real lineage). A set votes for the fine entry named by its
  ``cell_type``; fine labels without a set of their own inherit their broad
  group's signature support when fine labels are compared.
* **celltypist** - fraction of the cluster's cells whose ``majority_voting``
  label maps (case-insensitively, through ``celltypist_labels`` aliases) to
  each entry. Present only when ``S3/celltypist.tsv.gz`` exists.
* **markers** - overlap of an entry's marker genes with the cluster's top S3
  markers: a hit at rank <= 10 counts 1, a later hit 0.5; the sum is divided
  by ``min(len(markers), 5, number of top markers the cluster has)`` and
  clipped to 1.

Scores are combined with :data:`DEFAULT_WEIGHTS` (renormalised over the
sources that are available) at the **broad** level first (a broad label takes
the best signature / marker support and the summed CellTypist fraction of its
fine entries). The best broad label wins; ``confidence = top - 0.5 * second``
(clipped to 0-1). Inside the winning broad group the fine label is the
non-generic entry whose own fine evidence beats the generic entry's by at
least ``FINE_MARGIN``; otherwise the generic entry (e.g. "T cell" rather than
"CD4 T cell").

``status``: ``conflict`` when two available sources each have a strong
(>= 0.5) top choice and those broad labels differ; else ``low_confidence``
when the confidence is below ``min_confidence`` (config
``annotation.min_confidence``, default 0.6) or nothing scored above zero
(label ``Unassigned``, CL ID empty); else ``consensus``.

Everything is deterministic: ties are broken by label order.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

__all__ = [
    "DEFAULT_WEIGHTS",
    "FINE_MARGIN",
    "STRONG_VOTE",
    "MarkerReference",
    "load_reference",
    "annotate",
    "annotate_bundle",
    "consensus_table",
    "CONSENSUS_COLUMNS",
]

DEFAULT_WEIGHTS: dict[str, float] = {"signature": 0.35, "celltypist": 0.30, "markers": 0.35}
FINE_MARGIN = 0.10
STRONG_VOTE = 0.5
TOP_RANK = 10
MARKER_DENOM = 5
CONSENSUS_COLUMNS = ["cluster", "label_broad", "label_fine", "cl_id", "confidence", "votes_json", "status"]
REFERENCE_PATH = Path(__file__).with_name("markers_human.yaml")


# --------------------------------------------------------------------------
# Reference
# --------------------------------------------------------------------------


@dataclass
class CellType:
    label_fine: str
    label_broad: str
    cl_id: str
    markers: tuple[str, ...]
    generic: bool = False
    celltypist_labels: tuple[str, ...] = ()
    tissue: tuple[str, ...] = ()


@dataclass
class MarkerReference:
    types: list[CellType]
    path: str = ""
    by_fine: dict[str, CellType] = field(default_factory=dict)
    by_broad: dict[str, list[CellType]] = field(default_factory=dict)
    celltypist_alias: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_fine = {t.label_fine: t for t in self.types}
        self.by_broad = {}
        for t in self.types:
            self.by_broad.setdefault(t.label_broad, []).append(t)
        self.celltypist_alias = {}
        for t in self.types:
            for a in (t.label_fine, *t.celltypist_labels):
                self.celltypist_alias.setdefault(_norm(a), t.label_fine)

    def generic_of(self, broad: str) -> CellType | None:
        for t in self.by_broad.get(broad, []):
            if t.generic:
                return t
        return None

    def map_celltypist(self, label: str) -> str | None:
        return self.celltypist_alias.get(_norm(label))


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().replace("_", " ").split())


def load_reference(path: str | Path | None = None) -> MarkerReference:
    """Load ``markers_human.yaml`` (or another file with the same layout) and check it."""
    p = Path(path) if path else REFERENCE_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    entries = (data or {}).get("cell_types")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{p}: no 'cell_types' list")
    types: list[CellType] = []
    seen: set[str] = set()
    for i, e in enumerate(entries):
        if not isinstance(e, Mapping):
            raise ValueError(f"{p}: cell_types[{i}] is not a mapping")
        for k in ("label_fine", "label_broad", "cl_id", "markers"):
            if k not in e:
                raise ValueError(f"{p}: cell_types[{i}] lacks {k!r}")
        fine = str(e["label_fine"])
        if fine in seen:
            raise ValueError(f"{p}: duplicated label_fine {fine!r}")
        seen.add(fine)
        cl = str(e["cl_id"])
        if not cl.startswith("CL:"):
            raise ValueError(f"{p}: {fine}: cl_id {cl!r} is not a CL: id")
        markers = tuple(str(g) for g in e["markers"])
        if len(markers) < 3:
            raise ValueError(f"{p}: {fine}: fewer than 3 markers")
        types.append(CellType(fine, str(e["label_broad"]), cl, markers, bool(e.get("generic", False)),
                              tuple(str(x) for x in e.get("celltypist_labels", []) or []),
                              tuple(str(x) for x in e.get("tissue", []) or [])))
    return MarkerReference(types, str(p))


# --------------------------------------------------------------------------
# Votes
# --------------------------------------------------------------------------


def _zscores(values: Mapping[str, float | None]) -> dict[str, float]:
    xs = [float(v) for v in values.values() if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(xs) < 2:
        return {k: 0.0 for k in values}
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    sd = math.sqrt(var)
    out = {}
    for k, v in values.items():
        if v is None or (isinstance(v, float) and math.isnan(v)) or sd == 0:
            out[k] = 0.0
        else:
            out[k] = (float(v) - mean) / sd
    return out


def _coverage(n_found: int, n_set: int) -> float:
    need = max(3, math.ceil(n_set / 2)) if n_set else 3
    return min(1.0, n_found / need) if n_found > 0 else 0.0


def _valid(x: Any) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def signature_votes(clusters: Mapping[str, Mapping[str, Any]], signature_sets: Mapping[str, Mapping[str, Any]],
                    ref: MarkerReference) -> tuple[dict[str, dict[str, float]], bool, set[str]]:
    """Per cluster, support per fine label from the lineage signature sets.

    Returns ``(votes, available, covered)``; ``covered`` is the set of fine labels that have at
    least one scored set (fine labels without one inherit their broad group's support).
    """
    votes: dict[str, dict[str, float]] = {c: {} for c in clusters}
    available = False
    covered: set[str] = set()
    usable: dict[str, tuple[str, float]] = {}
    for name, meta in signature_sets.items():
        fine = meta.get("cell_type")
        if (meta.get("category", "lineage") != "lineage") or not fine or fine not in ref.by_fine:
            continue
        n_found = int(meta.get("n_genes_found", 0) or 0)
        if n_found <= 0:
            continue
        usable[name] = (fine, _coverage(n_found, int(meta.get("n_genes_set", 0) or 0)))
        covered.add(fine)
    # the cluster's strongest lineage set defines the scale: weaker sets are scaled by their ratio to it,
    # so z-scores (scale free) cannot promote a set that is barely above its background
    top_score: dict[str, float] = {}
    for c in clusters:
        sc = clusters[c].get("signatures") or {}
        pos = [float(sc[n]) for n in usable if _valid(sc.get(n)) and float(sc[n]) > 0]
        top_score[c] = max(pos) if pos else 0.0
    for name, (fine, cov) in usable.items():
        scores = {c: (clusters[c].get("signatures") or {}).get(name) for c in clusters}
        z = _zscores(scores)
        available = True
        for c in clusters:
            raw = scores[c]
            if not _valid(raw) or float(raw) <= 0 or top_score[c] <= 0:
                support = 0.0
            else:
                support = min(1.0, max(0.0, z[c] / 2.0)) * cov * min(1.0, float(raw) / top_score[c])
            if support > votes[c].get(fine, 0.0):
                votes[c][fine] = round(support, 4)
    return votes, available, covered


def celltypist_votes(clusters: Mapping[str, Mapping[str, Any]], ref: MarkerReference
                     ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]], bool]:
    """Per cluster, fraction of cells per mapped fine label; also the unmapped labels. Returns (votes, unmapped, available)."""
    votes: dict[str, dict[str, float]] = {c: {} for c in clusters}
    unmapped: dict[str, dict[str, float]] = {c: {} for c in clusters}
    available = False
    for c, ev in clusters.items():
        ct = ev.get("celltypist")
        if not ct:
            continue
        fr = ct.get("majority_voting") or ct.get("predicted_label") or {}
        if not fr:
            continue
        available = True
        for label, frac in fr.items():
            fine = ref.map_celltypist(label)
            if fine is None:
                unmapped[c][label] = round(float(frac), 4)
            else:
                votes[c][fine] = round(votes[c].get(fine, 0.0) + float(frac), 4)
    return votes, unmapped, available


def _marker_list(ev: Mapping[str, Any]) -> list[str]:
    tm = ev.get("top_markers") or []
    out = []
    for m in tm:
        g = m.get("gene") if isinstance(m, Mapping) else m
        if g:
            out.append(str(g))
    return out


def marker_votes(clusters: Mapping[str, Mapping[str, Any]], ref: MarkerReference) -> dict[str, dict[str, float]]:
    votes: dict[str, dict[str, float]] = {}
    for c, ev in clusters.items():
        genes = _marker_list(ev)
        rank: dict[str, int] = {}
        for i, g in enumerate(genes):
            rank.setdefault(g, i + 1)
        d: dict[str, float] = {}
        for t in ref.types:
            s = 0.0
            for g in t.markers:
                r = rank.get(g)
                if r is not None:
                    s += 1.0 if r <= TOP_RANK else 0.5
            # a cluster with fewer than MARKER_DENOM markers in total (tiny gene universe) is judged on those
            denom = min(len(t.markers), MARKER_DENOM, max(1, len(genes)))
            v = min(1.0, s / denom) if denom else 0.0
            if v > 0:
                d[t.label_fine] = round(v, 4)
        votes[c] = d
    return votes


# --------------------------------------------------------------------------
# Combination
# --------------------------------------------------------------------------


def _combine(parts: Mapping[str, float], weights: Mapping[str, float], present: Iterable[str]) -> float:
    present = [s for s in present if weights.get(s, 0) > 0]
    tot = sum(weights[s] for s in present)
    if tot <= 0:
        return 0.0
    return sum(weights[s] * parts.get(s, 0.0) for s in present) / tot


def _top2(d: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:2]


def _broad_support(fine_votes: Mapping[str, float], ref: MarkerReference, how: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for fine, v in fine_votes.items():
        b = ref.by_fine[fine].label_broad
        if how == "sum":
            out[b] = round(out.get(b, 0.0) + v, 4)
        else:
            out[b] = max(out.get(b, 0.0), v)
    return out


def annotate(clusters: Mapping[str, Mapping[str, Any]], signature_sets: Mapping[str, Mapping[str, Any]] | None = None,
             ref: MarkerReference | None = None, *, min_confidence: float = 0.6,
             weights: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    """Consensus labels for ``clusters`` (see the module docstring for the inputs).

    ``clusters``: ``{cluster_id: {"top_markers": [gene | {"gene": ...}, ...] (ranked),
    "signatures": {set: mean_score}, "celltypist": {"majority_voting": {label: fraction}} | None}}``.
    ``signature_sets``: ``{set: {"cell_type", "category", "n_genes_set", "n_genes_found"}}``.
    Returns one record per cluster, in input order, with the columns of
    :data:`CONSENSUS_COLUMNS` (``votes`` as a dict, ``votes_json`` as its JSON).
    """
    ref = ref or load_reference()
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items()})
    signature_sets = signature_sets or {}
    sig, sig_av, sig_covered = signature_votes(clusters, signature_sets, ref)
    ct, ct_unmapped, ct_av = celltypist_votes(clusters, ref)
    mk = marker_votes(clusters, ref)
    sources = [s for s, av in (("signature", sig_av), ("celltypist", ct_av), ("markers", True)) if av]

    records: list[dict[str, Any]] = []
    for c in clusters:
        per_source_fine = {"signature": sig[c], "celltypist": ct[c], "markers": mk[c]}
        per_source_broad = {
            "signature": _broad_support(sig[c], ref, "max"),
            "celltypist": _broad_support(ct[c], ref, "sum"),
            "markers": _broad_support(mk[c], ref, "max"),
        }
        broads = set().union(*(d.keys() for d in per_source_broad.values()))
        broad_scores = {b: round(_combine({s: per_source_broad[s].get(b, 0.0) for s in sources}, w, sources), 4)
                        for b in broads}
        top = _top2({b: v for b, v in broad_scores.items() if v > 0})
        votes: dict[str, Any] = {
            "sources": sources,
            "weights": {s: w[s] for s in sources},
            "signature": dict(_top_n(per_source_broad["signature"])),
            "celltypist": dict(_top_n(per_source_broad["celltypist"])),
            "celltypist_unmapped": ct_unmapped[c],
            "markers": dict(_top_n(per_source_broad["markers"])),
            "broad_scores": dict(_top_n(broad_scores, 5)),
        }
        if not top:
            records.append({"cluster": c, "label_broad": "Unassigned", "label_fine": "Unassigned", "cl_id": "",
                            "confidence": 0.0, "votes": votes, "status": "low_confidence"})
            continue
        best_b, best_s = top[0]
        second_s = top[1][1] if len(top) > 1 else 0.0
        confidence = round(max(0.0, min(1.0, best_s - 0.5 * second_s)), 4)

        # fine label inside the winning broad group
        fine_scores: dict[str, float] = {}
        broad_sig = per_source_broad["signature"].get(best_b, 0.0)
        for t in ref.by_broad[best_b]:
            parts = {s: per_source_fine[s].get(t.label_fine, 0.0) for s in sources}
            if "signature" in sources and not t.generic and t.label_fine not in sig_covered:
                parts["signature"] = broad_sig  # lineage-level evidence applies to every fine label of the group
            fine_scores[t.label_fine] = round(_combine(parts, w, sources), 4)
        generic = ref.generic_of(best_b)
        ranked = sorted(fine_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        best_fine, best_fine_s = ranked[0]
        if generic is not None and best_fine != generic.label_fine:
            if best_fine_s < fine_scores[generic.label_fine] + FINE_MARGIN:
                best_fine = generic.label_fine
        elif generic is None and len(ranked) > 1 and ranked[0][1] - ranked[1][1] < FINE_MARGIN:
            best_fine = ranked[0][0]  # no generic entry: keep the best (ties broken by name)
        votes["fine_scores"] = dict(_top_n(fine_scores, 5))

        # status
        strong: dict[str, str] = {}
        for s in sources:
            t2 = _top2(per_source_broad[s])
            if t2 and t2[0][1] >= STRONG_VOTE:
                strong[s] = t2[0][0]
        votes["source_top"] = strong
        if len(set(strong.values())) > 1:
            status = "conflict"
        elif confidence < min_confidence:
            status = "low_confidence"
        else:
            status = "consensus"
        ct_entry = ref.by_fine[best_fine]
        records.append({"cluster": c, "label_broad": best_b, "label_fine": best_fine, "cl_id": ct_entry.cl_id,
                        "confidence": confidence, "votes": votes, "status": status})
    for r in records:
        r["votes_json"] = json.dumps(r["votes"], sort_keys=True, separators=(",", ":"))
    return records


def _top_n(d: Mapping[str, float], n: int = 5) -> list[tuple[str, float]]:
    return sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def annotate_bundle(bundle: Mapping[str, Any], ref: MarkerReference | None = None, *,
                    min_confidence: float = 0.6, weights: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    """:func:`annotate` on the layout of ``S3/evidence_bundle.json`` (``clusters`` + ``signature_sets``)."""
    return annotate(bundle["clusters"], bundle.get("signature_sets") or {}, ref, min_confidence=min_confidence,
                    weights=weights)


def consensus_table(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """``annotation_consensus.tsv`` as a DataFrame (columns :data:`CONSENSUS_COLUMNS`)."""
    rows = [{k: r[k] for k in CONSENSUS_COLUMNS} for r in records]
    return pd.DataFrame(rows, columns=CONSENSUS_COLUMNS)
