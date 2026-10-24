"""Readable names for the pipeline stages.

Stage IDs (S0, S1a, ..., G4) stay the keys on disk: folders, ``logs/<id>.log``,
``state.json``, the ``gates:`` config section and the R ``--stage`` argument.
What people see and type uses the names below: the terminal output, ``cellwarden
status``, the dashboard and reports show the title, and the CLI (``--until``,
``cellwarden approve --stage``) also accepts the slug or the title, in any case.
Unknown IDs (test registries, plugins via ``CELLWARDEN_STAGES``) show as
themselves.
"""

from __future__ import annotations

from typing import Iterable

__all__ = ["STAGES", "slug", "title", "label", "resolve", "gate_id"]

# id: (slug, title)
STAGES: dict[str, tuple[str, str]] = {
    "S0": ("ingest", "Ingest"),
    "S1a": ("qc-metrics", "QC metrics"),
    "S1b": ("doublets", "Doublet detection"),
    "S1c": ("qc-filter", "QC filtering"),
    "G1": ("qc-review", "QC review"),
    "S2": ("clustering", "Normalise, PCA, batch correction, clustering"),
    "S3": ("evidence", "Markers and cell-type evidence"),
    "G2": ("context-review", "Tissue and disease context review"),
    "G3": ("annotation-review", "Cell-type annotation review"),
    "S4": ("dge", "Differential expression"),
    "G4": ("interpretation-review", "DGE interpretation review"),
    "S5": ("report", "Report and export"),
}


def slug(stage_id: str) -> str:
    """CLI name of a stage (``clustering``); the ID itself for an unknown stage."""
    return STAGES.get(stage_id, (stage_id, stage_id))[0]


def title(stage_id: str) -> str:
    """Display name of a stage (``Normalise, PCA, batch correction, clustering``)."""
    return STAGES.get(stage_id, (stage_id, stage_id))[1]


def label(stage_id: str) -> str:
    """Title with the short ID, for messages: ``QC review (G1)``."""
    t = title(stage_id)
    return t if t == stage_id else f"{t} ({stage_id})"


def _norm(s: str) -> str:
    return " ".join(str(s).lower().replace("_", "-").split())


def resolve(name: str, ids: Iterable[str] | None = None) -> str | None:
    """Stage ID for an ID, slug or title (any case); None when nothing matches.

    ``ids`` limits the answer to those stages (the active registry); by default
    the built-in ones.
    """
    pool = list(ids) if ids is not None else list(STAGES)
    n = _norm(name)
    for sid in pool:
        if n in (_norm(sid), _norm(slug(sid)), _norm(title(sid))):
            return sid
    return None


def gate_id(name: str) -> str:
    """G1..G4 for a gate ID or name (``context-review``); the input unchanged when unknown."""
    return resolve(name, [s for s in STAGES if s.startswith("G")]) or name
