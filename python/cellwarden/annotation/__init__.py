"""Code-only consensus cell-type annotation (docs/agents-spec.md section 5).

:mod:`cellwarden.annotation.consensus` turns the per-cluster evidence of stage S3
(signature scores, CellTypist votes, top markers) into labels with a
confidence and a status, using the marker reference ``markers_human.yaml``.
No LLM is involved; the G3 agent may later revise only clusters whose status
is ``low_confidence`` or ``conflict``.
"""

from .consensus import (  # noqa: F401
    DEFAULT_WEIGHTS,
    MarkerReference,
    annotate,
    consensus_table,
    load_reference,
)

__all__ = ["DEFAULT_WEIGHTS", "MarkerReference", "annotate", "consensus_table", "load_reference"]
