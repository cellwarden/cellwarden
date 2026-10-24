"""Open Targets Platform client and the S4 evidence summary (docs/agents-spec.md section 6).

    from cellwarden.opentargets import OpenTargetsClient
    c = OpenTargetsClient("refs/opentargets", offline=True)
    c.target_by_symbol("MS4A1")          # evidence dict, {} if unknown, None on an offline cache miss
    c.search_entity("systemic lupus erythematosus", "disease")

``configure(cache_dir, ...)`` sets a module-level default so the plain
functions ``target_by_symbol`` / ``target_by_ensembl`` / ``search_entity``
work without passing a client (what the G2 context agent uses).
"""

from .client import (
    OpenTargetsClient,
    OpenTargetsError,
    cache_key,
    configure,
    search_entity,
    shape_target,
    target_by_ensembl,
    target_by_symbol,
)
from .queries import ENDPOINT
from .summary import build_summary, read_dge_table, read_gene_map

__all__ = [
    "ENDPOINT",
    "OpenTargetsClient",
    "OpenTargetsError",
    "cache_key",
    "configure",
    "search_entity",
    "shape_target",
    "target_by_ensembl",
    "target_by_symbol",
    "build_summary",
    "read_dge_table",
    "read_gene_map",
]
