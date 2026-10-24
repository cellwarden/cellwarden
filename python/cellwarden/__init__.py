"""CellWarden (formerly scpipe): single-cell RNA-seq pipeline, Python side.

Environment variables are named ``CELLWARDEN_*``; the former ``SCPIPE_*`` names are still read
(copied to the new name when the new one is unset), so existing scripts and images keep working.
"""

import os as _os

for _k in [k for k in _os.environ if k.startswith("SCPIPE_")]:
    _os.environ.setdefault("CELLWARDEN_" + _k[len("SCPIPE_"):], _os.environ[_k])

from .exchange import (
    SCHEMA_VERSION,
    ExchangeError,
    counts_checksum,
    read_exchange,
    validate_exchange,
    write_exchange,
)

__all__ = [
    "SCHEMA_VERSION",
    "ExchangeError",
    "counts_checksum",
    "read_exchange",
    "validate_exchange",
    "write_exchange",
]
__version__ = "0.3.0"
