"""Decision gates G1-G4 (docs/agents-spec.md section 2).

:mod:`cellwarden.gates.protocol` holds the one flow every gate runs
(:func:`run_gate`), human approval (:func:`approve`) and the stage CLI
(:func:`gate_main`); :mod:`cellwarden.gates.report` renders ``report.html``;
``g1_qc`` .. ``g4_interpretation`` are the gates themselves.
"""

from .protocol import (
    DEFAULT_MODES,
    EXIT_WAITING,
    GATE_MODULES,
    MODES,
    GateError,
    GateResult,
    approve,
    diff_json,
    gate_main,
    gate_mode,
    load_gate,
    run_gate,
    write_gate_files,
)

__all__ = [
    "DEFAULT_MODES",
    "EXIT_WAITING",
    "GATE_MODULES",
    "MODES",
    "GateError",
    "GateResult",
    "approve",
    "diff_json",
    "gate_main",
    "gate_mode",
    "load_gate",
    "run_gate",
    "write_gate_files",
]
