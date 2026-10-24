"""Cost ledger (budget guard) and the agent call log with replay.

Ledger (DESIGN 4.8, agents-spec section 3)
------------------------------------------
:class:`Ledger` records the token use and cost of every LLM call in two
files: the run ledger (``<run-dir>/llm/ledger.json``) and the total ledger
shared across runs (``llm.budget.ledger_path``, default
``~/.cellwarden/ledger.json``). The total ledger keeps one running total per
``run_id`` and sums them, so re-running a run never double-counts; both
files are rewritten atomically on every charge.

Prices come from ``prices.yaml`` next to this module (USD per million
tokens) and are converted to GBP with ``usd_to_gbp``. Limits are in GBP:
``per_run_limit`` (default 5) and ``total_limit`` (default 30).
:meth:`Ledger.would_exceed` is the hard stop: the agent asks before every
call with an estimate of the tokens it is about to send and the answer it
expects; a call that would cross either limit is never made. At
``warn_at_pct`` (default 80 %) of either limit :meth:`Ledger.warnings`
reports it, and the gate report shows it.

Agent log (agents-spec section 2)
---------------------------------
:class:`AgentLog` appends one JSON line per LLM call to
``<run-dir>/llm/agent_log.jsonl``::

    {ts, gate, role, model, prompt_sha256, input_sha256, tool_calls, output,
     usage: {input_tokens, output_tokens, cached_tokens}, cost_gbp, replay}

``prompt_sha256`` hashes the system prompt plus the tool definitions,
``input_sha256`` the conversation sent. With ``llm.replay: true``,
:meth:`AgentLog.find` returns the logged output for an identical
prompt + input and the agent reuses it without a call (and without cost);
the reuse is itself logged with ``replay: true``. Outputs are stored in
full (``Response.to_dict``) so a replay reconstructs the exact turn.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .provider import Response, Usage

__all__ = ["PRICES_FILE", "BudgetExceeded", "load_prices", "price_for", "Ledger", "AgentLog", "sha256_text"]

PRICES_FILE = Path(__file__).with_name("prices.yaml")
DEFAULT_TOTAL_LEDGER = "~/.cellwarden/ledger.json"
DEFAULT_BUDGET: dict[str, Any] = {
    "currency": "GBP",
    "total_limit": 30.0,
    "per_run_limit": 5.0,
    "warn_at_pct": 80.0,
    "usd_to_gbp": 0.78,
    "ledger_path": DEFAULT_TOTAL_LEDGER,
}
CHARS_PER_TOKEN = 4.0
_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    """A call was refused because it would cross a budget limit."""


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_prices(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """``{model: {input, output, cached}}`` in USD per million tokens."""
    data = yaml.safe_load(Path(path or PRICES_FILE).read_text(encoding="utf-8")) or {}
    out: dict[str, dict[str, float]] = {}
    for name, p in (data.get("models") or {}).items():
        out[str(name)] = {"input": float(p.get("input", 0)), "output": float(p.get("output", 0)),
                          "cached": float(p.get("cached", p.get("input", 0)))}
    return out


def price_for(prices: Mapping[str, Mapping[str, float]], model: str) -> tuple[dict[str, float], str]:
    """Price row for ``model`` and how it was matched (``exact``, ``prefix:<row>`` or ``default``)."""
    if model in prices:
        return dict(prices[model]), "exact"
    best = ""
    for name in prices:
        if name.startswith("_"):
            continue
        if model.startswith(name) and len(name) > len(best):
            best = name
    if best:
        return dict(prices[best]), f"prefix:{best}"
    if "_default" in prices:
        return dict(prices["_default"]), "default"
    return {"input": 0.0, "output": 0.0, "cached": 0.0}, "none"


class Ledger:
    """Per-run and total cost ledgers with the budget guard (see module docstring)."""

    def __init__(self, path: str | Path, *, run_id: str = "adhoc", budget: Mapping[str, Any] | None = None,
                 total_path: str | Path | None = None, prices: Mapping[str, Mapping[str, float]] | None = None,
                 per_run_limit: float | None = None, total_limit: float | None = None,
                 usd_to_gbp: float | None = None):
        b = dict(DEFAULT_BUDGET, **{k: v for k, v in (budget or {}).items() if v is not None})
        for k, v in (("per_run_limit", per_run_limit), ("total_limit", total_limit), ("usd_to_gbp", usd_to_gbp)):
            if v is not None:
                b[k] = v
        self.path = Path(path)
        self.run_id = str(run_id)
        self.per_run_limit = float(b["per_run_limit"])
        self.total_limit = float(b["total_limit"])
        self.warn_at_pct = float(b["warn_at_pct"])
        self.usd_to_gbp = float(b["usd_to_gbp"])
        self.total_path = Path(os.path.expanduser(str(total_path or b.get("ledger_path") or DEFAULT_TOTAL_LEDGER)))
        self.prices = dict(prices) if prices is not None else load_prices()
        self.run = _read_json(self.path) or {"schema": 1, "run_id": self.run_id, "currency": "GBP",
                                             "entries": [], "total_gbp": 0.0}
        self.run.setdefault("entries", [])
        self.run["run_id"] = self.run_id
        self.run["total_gbp"] = float(self.run.get("total_gbp") or 0.0)

    # -- pricing -----------------------------------------------------------

    def cost_gbp(self, model: str, usage: Usage | Mapping[str, Any]) -> tuple[float, str]:
        u = usage if isinstance(usage, Usage) else Usage.from_dict(usage)
        p, how = price_for(self.prices, model)
        usd = (u.input_tokens * p["input"] + u.output_tokens * p["output"] + u.cached_tokens * p["cached"]) / 1e6
        return round(usd * self.usd_to_gbp, 8), how

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return int(len(text) / CHARS_PER_TOKEN) + 1

    # -- totals -----------------------------------------------------------

    def run_total(self) -> float:
        return float(self.run.get("total_gbp") or 0.0)

    def grand_total(self) -> float:
        """Total across all runs, with this run's current total substituted in."""
        data = _read_json(self.total_path)
        runs = dict(data.get("runs") or {})
        runs[self.run_id] = self.run_total()
        return float(sum(float(v) for v in runs.values()))

    def would_exceed(self, est_input_tokens: int, est_output_tokens: int, model: str | None = None) -> bool:
        """True when a call of this size would cross the per-run or the total limit."""
        model = model or self._last_model or next(iter(self.prices), "")
        est, _ = self.cost_gbp(model, Usage(int(est_input_tokens), int(est_output_tokens), 0))
        return (self.run_total() + est > self.per_run_limit) or (self.grand_total() + est > self.total_limit)

    _last_model: str | None = None

    def charge(self, role: str, model: str, usage: Usage | Mapping[str, Any], *, gate: str | None = None) -> float:
        """Record a completed call; returns its cost in GBP. Both ledgers are rewritten."""
        u = usage if isinstance(usage, Usage) else Usage.from_dict(usage)
        cost, how = self.cost_gbp(model, u)
        self._last_model = model
        entry = {"ts": _utc(), "gate": gate, "role": role, "model": model, "usage": u.to_dict(),
                 "cost_gbp": cost, "price_source": how}
        with _lock:
            self.run["entries"].append(entry)
            self.run["total_gbp"] = round(self.run_total() + cost, 8)
            self.run["updated_utc"] = entry["ts"]
            self.run["limits"] = {"per_run_limit": self.per_run_limit, "total_limit": self.total_limit,
                                  "warn_at_pct": self.warn_at_pct, "usd_to_gbp": self.usd_to_gbp}
            _write_json(self.path, self.run)
            data = _read_json(self.total_path)
            runs = dict(data.get("runs") or {})
            runs[self.run_id] = self.run_total()
            _write_json(self.total_path, {"schema": 1, "currency": "GBP", "runs": runs,
                                          "total_gbp": round(sum(float(v) for v in runs.values()), 8),
                                          "updated_utc": entry["ts"]})
        return cost

    # -- reporting --------------------------------------------------------

    def status(self) -> dict[str, Any]:
        run, total = self.run_total(), self.grand_total()
        return {
            "run_id": self.run_id,
            "currency": "GBP",
            "run_gbp": round(run, 6),
            "per_run_limit": self.per_run_limit,
            "run_pct": round(100.0 * run / self.per_run_limit, 2) if self.per_run_limit > 0 else None,
            "total_gbp": round(total, 6),
            "total_limit": self.total_limit,
            "total_pct": round(100.0 * total / self.total_limit, 2) if self.total_limit > 0 else None,
            "n_calls": len(self.run.get("entries") or []),
            "warnings": self.warnings(),
        }

    def warnings(self) -> list[str]:
        out = []
        run, total = self.run_total(), self.grand_total()
        if self.per_run_limit > 0 and 100.0 * run / self.per_run_limit >= self.warn_at_pct:
            out.append(f"run budget: {run:.3f} of {self.per_run_limit:.2f} GBP used "
                       f"({100.0 * run / self.per_run_limit:.0f}%)")
        if self.total_limit > 0 and 100.0 * total / self.total_limit >= self.warn_at_pct:
            out.append(f"total budget: {total:.3f} of {self.total_limit:.2f} GBP used "
                       f"({100.0 * total / self.total_limit:.0f}%)")
        return out


class AgentLog:
    """Append-only ``agent_log.jsonl`` with lookup for replay (see module docstring)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._index: dict[tuple[str, str], dict[str, Any]] | None = None

    def entries(self) -> Iterable[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def _load_index(self) -> dict[tuple[str, str], dict[str, Any]]:
        if self._index is None:
            self._index = {}
            for e in self.entries():
                if e.get("replay"):
                    continue
                key = (str(e.get("prompt_sha256")), str(e.get("input_sha256")))
                self._index.setdefault(key, e)  # first real call wins
        return self._index

    def find(self, prompt_sha256: str, input_sha256: str) -> dict[str, Any] | None:
        return self._load_index().get((prompt_sha256, input_sha256))

    def append(self, *, gate: str | None, role: str, model: str, prompt_sha256: str, input_sha256: str,
               response: Response, cost_gbp: float, replay: bool = False, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        entry = {
            "ts": _utc(),
            "gate": gate,
            "role": role,
            "model": model,
            "prompt_sha256": prompt_sha256,
            "input_sha256": input_sha256,
            "tool_calls": [t.to_dict() for t in response.tool_calls],
            "output": response.to_dict(),
            "usage": response.usage.to_dict(),
            "cost_gbp": cost_gbp,
            "replay": bool(replay),
        }
        if extra:
            entry.update(extra)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        if self._index is not None and not replay:
            self._index.setdefault((prompt_sha256, input_sha256), entry)
        return entry
