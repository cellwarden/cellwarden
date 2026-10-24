"""What ``cellwarden run`` / ``cellwarden resume`` print while the stages run.

Two reporters share one set of events (run start, a stage skipped, started,
a step inside it, done, waiting at a gate, failed, interrupted, finished):

* :class:`PlainReporter`: one stable line per event, with the stage counter,
  ID and name (``[ 6/12] S2   Normalise, PCA, ... — done in 32.4s``). Used
  for pipes, ``tmux``/``nohup`` logs, scheduler logs and tests.
* :class:`LiveReporter`: a ``rich`` display for a terminal. Finished stages
  scroll above a live block with the current stage, its latest step, an
  overall bar, the elapsed time and an estimate of the time left.

:func:`make_reporter` picks one (``progress: auto | live | plain``).

Steps inside a stage come from the stage's log as it is written: the R
``scp_timed`` lines (``[INFO] start: harmony``) and the Python ``log()`` lines
(``[12:01:02] message``); see :class:`LogFollower`.

Time left (:class:`Estimator`): each stage has a built-in model, seconds =
fixed + per-1,000-cells, taken from PBMC3k runs; gates have a fixed time
(longer with LLM agents). After each run the ratio actual / predicted is kept
per stage in ``~/.cellwarden/stage_times.json`` (``CELLWARDEN_STAGE_TIMES``
overrides the path), and the median of the recent ratios scales later
predictions, so estimates adapt to the machine. Until the number of cells is
known (after ingest) only the elapsed time is shown.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .stage_names import title

__all__ = ["fmt_s", "fmt_bytes", "parse_step", "LogFollower", "Estimator", "PlainReporter", "LiveReporter",
           "make_reporter"]


def fmt_bytes(b: int | None) -> str:
    if b is None:
        return "-"
    for unit, div in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if b >= div:
            return f"{b / div:.2f} {unit}"
    return f"{b} B"


def fmt_s(s: float | None) -> str:
    if s is None:
        return "-"
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def fmt_eta(s: float) -> str:
    """Rounded "about" time for the estimate: under a minute, N min, or H h M min."""
    if s < 60:
        return "under a minute"
    m = int(round(s / 60))
    return f"about {m} min" if m < 60 else f"about {m // 60} h {m % 60:02d} min"


# --------------------------------------------------------------------------
# Steps inside a stage
# --------------------------------------------------------------------------

_R_STEP = re.compile(r"\[INFO\]\s+start:\s*(.+?)\s*$")
_PY_STEP = re.compile(r"^\[\d\d:\d\d:\d\d\]\s+(.+?)\s*$")
_NOISE = re.compile(r"(:\s*(start|done)\b)|\bdone in\b|^(S\d\w*|G\d)\s*:\s*start|^running:", re.I)


def parse_step(line: str) -> str | None:
    """The step a log line announces, or None (``# cellwarden:`` lines, ends of steps, other output)."""
    if line.startswith("#"):
        return None
    m = _R_STEP.search(line)
    if m:
        return m.group(1)
    m = _PY_STEP.match(line)
    if m and not _NOISE.search(m.group(1)):
        msg = m.group(1)
        return msg if len(msg) <= 90 else msg[:87] + "..."
    return None


class LogFollower:
    """Reads what a stage appended to its log since the last call; returns the latest step."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.pos = 0
        self.partial = ""

    def latest_step(self) -> str | None:
        try:
            with open(self.path, "r", errors="replace") as fh:
                fh.seek(self.pos)
                chunk = fh.read()
                self.pos = fh.tell()
        except OSError:
            return None
        if not chunk:
            return None
        lines = (self.partial + chunk).split("\n")
        self.partial = lines.pop()
        step = None
        for ln in lines:
            s = parse_step(ln)
            if s:
                step = s
        return step


# --------------------------------------------------------------------------
# Time left
# --------------------------------------------------------------------------

# seconds = fixed + per_kcell * (cells / 1000); PBMC3k (2,700 cells) on Iridis login/compute nodes.
STAGE_MODEL: dict[str, tuple[float, float]] = {
    "S0": (6.0, 1.3), "S1a": (10.0, 8.0), "S1b": (3.0, 1.1), "S1c": (3.0, 0.7), "S2": (12.0, 8.5),
    "S3": (20.0, 13.0), "S4": (10.0, 3.0), "S5": (6.0, 3.0),
}
GATE_SECONDS_LLM = {"G1": 40.0, "G2": 60.0, "G3": 25.0, "G4": 20.0}
GATE_SECONDS_RULES = 3.0
HISTORY_KEEP = 20


def _history_path() -> Path:
    return Path(os.environ.get("CELLWARDEN_STAGE_TIMES") or "~/.cellwarden/stage_times.json").expanduser()


class Estimator:
    """Predicted stage times for one run, scaled by what earlier runs took on this machine."""

    def __init__(self, run_dir: str | Path, cfg: Mapping[str, Any], history_path: str | Path | None = None):
        self.run_dir = Path(run_dir)
        self.llm = bool(cfg.get("llm"))
        self.path = Path(history_path) if history_path else _history_path()
        self._kcells: float | None = None
        try:
            self.history: dict[str, list[float]] = json.loads(self.path.read_text())
            if not isinstance(self.history, dict):
                self.history = {}
        except (OSError, ValueError):
            self.history = {}

    def kcells(self) -> float | None:
        if self._kcells is None:
            try:
                rep = json.loads((self.run_dir / "S0" / "ingest_report.json").read_text())
                n = sum(int(s.get("cells") or 0) for s in rep.get("samples") or [])
                self._kcells = n / 1000 if n > 0 else None
            except (OSError, ValueError, TypeError, AttributeError):
                return None
        return self._kcells

    def _base(self, sid: str) -> float | None:
        if sid in GATE_SECONDS_LLM:
            return GATE_SECONDS_LLM[sid] if self.llm else GATE_SECONDS_RULES
        if sid in STAGE_MODEL:
            kc = self.kcells()
            if kc is None:
                return None
            fixed, per = STAGE_MODEL[sid]
            return fixed + per * kc
        return None

    def _factor(self, sid: str) -> float:
        ratios = [r for r in self.history.get(sid, []) if isinstance(r, (int, float)) and r > 0]
        return statistics.median(ratios[-10:]) if ratios else 1.0

    def predict(self, sid: str) -> float | None:
        b = self._base(sid)
        return None if b is None else b * self._factor(sid)

    def remaining(self, ids: Sequence[str], current: str | None = None, elapsed: float = 0.0) -> float | None:
        """Seconds left for the current stage (``elapsed`` into it) and the stages in ``ids``; None if unknown."""
        total = 0.0
        if current is not None:
            p = self.predict(current)
            if p is None:
                return None
            total += max(p - elapsed, 0.1 * p)
        for sid in ids:
            p = self.predict(sid)
            if p is None:
                return None
            total += p
        return total

    def record(self, sid: str, wall: float) -> None:
        """Remember actual / predicted for this stage (never raises)."""
        b = self._base(sid)
        if not b or wall <= 0:
            return
        hist = self.history.setdefault(sid, [])
        hist.append(round(wall / b, 4))
        del hist[:-HISTORY_KEEP]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}")
            tmp.write_text(json.dumps(self.history, indent=1, sort_keys=True))
            os.replace(tmp, self.path)
        except OSError:
            pass


# --------------------------------------------------------------------------
# Reporters
# --------------------------------------------------------------------------


class PlainReporter:
    """One line per event; no colours or control characters."""

    def __init__(self, out, total: int):
        self.out = out
        self.total = total

    def say(self, msg: str = "") -> None:
        print(msg, file=self.out, flush=True)

    def _tag(self, i: int, sid: str) -> str:
        return f"  [{i + 1:>2}/{self.total}] {sid:<4} {title(sid)}"

    def run_start(self, run_id: str, run_dir: Path, ids: Sequence[str]) -> None:
        self.say(f"cellwarden: run {run_id} in {run_dir}")

    def skip(self, i: int, sid: str, why: str) -> None:
        self.say(f"{self._tag(i, sid)} — skipped ({why})")

    def start(self, i: int, sid: str, reason: str) -> None:
        self.say(f"{self._tag(i, sid)} — running ({reason})")

    def step(self, text: str) -> None:
        pass

    def done(self, i: int, sid: str, wall: float, peak: int | None) -> None:
        self.say(f"{self._tag(i, sid)} — done in {fmt_s(wall)}, peak RSS {fmt_bytes(peak)}")

    def waiting(self, i: int, sid: str, wall: float, report: Path, approve: str, resume: str) -> None:
        self.say(f"{self._tag(i, sid)} — WAITING for your decision after {fmt_s(wall)}")
        self.say(f"cellwarden: review the gate report: {report}\n"
                 f"  approve:  {approve} [--from-file edited.json] [--by NAME]\n"
                 f"  continue: {resume}")

    def failed(self, i: int, sid: str, code: int, wall: float, log_path: Path, lines: Sequence[str],
               resume: str) -> None:
        self.say(f"{self._tag(i, sid)} — FAILED (exit code {code}) after {fmt_s(wall)}. "
                 f"Last {len(lines)} lines of {log_path}:")
        for line in lines:
            self.say(f"         | {line}")
        self.say(f"cellwarden: fix the problem, then: {resume}")

    def interrupted(self, i: int, sid: str, resume: str) -> None:
        self.say(f"{self._tag(i, sid)} — interrupted; resume with: {resume}")

    def message(self, msg: str) -> None:
        self.say(msg)

    def pause(self) -> None:
        """Before prompting in the terminal."""

    def unpause(self) -> None:
        """After prompting."""

    def close(self) -> None:
        pass


class LiveReporter(PlainReporter):
    """``rich`` display: finished stages scroll above a live progress block."""

    def __init__(self, out, total: int, estimator: Estimator | None = None):
        super().__init__(out, total)
        from rich.console import Console
        from rich.live import Live

        self.console = Console(file=out, highlight=False)
        self.est = estimator
        self.t_run = time.monotonic()
        self.ids: list[str] = []
        self.done_n = 0
        self.cur: tuple[int, str] | None = None
        self.t_stage = 0.0
        self.step_text = ""
        self.closing = False
        self.live = Live(console=self.console, get_renderable=self._render, refresh_per_second=4, transient=True)
        self.live.start()

    # rendering ----------------------------------------------------------------
    def _render(self):
        from rich.progress_bar import ProgressBar
        from rich.spinner import Spinner
        from rich.table import Table
        from rich.text import Text

        if self.closing:
            return Text("")
        now = time.monotonic()
        grid = Table.grid(padding=(0, 1))
        grid.add_column(no_wrap=True)
        grid.add_column(ratio=1)
        frac = float(self.done_n)
        eta = None
        if self.cur is not None:
            i, sid = self.cur
            el = now - self.t_stage
            p = self.est.predict(sid) if self.est else None
            if p:
                frac += min(el / p, 0.95)
            if self.est:
                left = [s for s in self.ids[i + 1:]]
                eta = self.est.remaining(left, sid, el)
            head = Text.assemble((f"{title(sid)}", "bold"), (f"  {sid} · {fmt_s(el)}", "dim"))
            grid.add_row(Spinner("dots", style="cyan"), head)
            if self.step_text:
                grid.add_row(Text(""), Text(f"↳ {self.step_text}", style="dim"))
        bar = ProgressBar(total=max(self.total, 1), completed=min(frac, self.total), width=40)
        info = f" {self.done_n}/{self.total} stages · {fmt_s(now - self.t_run)} elapsed"
        if eta is not None:
            info += f" · {fmt_eta(eta)} left"
        row = Table.grid(padding=(0, 1))
        row.add_row(bar, Text(info, style="dim"))
        grid.add_row(Text(""), row)
        return grid

    def _line(self, mark: str, style: str, sid: str, rest: str) -> None:
        self.console.print(f"[{style}]{mark}[/] {title(sid)} [dim]({sid})[/] {rest}")

    # events -----------------------------------------------------------------------
    def run_start(self, run_id: str, run_dir: Path, ids: Sequence[str]) -> None:
        self.ids = list(ids)
        self.console.print(f"[bold]cellwarden[/] run [cyan]{run_id}[/] in {run_dir}")

    def skip(self, i: int, sid: str, why: str) -> None:
        self.done_n += 1
        self._line("·", "dim", sid, f"[dim]skipped, {why}[/]")

    def start(self, i: int, sid: str, reason: str) -> None:
        self.cur, self.t_stage, self.step_text = (i, sid), time.monotonic(), ""
        if reason != "not run yet" and not reason.startswith("upstream "):
            self.console.print(f"  [dim]{title(sid)}: {reason}[/]")

    def step(self, text: str) -> None:
        self.step_text = text

    def done(self, i: int, sid: str, wall: float, peak: int | None) -> None:
        self.done_n += 1
        self.cur = None
        self._line("✓", "green", sid, f"[dim]{fmt_s(wall)} · peak {fmt_bytes(peak)}[/]")

    def waiting(self, i: int, sid: str, wall: float, report: Path, approve: str, resume: str) -> None:
        self.cur = None
        self.close()
        self._line("⏸", "yellow", sid, f"waiting for your decision [dim]({fmt_s(wall)})[/]")
        self.console.print(f"  report:   {report}\n  approve:  {approve} [dim][--from-file edited.json] "
                           f"[--by NAME][/]\n  continue: {resume}", highlight=False, markup=True)

    def failed(self, i: int, sid: str, code: int, wall: float, log_path: Path, lines: Sequence[str],
               resume: str) -> None:
        self.cur = None
        self.close()
        self._line("✗", "red", sid, f"[red]failed (exit code {code}) after {fmt_s(wall)}[/]")
        self.console.print(f"  last {len(lines)} lines of {log_path}:", markup=False)
        for line in lines:
            self.console.print(f"  | {line}", markup=False, style="dim")
        self.console.print(f"  fix the problem, then: {resume}", markup=False)

    def interrupted(self, i: int, sid: str, resume: str) -> None:
        self.cur = None
        self.close()
        self._line("✗", "yellow", sid, f"interrupted; resume with: {resume}")

    def message(self, msg: str) -> None:
        self.console.print(msg, markup=False)

    def pause(self) -> None:
        self.close()

    def unpause(self) -> None:
        self.closing = False
        if not self.live.is_started:
            self.live.start()

    def close(self) -> None:
        if self.live.is_started:
            self.closing = True
            self.live.refresh()
            self.live.stop()


def make_reporter(mode: str, out, total: int, estimator: Estimator | None = None) -> PlainReporter:
    """``live`` when asked, or for ``auto`` when ``out`` is the terminal and ``rich`` is installed."""
    if mode not in ("auto", "live", "plain"):
        raise ValueError(f"progress must be auto, live or plain, got {mode!r}")
    want = mode == "live" or (mode == "auto" and out is sys.stdout and _is_tty(out)
                              and os.environ.get("TERM", "") != "dumb")
    if want:
        try:
            return LiveReporter(out, total, estimator)
        except ImportError:
            pass
    return PlainReporter(out, total)


def _is_tty(out) -> bool:
    try:
        return out.isatty()
    except (AttributeError, ValueError):
        return False
