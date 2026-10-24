#!/usr/bin/env python3
"""Summarise the scale runs (docs/scfast-integration.md, Phase 8) into a table and verdicts.

    python bench/hpc/summarise.py --runs bench/results/scale/runs.jsonl --out bench/results/scale

Reads the one-line-per-run records that ``run_scale.sh`` appends to ``runs.jsonl`` and, for
each run that has one, the run directory (``S2/summary.json`` for the S2 time, path and R peak
RSS; ``time.log`` for the whole run's peak RSS; ``gpu_mem.log`` for peak GPU memory;
``clusters.tsv.gz`` + ``truth.tsv`` for ARI). A record may also carry these numbers itself
(``s2_s``, ``wall_s``, ``peak_rss_gb``, ``peak_gpu_gb``, ``ari_vs_seurat``, ``ari_vs_truth``,
``path``), which is what the unit test uses; a file on disk fills in whatever is missing.

Writes ``summary.csv`` (one row per run), ``summary.md`` (the table per size: S2 time, whole-run
time, peak RSS, peak GPU memory, ARI vs the Seurat run and vs the planted types, verdict) and
``summary.json``. The verdict applies the adoption rule from docs/RUST_ENGINE.md section 8.2 to
each non-baseline engine at each size: it needs parity (ARI vs Seurat >= 0.90) and, at 100k
cells or more, at least 1.5x the Seurat S2 speed or at least 2x less peak host memory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402  (bench/common.py: thresholds)

BASELINE = "seurat-cpu"
ENGINE_ORDER = ("seurat-cpu", "scfast-cpu", "scfast-cuda", "rapids")
ARI_MIN = common.PARITY_THRESHOLDS["ari"]


# ------------------------------------------------------------------ readers


def read_runs(path: Path) -> list[dict]:
    runs = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: not JSON ({e})") from e
            for k in ("size", "engine"):
                if k not in rec:
                    raise ValueError(f"{path}:{i}: record without {k!r}")
            rec["size"] = int(rec["size"])
            rec.setdefault("status", "ok")
            runs.append(rec)
    return runs


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def read_clusters(path: Path) -> dict[str, str]:
    """cell_id -> label from a TSV (optionally gzipped) with a header; the first two columns."""
    opener = gzip.open if str(path).endswith(".gz") else open
    out: dict[str, str] = {}
    with opener(path, "rt") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def ari_between(a: dict[str, str], b: dict[str, str]) -> float | None:
    common_ids = [k for k in a if k in b]
    if len(common_ids) < 2:
        return None
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score([a[k] for k in common_ids], [b[k] for k in common_ids]))


def peak_rss_from_time_log(path: Path) -> float | None:
    """GNU time -v 'Maximum resident set size (kbytes)' as GB."""
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text(errors="replace"))
    return int(m.group(1)) / 1024**2 if m else None


def peak_gpu_from_log(path: Path) -> float | None:
    """Largest memory.used sample (MiB, last comma-separated field of each line) as GB."""
    best = None
    for line in path.read_text(errors="replace").splitlines():
        v = _num(line.strip().split(",")[-1])
        if v is not None:
            best = v if best is None else max(best, v)
    return None if best is None else best / 1024


def enrich(rec: dict) -> dict:
    """Fill the numbers a record lacks from its run directory and side files (when they exist)."""
    r = dict(rec)
    run_dir = Path(r["run_dir"]) if r.get("run_dir") else None
    s2 = None
    if run_dir is not None and (run_dir / "S2" / "summary.json").exists():
        try:
            s2 = json.loads((run_dir / "S2" / "summary.json").read_text())
        except (OSError, json.JSONDecodeError):
            s2 = None
    if s2 is not None:
        r.setdefault("s2_s", _num(s2.get("total_s")))
        r.setdefault("path", s2.get("path"))
        r.setdefault("cells", s2.get("cells"))
        r.setdefault("clusters_n", s2.get("clusters"))
        pm = s2.get("peak_memory") or {}
        cands = [_num(pm.get("r_peak_rss_gb")), _num(pm.get("gpu_core_peak_host_rss_gb"))]
        cands = [c for c in cands if c is not None]
        if cands and "peak_rss_gb" not in r:
            r["s2_peak_rss_gb"] = max(cands)
        if "peak_gpu_gb" not in r and _num(pm.get("gpu_pool_peak_gb")) is not None:
            r["peak_gpu_gb_pool"] = _num(pm.get("gpu_pool_peak_gb"))
        r.setdefault("s2_timings_s", s2.get("timings_s"))
    tl = Path(r["time_log"]) if r.get("time_log") else None
    if "peak_rss_gb" not in r and tl is not None and tl.exists():
        v = peak_rss_from_time_log(tl)
        if v is not None:
            r["peak_rss_gb"] = v
    if "peak_rss_gb" not in r and r.get("s2_peak_rss_gb") is not None:
        r["peak_rss_gb"] = r["s2_peak_rss_gb"]
    gl = Path(r["gpu_mem_log"]) if r.get("gpu_mem_log") else None
    if "peak_gpu_gb" not in r and gl is not None and gl.exists():
        v = peak_gpu_from_log(gl)
        if v is not None:
            r["peak_gpu_gb"] = v
    if "peak_gpu_gb" not in r and r.get("peak_gpu_gb_pool") is not None:
        r["peak_gpu_gb"] = r["peak_gpu_gb_pool"]
    return r


# ------------------------------------------------------------------ table


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b <= 0:
        return None
    return a / b


def verdict(row: dict, base: dict | None) -> str:
    if row["engine"] == BASELINE:
        return "baseline"
    if row.get("status", "ok") != "ok":
        return row.get("status", "failed")
    ari = _num(row.get("ari_vs_seurat"))
    if base is None or base.get("status", "ok") != "ok":
        return "no Seurat baseline"
    if ari is None:
        return "no parity data"
    if ari < ARI_MIN:
        return f"fails parity (ARI {ari:.2f} < {ARI_MIN:.2f})"
    if row["size"] < common.ADOPT_MIN_CELLS:
        return f"parity ok; too small to judge (< {common.ADOPT_MIN_CELLS:,} cells)"
    fast = row.get("speedup_s2")
    lean = row.get("mem_ratio")
    ok_fast = fast is not None and fast >= common.ADOPT_MIN_SPEEDUP
    ok_lean = lean is not None and lean >= common.ADOPT_MIN_MEM_RATIO
    if ok_fast or ok_lean:
        return "adopt"
    if fast is None and lean is None:
        return "no timing data"
    return "keep optional"


def build_rows(runs: list[dict]) -> list[dict]:
    rows = [enrich(r) for r in runs]
    # ARI against the Seurat run at the same size and against the planted types, from cluster files
    by_key = {(r["size"], r["engine"]): r for r in rows}
    cluster_cache: dict[str, dict[str, str]] = {}

    def clusters_of(r: dict) -> dict[str, str] | None:
        p = r.get("clusters")
        if not p or not Path(p).exists():
            return None
        if p not in cluster_cache:
            cluster_cache[p] = read_clusters(Path(p))
        return cluster_cache[p]

    for r in rows:
        base = by_key.get((r["size"], BASELINE))
        if "ari_vs_seurat" not in r and r["engine"] != BASELINE and base is not None:
            a, b = clusters_of(r), clusters_of(base)
            if a is not None and b is not None:
                r["ari_vs_seurat"] = ari_between(a, b)
        if "ari_vs_truth" not in r and r.get("truth") and Path(r["truth"]).exists():
            a = clusters_of(r)
            if a is not None:
                r["ari_vs_truth"] = ari_between(a, read_clusters(Path(r["truth"])))
    for r in rows:
        base = by_key.get((r["size"], BASELINE))
        if base is not None and r["engine"] != BASELINE:
            r["speedup_s2"] = _ratio(_num(base.get("s2_s")), _num(r.get("s2_s")))
            r["speedup_wall"] = _ratio(_num(base.get("wall_s")), _num(r.get("wall_s")))
            r["mem_ratio"] = _ratio(_num(base.get("peak_rss_gb")), _num(r.get("peak_rss_gb")))
        r["verdict"] = verdict(r, base)
    order = {e: i for i, e in enumerate(ENGINE_ORDER)}
    rows.sort(key=lambda r: (r["size"], order.get(r["engine"], len(order)), r["engine"]))
    return rows


COLUMNS = ("size", "engine", "status", "path", "s2_s", "wall_s", "peak_rss_gb", "peak_gpu_gb",
           "speedup_s2", "speedup_wall", "mem_ratio", "ari_vs_seurat", "ari_vs_truth", "verdict",
           "threads", "node", "commit")


def fmt(x, nd=2) -> str:
    v = _num(x)
    if v is None:
        return "-" if x in (None, "") else str(x)
    return f"{v:,.{nd}f}"


def to_markdown(rows: list[dict]) -> str:
    out = ["# Scale runs (docs/scfast-integration.md Phase 8)", ""]
    nodes = sorted({str(r.get("node")) for r in rows if r.get("node")})
    commits = sorted({str(r.get("commit")) for r in rows if r.get("commit")})
    threads = sorted({str(r.get("threads")) for r in rows if r.get("threads") is not None})
    out.append(f"Node(s): {', '.join(nodes) or '-'}; commit(s): {', '.join(commits) or '-'}; "
               f"threads: {', '.join(threads) or '-'}. Baseline: `{BASELINE}`. Adoption rule (RUST_ENGINE.md 8.2): "
               f"ARI vs Seurat >= {ARI_MIN:.2f} and, at >= {common.ADOPT_MIN_CELLS:,} cells, "
               f">= {common.ADOPT_MIN_SPEEDUP}x faster S2 or >= {common.ADOPT_MIN_MEM_RATIO}x less peak RSS.")
    out.append("")
    hdr = "| engine | status | S2 path | S2 time (s) | run time (s) | peak RSS (GB) | peak GPU (GB) | S2 speed-up | RSS ratio | ARI vs Seurat | ARI vs truth | verdict |"
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    for size in sorted({r["size"] for r in rows}):
        out += [f"## {size:,} cells", "", hdr, sep]
        for r in rows:
            if r["size"] != size:
                continue
            out.append("| " + " | ".join([
                str(r["engine"]), str(r.get("status", "ok")), str(r.get("path") or "-"),
                fmt(r.get("s2_s"), 1), fmt(r.get("wall_s"), 1), fmt(r.get("peak_rss_gb")), fmt(r.get("peak_gpu_gb")),
                fmt(r.get("speedup_s2")) + ("x" if _num(r.get("speedup_s2")) is not None else ""),
                fmt(r.get("mem_ratio")) + ("x" if _num(r.get("mem_ratio")) is not None else ""),
                fmt(r.get("ari_vs_seurat"), 3), fmt(r.get("ari_vs_truth"), 3), str(r.get("verdict")),
            ]) + " |")
        out.append("")
    return "\n".join(out)


def write_outputs(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUMNS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLUMNS})
    (out_dir / "summary.md").write_text(to_markdown(rows))
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", required=True, help="runs.jsonl written by run_scale.sh")
    p.add_argument("--out", required=True, help="folder for summary.md / summary.csv / summary.json")
    a = p.parse_args(argv)
    try:
        runs = read_runs(Path(a.runs))
    except (OSError, ValueError) as e:
        print(f"summarise: {e}", file=sys.stderr)
        return 2
    if not runs:
        print(f"summarise: no runs in {a.runs}", file=sys.stderr)
        return 2
    rows = build_rows(runs)
    write_outputs(rows, Path(a.out))
    print(to_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
