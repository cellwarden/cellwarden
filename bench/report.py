"""Turn results.jsonl and parity.jsonl into report.md, a CSV summary and scaling plots.

    python bench/report.py --run bench/results/run1

Medians use measured repeats only (the warm-up repeat is excluded). Speed-up and
memory ratio are against the baseline engine (Seurat) at the same size. The
verdict applies the adoption rule from docs/RUST_ENGINE.md section 8.2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

BASELINE = "seurat"
# Fixed order: an engine keeps its colour and marker in every chart and run.
ENGINE_ORDER = ["seurat", "scanpy", "scanpy-gpu", "rapids", "scfast-cpu", "scfast-gpu"]
COLORS = {"seurat": "#2a78d6", "scanpy": "#eb6834", "scanpy-gpu": "#1baf7a", "rapids": "#1baf7a",
          "scfast-cpu": "#eda100", "scfast-gpu": "#e87ba4"}
MARKERS = {"seurat": "o", "scanpy": "s", "scanpy-gpu": "^", "rapids": "^", "scfast-cpu": "D", "scfast-gpu": "v"}
STEP_ORDER = ["read", "to_gpu", "normalize", "hvg", "scale", "pca", "neighbors", "cluster", "umap", "to_cpu"]
COMPUTE_STEPS = ["normalize", "hvg", "scale", "pca", "neighbors", "cluster", "umap"]


def load(run: Path):
    rows = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    parity = pd.DataFrame()
    if (run / "parity.jsonl").exists() and (run / "parity.jsonl").read_text().strip():
        parity = pd.DataFrame([json.loads(l) for l in (run / "parity.jsonl").read_text().splitlines() if l.strip()])
    return df, parity


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    m = df[(df["status"] == "ok") & (~df["warmup"]) & (df["kind"] != "startup")]
    g = m.groupby(["engine", "cells", "step"]).agg(
        wall_s=("wall_s", "median"), wall_min=("wall_s", "min"), wall_max=("wall_s", "max"),
        reps=("wall_s", "size"), peak_rss_gb=("peak_rss_gb", "max"),
        peak_gpu_gb=("peak_gpu_gb", "max"), rss_scope=("rss_scope", "first"),
    ).reset_index()
    return g


def totals(summary: pd.DataFrame) -> pd.DataFrame:
    """End-to-end per engine and size: compute steps, plus I/O and transfers, plus peak memory."""
    out = []
    for (eng, cells), s in summary.groupby(["engine", "cells"]):
        steps = set(s["step"])
        out.append({
            "engine": eng, "cells": cells,
            "compute_s": s[s["step"].isin(COMPUTE_STEPS)]["wall_s"].sum(),
            "transfer_s": s[s["step"].isin(["to_gpu", "to_cpu"])]["wall_s"].sum(),
            "read_s": s[s["step"] == "read"]["wall_s"].sum(),
            "total_s": s["wall_s"].sum(),
            "peak_rss_gb": s["peak_rss_gb"].max(),
            "peak_gpu_gb": s["peak_gpu_gb"].max(),
            "complete": all(x in steps for x in ["normalize", "hvg", "scale", "pca", "neighbors", "cluster"]),
        })
    return pd.DataFrame(out)


def verdict(row, parity_ok) -> str:
    if row["engine"] == BASELINE:
        return "baseline"
    if parity_ok is None:
        return "no parity data"
    if not parity_ok:
        return "fails parity"
    if row["cells"] < common.ADOPT_MIN_CELLS:
        return f"parity ok; too small to judge (< {common.ADOPT_MIN_CELLS:,} cells)"
    fast = row.get("speedup", 0) >= common.ADOPT_MIN_SPEEDUP
    lean = row.get("mem_ratio", 0) >= common.ADOPT_MIN_MEM_RATIO
    return "adopt" if (fast or lean) else "keep optional"


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "–"
    return f"{x:.{nd}f}"


def _cells_label(n: float) -> str:
    n = int(round(n))
    if n >= 1_000_000 and n % 100_000 == 0:
        return f"{n / 1e6:g}M"
    if n >= 1_000 and n % 100 == 0:
        return f"{n / 1e3:g}k"
    return f"{n:,}"


def _log_axes(ax, sizes) -> None:
    """Log-log axes with x ticks at the benchmarked sizes and plain numbers, so labels do not collide."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _cells_label(v)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    # y: labelled ticks at 1-2-5 per decade as plain numbers (0.2, 5, 50), never 2 x 10^-1
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plots(summary: pd.DataFrame, tot: pd.DataFrame, run: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    if summary["cells"].nunique() < 2:
        return files  # a scaling curve needs at least two sizes
    sizes = sorted(summary["cells"].unique())
    ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3dd"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.titlecolor": ink})
    steps = [s for s in STEP_ORDER if s in set(summary["step"])]
    cols = 4
    rows = -(-len(steps) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    engines = [e for e in ENGINE_ORDER if e in set(summary["engine"])] + \
              sorted(set(summary["engine"]) - set(ENGINE_ORDER))
    for ax, step in zip(axes.flat, steps):
        d = summary[summary["step"] == step]
        for eng in engines:
            e = d[d["engine"] == eng].sort_values("cells")
            if e.empty:
                continue
            ax.plot(e["cells"], e["wall_s"], color=COLORS.get(eng, muted), lw=2,
                    marker=MARKERS.get(eng, "o"), ms=5, label=eng)
        _log_axes(ax, sizes)
        ax.set_title(step, fontsize=10, loc="left")
        ax.grid(True, which="major", color=grid, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("cells"); ax.set_ylabel("seconds (median)")
    for ax in list(axes.flat)[len(steps):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    f = run / "scaling_time.png"
    fig.savefig(f, dpi=150); plt.close(fig); files.append(f.name)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, col, label in [(axes[0], "compute_s", "compute time, seconds (sum of median steps)"),
                           (axes[1], "peak_rss_gb", "peak host memory, GB")]:
        for eng in engines:
            e = tot[tot["engine"] == eng].sort_values("cells")
            if e.empty or e[col].isna().all():
                continue
            ax.plot(e["cells"], e[col], color=COLORS.get(eng, muted), lw=2, marker=MARKERS.get(eng, "o"), ms=5)
            last = e.iloc[-1]
            ax.annotate(eng, (last["cells"], last[col]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=ink)  # direct label: identity is not colour-only
        _log_axes(ax, sizes)
        ax.set_xlabel("cells"); ax.set_title(label, fontsize=10, loc="left")
        ax.grid(True, color=grid, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    f = run / "scaling_totals.png"
    fig.savefig(f, dpi=150); plt.close(fig); files.append(f.name)
    return files


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", required=True)
    a = p.parse_args(argv)
    run = Path(a.run)
    df, parity = load(run)
    summary = summarise(df)
    tot = totals(summary)

    base = tot[tot["engine"] == BASELINE].set_index("cells")
    tot["speedup"] = tot.apply(lambda r: base["compute_s"].get(r["cells"]) / r["compute_s"]
                               if r["cells"] in base.index and r["compute_s"] > 0 else float("nan"), axis=1)
    tot["mem_ratio"] = tot.apply(lambda r: base["peak_rss_gb"].get(r["cells"]) / r["peak_rss_gb"]
                                 if r["cells"] in base.index and r["peak_rss_gb"] > 0 else float("nan"), axis=1)

    def parity_for(eng, cells):
        if parity.empty:
            return None
        m = parity[(parity["engine"] == eng) & (parity["cells"] == cells) & (parity["status"] == "ok")]
        return None if m.empty else m.iloc[0]

    lines = [f"# Benchmark report: `{run.name}`", ""]
    envs = sorted((run / "env").glob("*.json")) if (run / "env").exists() else []
    if envs:
        e = json.loads(envs[0].read_text())
        gpu = ", ".join(g["name"] if isinstance(g, dict) else str(g) for g in e.get("gpus", [])) or "none"
        commits = sorted({c for c in df.get("commit", pd.Series(dtype=str)).dropna().unique() if c})
        lines += [f"Node `{e.get('node')}` · {e.get('cpu_model')} · {e.get('logical_cpus')} logical CPUs · GPU: {gpu} · "
                  f"commit{'s' if len(commits) > 1 else ''} {', '.join(f'`{c}`' for c in commits) or 'unknown'}", ""]
        if len(commits) > 1:
            lines += ["**Warning:** records in this run come from more than one commit, so engines may not have run the "
                      "same harness code. Rerun on a single commit for publishable numbers.", ""]
        for p in envs:
            if p.name.startswith("seurat_"):
                s = json.loads(p.read_text())
                backend = s.get("leiden_backend")
                if backend:
                    lines += [f"Seurat {s.get('packages', {}).get('Seurat', '?')} Leiden backend: {backend.split(' (')[0]}."
                              + (" Seurat 5.0.x builds a dense cells × cells matrix for Leiden, which inflates its cluster-step "
                                 "time and memory; compare cluster-step numbers with care." if "reticulate" in backend else ""),
                              ""]
                break
    unavailable = df[df["status"] == "unavailable"]["engine"].unique().tolist()
    failed = df[df["status"] == "error"][["engine", "cells", "step", "error"]].drop_duplicates()
    if unavailable:
        lines += [f"Not available on this machine: {', '.join(unavailable)}.", ""]

    lines += ["## End to end", "",
              "Compute = normalise → UMAP (median of measured repeats, summed). Read and CPU⇄GPU transfers are "
              "shown separately. Speed-up and memory ratio are against Seurat at the same size; higher is better.", "",
              "| Cells | Engine | Compute s | Read s | Transfer s | Peak RSS GB | Peak GPU GB | Speed-up | Memory ratio | "
              "PC sim | kNN overlap | ARI | Verdict |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    order = {e: i for i, e in enumerate(ENGINE_ORDER)}
    tot = tot.sort_values(["cells", "engine"], key=lambda s: s.map(order) if s.name == "engine" else s)
    for _, r in tot.iterrows():
        par = parity_for(r["engine"], r["cells"])
        v = verdict(r, None if par is None else bool(par["pass_all"]))
        if not r["complete"]:
            v = "incomplete run"
        lines.append(
            f"| {int(r['cells']):,} | {r['engine']} | {fmt(r['compute_s'])} | {fmt(r['read_s'])} | "
            f"{fmt(r['transfer_s'])} | {fmt(r['peak_rss_gb'])} | {fmt(r['peak_gpu_gb'])} | "
            f"{fmt(r['speedup'])}× | {fmt(r['mem_ratio'])}× | "
            f"{fmt(None if par is None else par['pc_similarity'], 3)} | "
            f"{fmt(None if par is None else par['knn_overlap'], 3)} | "
            f"{fmt(None if par is None else par['ari'], 3)} | {v} |")
    lines += ["", f"Parity thresholds: PC similarity ≥ {common.PARITY_THRESHOLDS['pc_similarity']}, kNN overlap ≥ "
              f"{common.PARITY_THRESHOLDS['knn_overlap']}, ARI ≥ {common.PARITY_THRESHOLDS['ari']}. Adoption: parity "
              f"passes and, at ≥ {common.ADOPT_MIN_CELLS:,} cells, ≥ {common.ADOPT_MIN_SPEEDUP}× faster or ≥ "
              f"{common.ADOPT_MIN_MEM_RATIO}× less peak memory. Scanpy and rapids run their own pipelines (different "
              "HVG, graph and Leiden implementations), so their parity is informative only; the rule is meant for scfast.", ""]

    if not parity.empty and "ari_truth" in parity.columns:
        lines += ["## Agreement with known cell types", "", "| Cells | Engine | ARI vs truth | Reference ARI vs truth |",
                  "|---:|---|---:|---:|"]
        for _, r in parity[parity["status"] == "ok"].iterrows():
            lines.append(f"| {int(r['cells']):,} | {r['engine']} | {fmt(r.get('ari_truth'), 3)} | "
                         f"{fmt(r.get('ari_truth_ref'), 3)} |")
        lines.append("")

    lines += ["## Per step", "", "Median wall time in seconds (min–max over measured repeats); peak RSS within the step "
              "when the platform supports per-step peaks.", "",
              "| Cells | Engine | Step | Median s | Min–max s | Repeats | Peak RSS GB | RSS scope |",
              "|---:|---|---|---:|---:|---:|---:|---|"]
    srt = summary.assign(_s=summary["step"].map({s: i for i, s in enumerate(STEP_ORDER)}),
                         _e=summary["engine"].map(order)).sort_values(["cells", "_e", "_s"])
    for _, r in srt.iterrows():
        lines.append(f"| {int(r['cells']):,} | {r['engine']} | {r['step']} | {fmt(r['wall_s'], 3)} | "
                     f"{fmt(r['wall_min'], 3)}–{fmt(r['wall_max'], 3)} | {int(r['reps'])} | {fmt(r['peak_rss_gb'])} | "
                     f"{r['rss_scope']} |")

    st = df[(df["step"] == "startup") & (df["status"] == "ok")]
    if not st.empty:
        lines += ["", "## Start-up (cold)", "", "| Engine | Cells | Start-up s |", "|---|---:|---:|"]
        for _, r in st.iterrows():
            lines.append(f"| {r['engine']} | {int(r['cells']):,} | {fmt(r['wall_s'])} |")
    if not failed.empty:
        lines += ["", "## Failures", "", "| Engine | Cells | Step | Error |", "|---|---:|---|---|"]
        for _, r in failed.iterrows():
            lines.append(f"| {r['engine']} | {int(r['cells']):,} | {r['step']} | {str(r['error'])[:160]} |")

    files = plots(summary, tot, run)
    if files:
        lines += ["", "## Scaling", ""] + [f"![{f}]({f})" for f in files] + \
                 ["", "Plotted values are in `summary_steps.csv` and `summary_totals.csv`."]

    summary.to_csv(run / "summary_steps.csv", index=False)
    tot.to_csv(run / "summary_totals.csv", index=False)
    (run / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {run / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
