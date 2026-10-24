"""Small dependency-free plots for the S5 report (inline SVG), plus a raster
UMAP through matplotlib when it is installed (base64 PNG; falls back to a
subsampled SVG scatter).

Everything returns a string that can be dropped into the HTML. The palette is
colour-blind friendly (Okabe-Ito first, then a tab20-like extension) and is
shared with the explorer so the two agree.
"""

from __future__ import annotations

import base64
import io
import math
from html import escape
from typing import Iterable, Mapping, Sequence

PALETTE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999",
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
    "#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
    "#bcbd22", "#dbdb8d", "#17becf", "#9edae5", "#393b79", "#637939", "#8c6d31", "#843c39",
]
NA_COLOUR = "#d0d0d0"
SVG_SCATTER_MAX = 20000


def colour_for(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def category_colours(levels: Sequence[str]) -> dict[str, str]:
    return {lv: colour_for(i) for i, lv in enumerate(levels)}


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    return f"{v:.2g}"


# --------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------


def bar_chart(items: Sequence[tuple[str, float]], *, title: str = "", width: int = 560, bar_h: int = 18,
              colours: Mapping[str, str] | None = None, unit: str = "") -> str:
    """Horizontal bar chart. ``items`` = [(label, value)]."""
    if not items:
        return _empty(title)
    label_w = min(220, 7 * max(len(str(k)) for k, _ in items) + 12)
    vmax = max((v for _, v in items), default=0) or 1
    plot_w = width - label_w - 70
    h = 30 + bar_h * len(items) + 6
    out = [f'<svg class="plot" xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" viewBox="0 0 {width} {h}" role="img" aria-label="{escape(title)}">']
    if title:
        out.append(f'<text x="0" y="14" class="t">{escape(title)}</text>')
    y = 26
    for i, (k, v) in enumerate(items):
        w = max(0.0, plot_w * float(v) / vmax)
        c = (colours or {}).get(k, colour_for(i))
        out.append(f'<text x="{label_w - 6}" y="{y + bar_h - 5}" text-anchor="end" class="l">{escape(str(k))}</text>')
        out.append(f'<rect x="{label_w}" y="{y + 2}" width="{w:.1f}" height="{bar_h - 4}" fill="{c}"><title>{escape(str(k))}: {_fmt(v)}{unit}</title></rect>')
        out.append(f'<text x="{label_w + w + 4:.1f}" y="{y + bar_h - 5}" class="v">{_fmt(v)}{escape(unit)}</text>')
        y += bar_h
    out.append("</svg>")
    return "".join(out)


def stacked_bars(rows: Sequence[tuple[str, Mapping[str, float]]], series: Sequence[str], *, title: str = "",
                 width: int = 560, bar_h: int = 20, colours: Mapping[str, str] | None = None) -> str:
    """Horizontal stacked bars: ``rows`` = [(label, {series: value})]."""
    if not rows:
        return _empty(title)
    colours = dict(colours or category_colours(series))
    label_w = min(220, 7 * max(len(str(k)) for k, _ in rows) + 12)
    plot_w = width - label_w - 60
    vmax = max((sum(float(d.get(s, 0) or 0) for s in series) for _, d in rows), default=0) or 1
    h = 30 + bar_h * len(rows) + 24
    out = [f'<svg class="plot" xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" viewBox="0 0 {width} {h}" role="img" aria-label="{escape(title)}">']
    if title:
        out.append(f'<text x="0" y="14" class="t">{escape(title)}</text>')
    y = 26
    for k, d in rows:
        x = label_w
        tot = sum(float(d.get(s, 0) or 0) for s in series)
        out.append(f'<text x="{label_w - 6}" y="{y + bar_h - 6}" text-anchor="end" class="l">{escape(str(k))}</text>')
        for s in series:
            v = float(d.get(s, 0) or 0)
            w = plot_w * v / vmax
            if w > 0:
                out.append(f'<rect x="{x:.1f}" y="{y + 2}" width="{w:.1f}" height="{bar_h - 4}" fill="{colours[s]}">'
                           f'<title>{escape(str(k))} {escape(s)}: {_fmt(v)}</title></rect>')
            x += w
        out.append(f'<text x="{x + 4:.1f}" y="{y + bar_h - 6}" class="v">{_fmt(tot)}</text>')
        y += bar_h
    x = label_w
    for s in series:
        out.append(f'<rect x="{x}" y="{y + 6}" width="10" height="10" fill="{colours[s]}"/>'
                   f'<text x="{x + 14}" y="{y + 15}" class="l">{escape(s)}</text>')
        x += 14 + 7 * len(s) + 16
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------
# Histograms
# --------------------------------------------------------------------------


def histogram(values: Iterable[float], *, title: str = "", bins: int = 40, width: int = 360, height: int = 160,
              log10: bool = False, vlines: Mapping[str, float] | None = None, colour: str = "#0072B2") -> str:
    xs = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if log10:
        xs = [math.log10(v) for v in xs if v > 0]
    if not xs:
        return _empty(title)
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        hi = lo + 1
    counts = [0] * bins
    for v in xs:
        i = min(bins - 1, int((v - lo) / (hi - lo) * bins))
        counts[i] += 1
    cmax = max(counts) or 1
    ml, mr, mt, mb = 36, 8, 22, 26
    pw, ph = width - ml - mr, height - mt - mb
    out = [f'<svg class="plot" xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">']
    if title:
        out.append(f'<text x="0" y="14" class="t">{escape(title)}</text>')
    bw = pw / bins
    for i, c in enumerate(counts):
        bh = ph * c / cmax
        out.append(f'<rect x="{ml + i * bw:.1f}" y="{mt + ph - bh:.1f}" width="{max(bw - 0.5, 0.5):.1f}" height="{bh:.1f}" fill="{colour}">'
                   f'<title>{_fmt(10 ** (lo + i * (hi - lo) / bins) if log10 else lo + i * (hi - lo) / bins)}: {c}</title></rect>')
    out.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" class="ax"/>')
    for frac in (0, 0.5, 1):
        v = lo + frac * (hi - lo)
        lab = _fmt(10 ** v) if log10 else _fmt(v)
        out.append(f'<text x="{ml + frac * pw:.1f}" y="{height - 8}" text-anchor="middle" class="l">{lab}</text>')
    out.append(f'<text x="{ml - 4}" y="{mt + 10}" text-anchor="end" class="l">{cmax}</text>')
    for name, v in (vlines or {}).items():
        if v is None:
            continue
        vv = math.log10(v) if log10 and v > 0 else v
        if lo <= vv <= hi:
            x = ml + (vv - lo) / (hi - lo) * pw
            out.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + ph}" class="vl"><title>{escape(name)} = {_fmt(v)}</title></line>')
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------
# Scatter (UMAP)
# --------------------------------------------------------------------------


def _subsample(n: int, cap: int, seed: int = 0) -> list[int]:
    if n <= cap:
        return list(range(n))
    import random

    rng = random.Random(seed)
    return sorted(rng.sample(range(n), cap))


def scatter(x: Sequence[float], y: Sequence[float], labels: Sequence[str] | None = None, *, title: str = "",
            values: Sequence[float] | None = None, size: int = 440, legend: bool = True, seed: int = 0,
            prefer_png: bool = True, max_legend: int = 40) -> str:
    """UMAP scatter coloured by category (``labels``) or by a continuous ``values`` vector.

    PNG through matplotlib when available (every point), else SVG with at most
    :data:`SVG_SCATTER_MAX` points.
    """
    n = len(x)
    if n == 0:
        return _empty(title)
    if labels is not None:
        levels = sorted({str(v) for v in labels}, key=_level_key)
        cmap = category_colours(levels)
        cols = [cmap[str(v)] for v in labels]
    elif values is not None:
        vals = [float(v) if v is not None else float("nan") for v in values]
        finite = [v for v in vals if not math.isnan(v)]
        lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
        cols = [NA_COLOUR if math.isnan(v) else viridis((v - lo) / (hi - lo) if hi > lo else 0.0) for v in vals]
        levels, cmap = [], {}
    else:
        cols = ["#0072B2"] * n
        levels, cmap = [], {}
    if prefer_png:
        png = _scatter_png(x, y, cols, size)
        if png is not None:
            return _scatter_wrap(png, title, levels, cmap, legend, max_legend, n, n)
    idx = _subsample(n, SVG_SCATTER_MAX, seed)
    return _scatter_wrap(_scatter_svg([x[i] for i in idx], [y[i] for i in idx], [cols[i] for i in idx], size),
                         title, levels, cmap, legend, max_legend, n, len(idx))


def _level_key(s: str):
    try:
        return (0, float(s), s)
    except ValueError:
        return (1, 0.0, s)


def _scatter_wrap(body: str, title: str, levels, cmap, legend: bool, max_legend: int, n: int, shown: int) -> str:
    out = ['<figure class="scatter">']
    if title:
        out.append(f'<figcaption>{escape(title)} <span class="muted">({n:,} cells{"" if shown == n else f", {shown:,} drawn"})</span></figcaption>')
    out.append('<div class="scatter-row">')
    out.append(body)
    if legend and levels:
        out.append('<ul class="legend">')
        for lv in levels[:max_legend]:
            out.append(f'<li><span class="sw" style="background:{cmap[lv]}"></span>{escape(lv)}</li>')
        if len(levels) > max_legend:
            out.append(f'<li class="muted">… {len(levels) - max_legend} more</li>')
        out.append("</ul>")
    out.append("</div></figure>")
    return "".join(out)


def _scatter_svg(x, y, cols, size: int) -> str:
    xs, ys = [float(v) for v in x], [float(v) for v in y]
    xlo, xhi, ylo, yhi = min(xs), max(xs), min(ys), max(ys)
    sx = (size - 10) / ((xhi - xlo) or 1)
    sy = (size - 10) / ((yhi - ylo) or 1)
    r = 1.2 if len(xs) > 5000 else 2.0
    out = [f'<svg class="plot umap" xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">']
    for px, py, c in zip(xs, ys, cols):
        out.append(f'<circle cx="{5 + (px - xlo) * sx:.1f}" cy="{size - 5 - (py - ylo) * sy:.1f}" r="{r}" fill="{c}"/>')
    out.append("</svg>")
    return "".join(out)


def _scatter_png(x, y, cols, size: int) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    dpi = 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    n = len(x)
    s = 0.5 if n > 50000 else (2 if n > 5000 else 6)
    ax.scatter(x, y, c=cols, s=s, linewidths=0, rasterized=True)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img class="plot umap" width="{size}" height="{size}" alt="UMAP" src="data:image/png;base64,{b64}"/>'


def viridis(t: float) -> str:
    """Approximate viridis via a few anchor colours (no matplotlib needed)."""
    anchors = [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.5, (33, 145, 140)), (0.75, (94, 201, 98)), (1.0, (253, 231, 37))]
    t = min(1.0, max(0.0, t))
    for (t0, c0), (t1, c1) in zip(anchors, anchors[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0
            r, g, b = (round(a + (b_ - a) * f) for a, b_ in zip(c0, c1))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#fde725"


def _empty(title: str) -> str:
    return f'<p class="muted plot-empty">{escape(title + ": " if title else "")}no data</p>'
