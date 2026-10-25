"""The CellWarden mark: one artwork in docs/logo, inlined into every generated page."""
from __future__ import annotations

import base64
import re
from pathlib import Path

from cellwarden import dashboard
from cellwarden.gates.report import render_report as plain_gate_report
from cellwarden.report import brand

LOGO = Path(__file__).resolve().parents[2] / "docs" / "logo"


def _path_d(svg: str) -> list[str]:
    return re.findall(r'\sd="([^"]+)"', svg)


def test_inline_mark_matches_docs_logo():
    for name in ("mark.svg", "mark-light.svg", "mark-dark.svg", "mark-mono.svg"):
        assert _path_d((LOGO / name).read_text()) == [brand.MARK_PATH], name
    assert brand.ACCENT_LIGHT in (LOGO / "mark-light.svg").read_text()
    assert brand.ACCENT_DARK in (LOGO / "mark-dark.svg").read_text()


def test_favicon_is_a_self_contained_svg():
    uri = brand.favicon_uri()
    assert uri.startswith("data:image/svg+xml;base64,")
    svg = base64.b64decode(uri.split(",", 1)[1]).decode()
    assert svg.startswith("<svg") and brand.MARK_PATH in svg and "prefers-color-scheme:dark" in svg
    assert "http" not in uri  # nothing that looks like an external reference


def test_dashboard_and_plain_gate_report_carry_the_mark(tmp_path):
    html = dashboard.build(tmp_path).read_text()
    assert html.count(brand.MARK_PATH) == 1 and 'rel="icon"' in html and "CellWarden run dashboard" in html
    gate = plain_gate_report("G1", tmp_path, {}, {}, None, None, "wait", None, [])
    assert brand.MARK_PATH in gate and 'rel="icon"' in gate and "http" not in gate.split("<body>")[0]
