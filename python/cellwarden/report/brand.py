"""The CellWarden mark, inlined into the generated pages so they stay self-contained.

The artwork's source of truth is ``docs/logo/`` (``mark-light.svg``). ``MARK_PATH`` is that
file's path data, and a test checks that the two stay the same.
"""

from __future__ import annotations

import base64

__all__ = ["MARK_PATH", "ACCENT_LIGHT", "ACCENT_DARK", "mark_svg", "favicon_uri", "favicon_link"]

# a shield with six cells (two clusters, as on a UMAP) cut out of it; 64 x 64 user units
MARK_PATH = ("M32 3.5C39.5 7.8 47 9.6 55.5 9.6V30C55.5 44.5 46 54.5 32 60.5C18 54.5 8.5 44.5 8.5 30V9.6"
             "C17 9.6 24.5 7.8 32 3.5ZM16.3 20.6a3.5 3.5 0 1 0 7 0a3.5 3.5 0 1 0 -7 0Z"
             "M24.7 22.4a4.9 4.9 0 1 0 9.8 0a4.9 4.9 0 1 0 -9.8 0ZM17.3 31.4a4.3 4.3 0 1 0 8.6 0"
             "a4.3 4.3 0 1 0 -8.6 0ZM35 35.4a6.6 6.6 0 1 0 13.2 0a6.6 6.6 0 1 0 -13.2 0Z"
             "M27.5 44.6a3.9 3.9 0 1 0 7.8 0a3.9 3.9 0 1 0 -7.8 0ZM42.6 20.2a2.6 2.6 0 1 0 5.2 0"
             "a2.6 2.6 0 1 0 -5.2 0Z")
ACCENT_LIGHT = "#2b6e5a"
ACCENT_DARK = "#6cc3a6"


def mark_svg(size: int = 24, fill: str = ACCENT_LIGHT, cls: str = "logo") -> str:
    """Inline ``<svg>`` of the mark. It is decorative next to the page title, so it is hidden
    from screen readers."""
    return (f'<svg class="{cls}" viewBox="0 0 64 64" width="{size}" '
            f'height="{size}" aria-hidden="true" focusable="false">'
            f'<path fill="{fill}" fill-rule="evenodd" d="{MARK_PATH}"/></svg>')


def favicon_uri() -> str:
    """``data:`` URI of the mark; it uses the dark accent when the browser is in dark mode.

    Base64 keeps the page free of any ``http`` text (the SVG namespace) outside the body.
    """
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><style>path{fill:%s}'
           '@media (prefers-color-scheme:dark){path{fill:%s}}</style>'
           '<path fill-rule="evenodd" d="%s"/></svg>') % (ACCENT_LIGHT, ACCENT_DARK, MARK_PATH)
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def favicon_link() -> str:
    return f'<link rel="icon" type="image/svg+xml" href="{favicon_uri()}">'
