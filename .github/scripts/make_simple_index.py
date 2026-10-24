"""Write a PEP 503 simple index of every wheel attached to this repository's GitHub Releases.

Published on GitHub Pages by .github/workflows/release.yml, so users install with
    pip install cellwarden --extra-index-url https://<owner>.github.io/<repo>/simple
Reads the releases with the gh CLI (GH_TOKEN and GITHUB_REPOSITORY set by Actions).
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def releases(repo: str) -> list[dict]:
    out = subprocess.run(["gh", "api", "--paginate", f"repos/{repo}/releases"],
                         check=True, capture_output=True, text=True).stdout
    # --paginate concatenates one JSON array per page
    return [r for page in json.loads("[" + out.replace("][", "],[") + "]") for r in page]


def main(repo: str, out_dir: Path) -> None:
    files: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rel in releases(repo):
        if rel.get("draft"):
            continue
        for asset in rel.get("assets", []):
            name = asset["name"]
            if not name.endswith(".whl"):
                continue
            url = asset["browser_download_url"]
            digest = asset.get("digest") or ""  # "sha256:<hex>" when GitHub provides it
            if digest.startswith("sha256:"):
                url += "#sha256=" + digest.split(":", 1)[1]
            files[normalize(name.split("-", 1)[0])].append((name, url))

    root = out_dir / "simple"
    root.mkdir(parents=True, exist_ok=True)
    links = "".join(f'<a href="{p}/">{p}</a><br>\n' for p in sorted(files))
    (root / "index.html").write_text(f"<!DOCTYPE html><html><body>\n{links}</body></html>\n")
    for project, wheels in files.items():
        (root / project).mkdir(exist_ok=True)
        body = "".join(f'<a href="{html.escape(u)}">{html.escape(n)}</a><br>\n' for n, u in sorted(wheels))
        (root / project / "index.html").write_text(f"<!DOCTYPE html><html><body>\n{body}</body></html>\n")
    print(f"indexed {sum(map(len, files.values()))} wheels for {sorted(files)}")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
