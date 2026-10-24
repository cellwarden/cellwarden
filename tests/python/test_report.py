"""Tests for the S5 report and explorer (docs/agents-spec.md section 7): cellwarden.report.*
and the S5 stage entry point, on a synthetic run directory (no R needed)."""

import gzip
import io
import json
import os
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest
import yaml

from cellwarden.report import build, plots

REPO = Path(__file__).resolve().parents[2]
SECTIONS = ("summary", "resources", "qc", "clusters", "markers", "dge", "opentargets", "gates", "llm", "software")


# --------------------------------------------------------------------------
# A synthetic run directory with every input the report reads
# --------------------------------------------------------------------------


def _w(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, (dict, list)):
        path.write_text(json.dumps(obj, indent=1))
    else:
        path.write_text(str(obj))
    return path


def _tsv_gz(path: Path, header, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    return path


def make_run(rd: Path, n_cells: int = 300, with_gates: bool = True, with_llm: bool = True) -> Path:
    cfg = {"project": "demo", "run_id": "demo-0001", "run_dir": str(rd), "seed": 7, "execution": "direct",
           "site": "workstation", "device": "cpu", "config_hash": "abc123",
           "resources": {"threads": 2, "max_memory_gb": 4, "gpu_mem_cap_gb": 2, "nice": 10},
           "container": {"runtime": "none", "image": None, "options": []},
           "report": {"explorer_max_cells": 100000, "top_markers": 5},
           "llm": {"default": {"provider": "mock", "model": "mock-model"}, "budget": {"per_run_limit": 5, "total_limit": 30}},
           "samples_resolved": [{"sample_id": "A", "path": "/x/a"}, {"sample_id": "B", "path": "/x/b"}]}
    (rd / "config").mkdir(parents=True, exist_ok=True)
    (rd / "config" / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    stages = {}
    for i, sid in enumerate(["S0", "S1a", "S1b", "S1c", "S2", "S3", "S4", "S5"]):
        stages[sid] = {"status": "done", "lang": "R" if sid in ("S0", "S1a", "S2") else "python",
                       "started_utc": f"2026-01-01T00:0{i}:00Z", "wall_s": 3.5 * (i + 1), "peak_rss_bytes": 10**8 * (i + 1),
                       "exit_code": 0}
    stages["S5"]["status"] = "running"
    _w(rd / "state.json", {"run_id": "demo-0001", "stage_order": list(stages), "stages": stages, "runner": {"host": "node1"}})
    _w(rd / "S0" / "ingest_report.json", {
        "samples": [{"sample_id": "A", "format": "10x_mtx", "format_detected": True, "cells": 160, "genes": 800, "total_umis": 300000, "notes": [], "warnings": []},
                    {"sample_id": "B", "format": "h5ad", "cells": 140, "genes": 796, "total_umis": 280000, "notes": ["4 genes missing"], "warnings": []}],
        "totals": {"cells": 300, "genes": 800, "total_umis": 580000}, "gene_key": "symbol"})
    _w(rd / "S1" / "qc_summary.json", {
        "policy_hash": "deadbeef", "decided_by": "rules",
        "thresholds": {"hard_limits": {"min_genes": 40, "min_umis": 80, "max_mito_pct": 25},
                       "per_sample": {s: {"adaptive_applied": True, "note": "", "adaptive_natural_scale": {"min_genes": 60, "max_genes": 900, "min_umis": 120, "max_umis": 4000, "max_mito_pct": 12.5}} for s in "AB"}},
        "counts_per_sample": {"A": {"keep": 140, "exclude": 15, "review": 5}, "B": {"keep": 120, "exclude": 14, "review": 6}},
        "counts_per_reason": {"HIGH_MT": 20, "DOUBLET": 9, "PROTECTED_POP:Platelet": 11},
        "protected_losses": [{"name": "Platelet", "sample": "A", "lost_fraction": 0.1}], "review_triggers_fired": []})
    import random

    rng = random.Random(1)
    _tsv_gz(rd / "S1" / "qc_metrics.tsv.gz", ["cell_id", "sample_id", "n_genes", "n_umis", "pct_mito"],
            [(f"{s}_{i}", s, rng.randint(50, 900), rng.randint(100, 4000), round(rng.random() * 20, 2))
             for s in "AB" for i in range(n_cells // 2)])
    _w(rd / "S2" / "summary.json", {"device": "cpu", "plan": "cpu", "reason": "device: cpu", "path": "seurat",
                                    "resolution": 0.5, "pcs": 12, "n_neighbors": 20, "integration": {"method": "none"},
                                    "cluster_sizes": {"0": 120, "1": 100, "2": 60, "3": 20},
                                    "packages": {"Seurat": "5.1.0", "R": "4.4.2"}})
    # S5 exports (what R writes): UMAP table and the sparse expression subset.
    clusters = ["0"] * 120 + ["1"] * 100 + ["2"] * 60 + ["3"] * 20
    labels = {"0": "T cell", "1": "B cell", "2": "Monocyte", "3": "NK cell"}
    rows = []
    for i in range(n_cells):
        cl = clusters[i % len(clusters)]
        rows.append((f"{'A' if i % 2 else 'B'}_{i}", round(rng.gauss(int(cl) * 3, 1), 3), round(rng.gauss(int(cl), 1), 3),
                     cl, "A" if i % 2 else "B", labels[cl], labels[cl] + " fine", rng.randint(50, 900), rng.randint(100, 4000),
                     round(rng.random() * 20, 2), "control" if i % 4 else "disease"))
    _tsv_gz(rd / "S5" / "umap.tsv.gz", ["cell_id", "umap_1", "umap_2", "seurat_clusters", "sample_id", "cell_type_broad",
                                         "cell_type_fine", "n_genes", "n_umis", "pct_mito", "condition"], rows)
    expr = []
    for g in ("CD3E", "MS4A1", "LYZ", "NKG7"):
        for i in range(0, n_cells, 3):
            expr.append((g, rows[i][0], round(rng.random() * 3, 3)))
    _tsv_gz(rd / "S5" / "expr_subset.tsv.gz", ["gene", "cell_id", "value"], expr)
    _w(rd / "S5" / "export.json", {"cells": n_cells, "reduction": "umap"})
    # S3
    _tsv_gz(rd / "S3" / "markers.tsv.gz", ["cluster", "gene", "avg_log2fc", "pct_in", "pct_out", "p_adj", "rank"],
            [(cl, f"G{cl}_{r}", 2.0 - r * 0.1, 0.9, 0.1, 1e-10 * (r + 1), r + 1) for cl in "0123" for r in range(20)])
    _w(rd / "S3" / "cluster_summary.json", {cl: {"n_cells": n, "pct_cells": 100.0 * n / 300, "median_genes": 400, "median_umis": 1500,
                                                  "median_pct_mito": 4.2, "top_markers": [f"G{cl}_{r}" for r in range(20)]}
                                            for cl, n in (("0", 120), ("1", 100), ("2", 60), ("3", 20))})
    (rd / "S3" / "annotation.tsv").write_text("cluster\tlabel_broad\tlabel_fine\tcl_id\tconfidence\tstatus\n" + "".join(
        f"{cl}\t{labels[cl]}\t{labels[cl]} fine\tCL:000{cl}\t0.9\tconsensus\n" for cl in "0123"))
    # S4
    _w(rd / "S4" / "dge_summary.json", {
        "method": {"requested": "deseq2", "used": ["limma_voom"]}, "group_by": {"requested": "cell_type_broad", "used": "cell_type_broad"},
        "thresholds": {"padj": 0.05, "lfc": 0.5, "min_cells_per_sample": 20, "min_replicates": 2},
        "contrasts": [{"name": "disease_vs_control", "numerator": "disease", "denominator": "control", "covariates": [],
                       "groups": [{"group": "T cell", "method_used": "limma_voom", "n_tested": 500, "n_up": 12, "n_down": 7,
                                   "n_samples": {"numerator": 2, "denominator": 2},
                                   "top_up": [{"gene": "IFI44L", "log2fc": 2.1, "padj": 1e-5}], "top_down": ["ACTB"]},
                                  {"group": "NK cell", "n_tested": 0, "n_up": 0, "n_down": 0, "n_samples": {"numerator": 1, "denominator": 2},
                                   "skipped_reason": "1 disease pseudobulk sample(s); min_replicates is 2"}]}]})
    _w(rd / "S4" / "opentargets_summary.json", {
        "offline": True, "counts": {"genes_queried": 2, "with_record": 1},
        "contrasts": [{"name": "disease_vs_control", "groups": [{"group": "T cell", "directions": {
            "up": {"n_de": 12, "n_listed": 2, "genes": [
                {"gene": "IFI44L", "status": "ok", "target": {"ensembl_id": "ENSG00000137959", "tractability": "small molecule",
                                                                "known_drugs": [{"drug": "DrugX", "phase": 2, "disease": "lupus", "mechanism": "inhibitor"}],
                                                                "associated_diseases": [{"efo_id": "EFO_1", "name": "lupus", "score": 0.81}]}},
                {"gene": "ZZZ9", "status": "no_record"}]}}}]}],
        "genes_without_record": ["ZZZ9"], "errors": []})
    if with_gates:
        g = rd / "gates" / "G1"
        _w(g / "bundle.json", {"gate": "G1", "per_sample": {"A": {"keep": 140}}})
        _w(g / "proposal.json", {"gate": "G1", "worker": {"model": "mock-model", "role": "qc_worker"},
                                 "proposal": {"thresholds": {"A": {"max_mito_pct": 12.5}}}, "rationale": "Mito tail looks fine.",
                                 "policy_check": {"ok": True, "violations": []}, "fallback": False})
        _w(g / "review.json", {"verdict": "approve", "findings": [{"severity": "info", "text": "Within policy."}], "reviewer": {"model": "mock-model"}})
        _w(g / "decision.json", {"gate": "G1", "mode": "auto", "decided_by": "agent", "decided_utc": "2026-01-01T00:05:00Z",
                                 "decision": {"thresholds": {"A": {"max_mito_pct": 12.5}}}, "changes_vs_proposal": []})
        g2 = rd / "gates" / "G2"
        _w(g2 / "bundle.json", {"gate": "G2"})
        _w(g2 / "proposal.json", {"gate": "G2", "worker": {"model": "mock-model"}, "proposal": {"tissue": {"label": "blood"}},
                                  "rationale": "PBMC markers.", "policy_check": {"ok": True, "violations": []}, "fallback": False})
        _w(g2 / "review.json", {"verdict": "escalate", "findings": [{"severity": "warn", "text": "Confirm tissue."}], "reviewer": {"model": "mock-model"}})
        _w(g2 / "pending.json", {"gate": "G2", "mode": "review"})
    if with_llm:
        _w(rd / "llm" / "ledger.json", {"run_id": "demo-0001", "total_gbp": 0.0123, "entries": [
            {"ts": "t", "gate": "G1", "role": "qc_worker", "model": "mock-model", "usage": {"input_tokens": 1000, "output_tokens": 200, "cached_tokens": 0}, "cost_gbp": 0.01},
            {"ts": "t", "gate": "G1", "role": "reviewer", "model": "mock-model", "usage": {"input_tokens": 500, "output_tokens": 50, "cached_tokens": 100}, "cost_gbp": 0.0023}]})
    return rd


@pytest.fixture
def run_dir(tmp_path):
    return make_run(tmp_path / "run")


class _Check(HTMLParser):
    """Collects ids, external references and unbalanced tags."""

    VOID = {"meta", "link", "img", "br", "hr", "input", "rect", "line", "circle", "path", "source", "wbr"}

    def __init__(self):
        super().__init__()
        self.ids, self.external, self.stack, self.errors = set(), [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.ids.add(a["id"])
        for k in ("src", "href"):
            v = a.get(k) or ""
            if v.startswith(("http://", "https://", "//")):
                self.external.append(v)
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unbalanced </{tag}> (open: {self.stack[-3:]})")
            return
        self.stack.pop()


def parse(html: str) -> _Check:
    p = _Check()
    p.feed(html)
    p.close()
    return p


# --------------------------------------------------------------------------
# report.html
# --------------------------------------------------------------------------


def test_full_report_has_every_section_and_is_self_contained(run_dir):
    out = build.build_report(run_dir)
    assert out == run_dir / "S5" / "report.html" and out.exists()
    html = out.read_text()
    p = parse(html)
    assert not p.errors, p.errors[:5]
    assert not p.stack, p.stack
    for sec in SECTIONS:
        assert sec in p.ids, f"section #{sec} missing"
    assert not p.external, p.external            # no CDN, no external resources
    assert "<script src" not in html and "<link" not in html
    # plots are inline: SVG (always) and base64 PNG when matplotlib is installed
    assert "<svg" in html
    try:
        import matplotlib  # noqa: F401

        assert "data:image/png;base64," in html
    except ImportError:
        pass
    # content from each stage
    assert "demo-0001" in html and "node1" in html
    assert "10x_mtx" in html and "580,000" in html                        # S0
    assert "deadbeef" in html and "HIGH_MT" in html                       # S1
    assert "UMAP by cluster" in html and "UMAP by broad cell type" in html and "Monocyte" in html  # S2 / S5 export
    assert "G0_0" in html and "G3_4" in html and "G0_12" not in html     # markers: top 5 per cluster (report.top_markers)
    assert html.count("<h3>Cluster ") == 4
    assert "disease_vs_control" in html and "IFI44L" in html and "min_replicates is 2" in html  # S4
    assert "DrugX" in html and "lupus" in html and "offline" in html       # Open Targets
    assert "Mito tail looks fine." in html and "decided by" in html and "agent" in html  # gates
    assert "waiting for a human" in html                                  # G2 pending
    assert "not run" in html                                              # G3/G4 absent
    assert "GBP 0.0123" in html and "G1/qc_worker" in html and "mock" in html  # LLM cost from llm/ledger.json
    assert "running" in html and "Seurat" in html                         # stage table and software


def test_empty_run_dir_degrades_gracefully(tmp_path):
    rd = tmp_path / "empty"
    (rd / "config").mkdir(parents=True)
    (rd / "config" / "config.resolved.yaml").write_text("project: p\nrun_id: r1\n")
    ctx = build.collect(rd)
    assert set(ctx["missing"]) >= {"S0 ingest", "S1 QC", "S3 markers", "S4 DGE", "S4 Open Targets", "LLM ledger"}
    html = build.render_report(ctx)
    p = parse(html)
    assert not p.errors and not p.stack
    for sec in SECTIONS:
        assert sec in p.ids
    assert html.count("not run") >= 8
    assert "no state.json" in html
    # nothing at all (not even a config) still renders
    html2 = build.render_report(build.collect(tmp_path / "nothing"))
    assert "<html" in html2 and "gates" in html2


def test_partial_run_dir_only_s0_s2(run_dir):
    import shutil

    for d in ("S3", "S4", "gates", "llm"):
        shutil.rmtree(run_dir / d)
    html = build.build_report(run_dir).read_text()
    assert "UMAP by cluster" in html and "S3 markers: not run" in html and "S4 DGE: not run" in html
    assert "No LLM calls recorded" in html
    assert "Per-cluster medians and top markers come from S3 (not run)" in html
    assert not parse(html).errors


def test_report_escapes_data(run_dir):
    s = json.loads((run_dir / "S4" / "dge_summary.json").read_text())
    s["notes"] = ["<script>alert(1)</script>"]
    (run_dir / "S4" / "dge_summary.json").write_text(json.dumps(s))
    html = build.build_report(run_dir).read_text()
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html


# --------------------------------------------------------------------------
# Gate reports
# --------------------------------------------------------------------------


def test_render_gate_report(run_dir):
    html = build.render_gate_report(run_dir / "gates" / "G1")
    p = parse(html)
    assert not p.errors and not p.stack
    assert "Gate G1" in html and "QC review" in html
    assert "max_mito_pct" in html                        # decision JSON shown in full
    assert "Mito tail looks fine." in html and "Within policy." in html
    assert "cellwarden approve --run-dir" in html and "--stage qc-review" in html
    assert "GBP 0.0123" in html                          # LLM cost so far
    assert not p.external
    # a gate waiting for a human
    html2 = build.render_gate_report(run_dir / "gates" / "G2")
    assert "waiting for a human" in html2 and "escalate" in html2 and "Confirm tissue." in html2
    # an empty gate directory renders a "not run" page rather than failing
    html3 = build.render_gate_report(run_dir / "gates" / "G4")
    assert "not run" in html3 and "Gate G4" in html3


def test_gate_report_link_is_relative(run_dir):
    (run_dir / "gates" / "G1" / "report.html").write_text("x")
    html = build.build_report(run_dir).read_text()
    assert 'href="../gates/G1/report.html"' in html


# --------------------------------------------------------------------------
# explorer.html
# --------------------------------------------------------------------------


def test_explorer_payload_and_html(run_dir):
    payload = build.explorer_payload(run_dir)
    n = payload["n"]
    assert n == payload["n_total"] == 300
    assert len(payload["x"]) == len(payload["y"]) == len(payload["ids"]) == n
    assert set(payload["cats"]) >= {"seurat_clusters", "cell_type_broad", "cell_type_fine", "sample_id", "condition"}
    assert payload["cats"]["seurat_clusters"]["levels"] == ["0", "1", "2", "3"]
    assert set(payload["nums"]) >= {"n_genes", "n_umis", "pct_mito"}
    assert set(payload["genes"]) == {"CD3E", "MS4A1", "LYZ", "NKG7"}
    g = payload["genes"]["CD3E"]
    assert len(g["i"]) == len(g["v"]) == 100 and max(g["i"]) < n
    out = build.build_explorer(run_dir)
    html = out.read_text()
    assert out == run_dir / "S5" / "explorer.html"
    assert '<script id="data" type="application/json">' in html and "<canvas" in html
    assert "getContext('2d')" in html
    assert "300 of 300 cells shown" in html
    p = parse(html)
    assert not p.errors and not p.external
    # the embedded JSON round-trips
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    data = json.loads(m.group(1))
    assert data["n"] == 300 and "palette" in data


def test_explorer_subsamples_above_cap(run_dir):
    payload = build.explorer_payload(run_dir, max_cells=50, seed=1)
    assert payload["n_total"] == 300 and payload["n"] == 50
    assert len(payload["x"]) == 50 and len(payload["cats"]["seurat_clusters"]["idx"]) == 50
    # cells with expression data are kept preferentially, and gene indices stay in range
    assert payload["expr_cells"] == 50
    for g in payload["genes"].values():
        assert all(0 <= i < 50 for i in g["i"])
    # deterministic for a seed
    again = build.explorer_payload(run_dir, max_cells=50, seed=1)
    assert again["ids"] == payload["ids"]
    # config cap is honoured by build_explorer (report.explorer_max_cells)
    cfgp = run_dir / "config" / "config.resolved.yaml"
    cfg = yaml.safe_load(cfgp.read_text())
    cfg["report"]["explorer_max_cells"] = 40
    cfgp.write_text(yaml.safe_dump(cfg))
    html = build.build_explorer(run_dir).read_text()
    assert "40 of 300 cells shown" in html


def test_explorer_json_cannot_break_out_of_script(run_dir):
    with gzip.open(run_dir / "S5" / "umap.tsv.gz", "rt") as fh:
        lines = fh.read().splitlines()
    lines[1] = lines[1].replace("B_0", "</script><b>x")
    with gzip.open(run_dir / "S5" / "umap.tsv.gz", "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    html = build.build_explorer(run_dir).read_text()
    assert html.count("</script>") == 2  # the data block and the code block only
    assert "<\\/script><b>x" in html


def test_explorer_without_expression_subset(run_dir):
    (run_dir / "S5" / "expr_subset.tsv.gz").unlink()
    payload = build.explorer_payload(run_dir)
    assert payload["genes"] == {} and payload["expr_cells"] == 0
    html = build.build_explorer(run_dir).read_text()
    assert "0 genes available" in html


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------


def test_plots_are_inline_and_tolerant():
    assert plots.bar_chart([("a", 1), ("b", 3)], title="t").startswith("<svg")
    assert "no data" in plots.bar_chart([], title="t")
    svg = plots.histogram([1, 10, 100, 1000], title="h", log10=True, vlines={"min": 5, "none": None})
    assert svg.startswith("<svg") and "min = 5" in svg
    assert "no data" in plots.histogram([], title="h")
    st = plots.stacked_bars([("A", {"keep": 10, "exclude": 2})], ["keep", "exclude"])
    assert st.count("<rect") >= 4  # two bars + two legend swatches
    # scatter: SVG fallback subsamples, PNG when matplotlib exists
    s = plots.scatter([0, 1, 2], [0, 1, 2], ["a", "b", "a"], title="u", prefer_png=False)
    assert "<svg" in s and "3 cells" in s and s.count("<li>") == 2
    s2 = plots.scatter(list(range(50)), list(range(50)), values=list(range(50)), prefer_png=False)
    assert "<circle" in s2
    assert plots.viridis(0) == "#440154" and plots.viridis(1) == "#fde725"
    assert plots.scatter([], [], [], title="e") .startswith("<p")


# --------------------------------------------------------------------------
# The S5 stage entry point (python -m cellwarden.stages.s5_report)
# --------------------------------------------------------------------------


def test_s5_stage_with_existing_export(run_dir):
    (run_dir / "S2").mkdir(exist_ok=True)
    (run_dir / "S2" / "object.rds").write_bytes(b"not a real object")
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"), CELLWARDEN_RSCRIPT="/nonexistent/Rscript")
    r = subprocess.run([sys.executable, "-m", "cellwarden.stages.s5_report", "--run-dir", str(run_dir), "--skip-export"],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (run_dir / "S5" / "report.html").exists() and (run_dir / "S5" / "explorer.html").exists()
    assert "reusing existing S5/umap.tsv.gz" in r.stdout
    assert "sections not run: none" in r.stdout


def test_s5_stage_needs_the_s2_object(tmp_path):
    rd = tmp_path / "r"
    (rd / "config").mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"))
    r = subprocess.run([sys.executable, "-m", "cellwarden.stages.s5_report", "--run-dir", str(rd)],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode != 0 and "S2/object.rds" in (r.stdout + r.stderr)


def test_s5_is_registered_last_in_the_runner():
    from cellwarden import runner

    ids = [s.id for s in runner.STAGES]
    assert ids[-1] == "S5"
    s5 = runner.STAGES[-1]
    assert "S5/report.html" in s5.outputs and "S5/explorer.html" in s5.outputs
    assert any("llm/ledger.json" in x for x in s5.optional_inputs)


def test_opentargets_section_reads_the_stage_shape(run_dir):
    """S4 writes genes with `target` = a key into the summary's `targets` table."""
    p = run_dir / "S4" / "opentargets_summary.json"
    _w(p, {"offline": False, "counts": {}, "targets": {"MS4A1": {
               "ensembl_id": "ENSG00000156738", "known_drugs": [{"drug": "RITUXIMAB", "phase": 4, "disease": "lymphoma"}],
               "associated_diseases": [{"name": "follicular lymphoma", "score": 0.9}]}},
           "contrasts": [{"contrast": "markers", "groups": [{"group": "B cell", "directions": {"up": {"n_de": 240, "n_listed": 2, "genes": [
               {"gene": "MS4A1", "lookup_status": "record", "target": "MS4A1"},
               {"gene": "XYZ", "lookup_status": "none", "target": None}]}}}]}]})
    ot = build._opentargets(run_dir)
    genes = ot["contrasts"][0]["groups"][0]["directions"][0]["genes"]
    assert ot["contrasts"][0]["name"] == "markers"
    assert genes[0]["ensembl"] == "ENSG00000156738" and genes[0]["drugs"] == ["RITUXIMAB (ph 4, lymphoma)"]
    assert genes[0]["status"] == "record" and genes[1]["drugs"] == []
