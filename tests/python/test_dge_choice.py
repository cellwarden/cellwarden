"""dge.interactive: the DE comparisons are chosen before S4, once the cell types are known."""

import io
import json
import sys

import yaml

from cellwarden import config as C
from cellwarden import dge_choice as DC
from cellwarden import runner as R
from cellwarden import wizard as W


def scripted(answers):
    it = iter(answers)

    def ask(text, default):
        a = next(it)
        return default if a == "" else a

    return ask


def project(tmp_path, dge=None):
    proj = tmp_path / "proj"
    (proj / "data" / "S1").mkdir(parents=True)
    for f in ("matrix.mtx", "features.tsv", "barcodes.tsv"):
        (proj / "data" / "S1" / f).write_text("x")
    (proj / "data" / "S2.h5ad").write_text("x")
    (proj / "data" / "S3.h5ad").write_text("x")
    (proj / "samples.csv").write_text("sample_id,path,format,condition\nS1,data/S1,,ctrl\nS2,data/S2.h5ad,h5ad,stim\n"
                                      "S3,data/S3.h5ad,h5ad,stim\n")
    cfg = {"project": "demo", "run_dir": "runs/demo", "samples": "samples.csv", "dge": dge or {"interactive": True}}
    (proj / "config.yaml").write_text(yaml.safe_dump(cfg))
    resolved = C.prepare_run_dir(C.load_config(proj / "config.yaml"))
    rd = proj / "runs" / "demo"
    # what S3 and G3 leave behind: cluster sizes and the reviewed labels
    (rd / "S3").mkdir(parents=True)
    (rd / "S3" / "cluster_summary.json").write_text(json.dumps({"0": {"n_cells": 600}, "1": {"n_cells": 300},
                                                                "2": {"n_cells": 100}}))
    (rd / "gates" / "G3").mkdir(parents=True)
    (rd / "gates" / "G3" / "decision.json").write_text(json.dumps({"decision": {"clusters": [
        {"cluster": "0", "label_broad": "T cell", "label_fine": "CD4 T cell"},
        {"cluster": "1", "label_broad": "T cell", "label_fine": "CD8 T cell"},
        {"cluster": "2", "label_broad": "B cell", "label_fine": "B cell"}]}}))
    return rd, resolved


def s4_stage():
    code = "import os, yaml; d = yaml.safe_load(open(os.environ['CELLWARDEN_CONFIG']))['dge']; print('S4 contrasts', len(d['contrasts']))"
    return [R.Stage("S4", "python", lambda ctx: [sys.executable, "-c", code])]


def test_available_groups(tmp_path):
    rd, cfg = project(tmp_path)
    a = DC.available_groups(rd, cfg)
    assert a["cell"]["cell_type_broad"] == {"T cell": 900, "B cell": 100}
    assert a["cell"]["cell_type_fine"] == {"CD4 T cell": 600, "CD8 T cell": 300, "B cell": 100}
    assert a["cell"]["seurat_clusters"] == {"0": 600, "1": 300, "2": 100}
    assert a["sample"] == {"sample_id": {"S1": 1, "S2": 1, "S3": 1}, "condition": {"ctrl": 1, "stim": 2}}


def test_runner_asks_before_s4_and_keeps_the_choice(tmp_path):
    rd, cfg = project(tmp_path)
    assert DC.needs_choice(cfg, rd)
    answers = ["2",            # two cell types or clusters
               "2", "2", "rest",  # from fine cell types: CD8 T cell vs the rest
               "y",            # another one
               "3", "2", "stim", "1", "y",  # samples: condition stim vs ctrl, within each cell type
               "n"]            # done
    out = io.StringIO()
    assert R.run_pipeline(rd, cfg, stages=s4_stage(), out=out, progress="plain", ask=scripted(answers)) == 0
    text = out.getvalue()
    assert "T cell (900 cells)" in text and "condition: ctrl (1 sample), stim (2 samples)" in text
    res = C.load_resolved(rd)
    assert res["dge"]["markers"] is False and res["dge"]["contrasts"] == [
        {"name": "CD8-T-cell_vs_rest", "column": "cell_type_fine", "numerator": "CD8 T cell", "denominator": "rest",
         "within": "none"},
        {"name": "stim_vs_ctrl", "column": "condition", "numerator": "stim", "denominator": "ctrl",
         "within": "cell_type_broad"}]
    assert "S4 contrasts 2" in (rd / "logs" / "S4.log").read_text()  # S4 saw the choice
    assert (rd / "config" / DC.CHOICE_FILE).exists() and not DC.needs_choice(res, rd)
    # a rerun does not ask again (no answers left would raise)
    assert R.run_pipeline(rd, res, stages=s4_stage(), out=io.StringIO(), progress="plain", ask=scripted([])) == 0


def test_without_a_terminal_the_run_waits_before_s4(tmp_path, monkeypatch):
    rd, cfg = project(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    out = io.StringIO()
    assert R.run_pipeline(rd, cfg, stages=s4_stage(), out=out, progress="plain") == R.GATE_WAITING
    assert "waits for the comparisons" in out.getvalue()
    st = json.loads((rd / "state.json").read_text())
    assert st["stages"]["S4"]["status"] == "waiting"
    assert "waits for the comparisons to test" in R.status_text(rd, s4_stage())
    # markers chosen on resume in a terminal
    assert R.run_pipeline(rd, cfg, stages=s4_stage(), out=io.StringIO(), progress="plain", ask=scripted(["1", "n"])) == 0
    assert C.load_resolved(rd)["dge"]["markers"] is True


def test_pseudobulk_offers_sample_groups_only(tmp_path):
    rd, cfg = project(tmp_path, {"interactive": True, "method": "pseudobulk"})
    said = []
    choice = DC.ask_choice(cfg, rd, scripted(["1", "2", "stim", "ctrl", "n"]), said.append)
    assert choice == {"markers": False, "contrasts": [{"name": "stim_vs_ctrl", "column": "condition", "numerator": "stim",
                                                       "denominator": "ctrl", "within": "cell_type_broad"}]}
    assert not any("two cell types" in s for s in said)


def test_wizard_can_leave_the_choice_for_later(tmp_path):
    rd, cfg = project(tmp_path, {})
    q = W._Asker(scripted(["1", "1", "3"]), lambda m: None)  # FindMarkers, MAST, choose later
    assert W._ask_dge(q, lambda m: None, cfg) == {"method": "findmarkers", "test": "MAST", "interactive": True, "contrasts": []}
    q = W._Asker(scripted(["1", "1", "2", "cell_type_broad"]), lambda m: None)  # a cell-level column: values later
    assert W._ask_dge(q, lambda m: None, cfg)["interactive"] is True
