"""cellwarden run -i: the questions, their checks, and what the run gets from the answers."""

import io
import sys
from pathlib import Path

import pytest
import yaml

from cellwarden import runner as R
from cellwarden import wizard as W

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "stage_inputs"
NO_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NVIDIA_API_KEY")


def scripted(answers):
    it = iter(answers)
    asked = []

    def ask(text, default):
        asked.append((text, default))
        a = next(it)
        return default if a == "" else a

    ask.asked = asked
    return ask


def quick_stage():
    return [R.Stage("S0", "python", lambda ctx: [sys.executable, "-c", "pass"])]


@pytest.fixture
def no_keys(monkeypatch):
    for k in NO_KEYS:
        monkeypatch.delenv(k, raising=False)


def run_i(tmp_path, answers, extra=()):
    out, ask = io.StringIO(), scripted(answers)
    code = R.main(["run", str(FIX / "donor_A"), str(FIX / "donor_B.h5ad"), "-i", "--out", str(tmp_path / "run"),
                   "--progress", "plain", *extra], stages=quick_stage(), out=out, ask=ask)
    return code, out.getvalue(), ask


def test_guided_answers_reach_the_run(tmp_path, no_keys):
    answers = ["2",            # guided
               "300", "", "abc", "15",   # min genes, min UMIs (default), max mito: invalid then 15
               "off",          # adaptive QC off
               "3",            # doublets: none (no 'remove' question then)
               "", "0", "0.8", "",       # PCs auto, resolution: 0 refused then 0.8, HVGs default
               "1", "2",       # DE: FindMarkers, wilcox
               "2", "sample_id",          # compare two groups of the sample_id column
               "donor_Z", "",  # group 1: an unknown value refused, then the suggested donor_B
               "",             # group 2: the suggested donor_A
               "n",            # across all cells, not within each cell type
               "y",            # Harmony (two samples)
               "2",            # stop at every gate
               ""]             # start
    code, text, ask = run_i(tmp_path, answers)
    assert code == 0, text
    assert "please enter a number" in text and "must be above 0" in text
    assert "no API key found" in text  # rules only, so no LLM question
    cdir = tmp_path / "run" / "config"
    saved = yaml.safe_load((cdir / "wizard.yaml").read_text())
    assert saved["gates"] == {g: "review" for g in ("G1", "G2", "G3", "G4")}
    assert saved["doublets"] == {"method": "none"} and saved["clustering"] == {"resolution": 0.8}
    assert saved["pcs"] == "auto" and saved["n_hvg"] == 2000
    assert saved["integration"] == {"method": "harmony", "batch_key": "sample_id"}
    assert "not a value of sample_id: donor_Z" in text
    assert saved["dge"] == {"method": "findmarkers", "test": "wilcox", "markers": "auto", "contrasts": [
        {"name": "donor_B_vs_donor_A", "column": "sample_id", "numerator": "donor_B", "denominator": "donor_A",
         "within": "none"}]}
    assert "FindMarkers (wilcox); sample_id: donor_B vs donor_A" in text
    res = yaml.safe_load((cdir / "config.resolved.yaml").read_text())
    pol = yaml.safe_load((cdir / "qc_policy.yaml").read_text())
    assert pol["hard_limits"] == {"min_genes": 300, "min_umis": 400, "max_mito_pct": 15}
    assert pol["adaptive"]["metrics"] == []
    assert res["integration"]["method"] == "harmony" and res["gates"]["G2"] == "review"
    assert "--config" in text and str(cdir / "wizard.yaml") in text and "--llm none" in text
    assert "Tissue and disease context review: review" in text  # summary uses the stage names


def test_automatic_mode_asks_almost_nothing(tmp_path, no_keys):
    code, text, ask = run_i(tmp_path, ["1", "n", ""])  # automatic, no Harmony, start
    assert code == 0, text
    assert [t for t, _ in ask.asked] == ["Mode", "Correct batch effects between samples with Harmony (y/n)",
                                         "\nStart the run (y/n)"]
    saved = yaml.safe_load((tmp_path / "run" / "config" / "wizard.yaml").read_text())
    assert saved["gates"] == {g: "auto" for g in ("G1", "G2", "G3", "G4")} and "integration" not in saved


def test_declining_starts_nothing(tmp_path, no_keys):
    code, text, _ = run_i(tmp_path, ["1", "n", "n"])
    assert code == 0 and "nothing started" in text
    assert not (tmp_path / "run" / "state.json").exists()


def test_llm_choice_from_detected_keys(tmp_path, no_keys, monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    cfg = {"_policy": None, "gates": {"G1": "auto"}, "pcs": "auto", "clustering": {"resolution": 0.5}, "n_hvg": 2000}
    res = W.run_wizard(cfg, ask=scripted(["1", ""]), say=lambda m: None, n_samples=1)
    assert res.llm == "nvidia" and res.overrides["gates"]["G4"] == "auto"
    res = W.run_wizard(cfg, ask=scripted(["1", ""]), say=lambda m: None, n_samples=1, ask_llm=False)
    assert res.llm is None


def test_interactive_needs_a_terminal_and_no_config(tmp_path):
    out = io.StringIO()
    assert R.main(["run", str(FIX / "donor_A"), "-i", "--out", str(tmp_path / "r")], out=out) == 2
    assert "needs a terminal" in out.getvalue()
    out = io.StringIO()
    assert R.main(["run", str(FIX / "donor_A"), "-i", "--config", "x.yaml"], out=out, ask=scripted([])) == 2
    assert "without --config" in out.getvalue()
