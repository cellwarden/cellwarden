"""Tests for cellwarden.annotation.consensus and markers_human.yaml (docs/agents-spec.md section 5).

Synthetic evidence only (no R, no pipeline run); the real fixture is covered by test_s3.py.
Run from the repo root: PYTHONPATH=python python3 -m pytest -q tests/python/test_annotation.py
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest
import yaml

from cellwarden.annotation import consensus as cons

REPO = Path(__file__).resolve().parents[2]
SIG_FILE = REPO / "R" / "stages" / "data" / "signatures.yaml"


@pytest.fixture(scope="module")
def ref():
    return cons.load_reference()


@pytest.fixture(scope="module")
def sig_sets():
    y = yaml.safe_load(SIG_FILE.read_text())["sets"]
    return {k: {"category": v.get("category", "lineage"), "cell_type": v.get("cell_type"),
                "n_genes_set": len(v["genes"]), "n_genes_found": len(v["genes"]), "genes": v["genes"]}
            for k, v in y.items()}


# ----------------------------------------------------------------- reference

def test_reference_has_40_plus_types_with_cl_ids_and_tissue_coverage(ref):
    assert len(ref.types) >= 40
    for t in ref.types:
        assert t.cl_id.startswith("CL:") and t.cl_id[3:].isdigit() and len(t.cl_id) == 10, t
        assert len(t.markers) >= 3
        assert t.label_broad and t.label_fine
    assert len(ref.by_fine) == len(ref.types)
    tissues = set().union(*(set(t.tissue) for t in ref.types))
    for need in ("blood", "immune", "epithelial", "stromal", "neural", "liver", "lung", "kidney", "gut"):
        assert need in tissues, need
    # every broad group with several fine labels has a generic entry to fall back to
    for b, members in ref.by_broad.items():
        if len(members) > 1:
            assert ref.generic_of(b) is not None, b
    # celltypist aliases are unique and map back
    assert ref.map_celltypist("Classical monocytes") == "CD14 monocyte"
    assert ref.map_celltypist("tcm/naive helper t cells") == "CD4 T cell"
    assert ref.map_celltypist("Some unknown label") is None


def test_every_signature_cell_type_is_a_reference_label(ref, sig_sets):
    for name, s in sig_sets.items():
        if s["category"] == "lineage":
            assert s["cell_type"] in ref.by_fine, f"signature {name}: cell_type {s['cell_type']!r} not in markers_human.yaml"
        else:
            assert s["cell_type"] is None
        assert 5 <= len(s["genes"]) <= 15, name
        assert len(set(s["genes"])) == len(s["genes"]), name
    for need in ("interferon", "inflammation", "proliferation", "hypoxia", "platelet", "erythrocyte", "T", "CD4_T",
                 "CD8_T", "NK", "B", "plasma", "mono_CD14", "mono_CD16", "cDC", "pDC", "neutrophil", "mast",
                 "epithelial", "endothelial", "fibroblast", "hepatocyte", "neuron", "glia"):
        assert need in sig_sets


def test_reference_rejects_bad_files(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"cell_types": [{"label_fine": "x", "label_broad": "x", "cl_id": "GO:1", "markers": ["A", "B", "C"]}]}))
    with pytest.raises(ValueError, match="CL:"):
        cons.load_reference(p)
    p.write_text(yaml.safe_dump({"cell_types": []}))
    with pytest.raises(ValueError):
        cons.load_reference(p)


# ----------------------------------------------------------------- synthetic evidence

def _sig(scores: dict[str, float], all_sets, base=-0.05) -> dict[str, float]:
    return {s: scores.get(s, base) for s in all_sets}


def synthetic_clusters(sig_sets, celltypist=False):
    names = list(sig_sets)
    cl = {
        "0": {"top_markers": ["CD3D", "CD3E", "IL7R", "LTB", "CD2", "TRAC", "CCR7", "LEF1", "TCF7", "MAL"],
              "signatures": _sig({"T": 1.2, "CD4_T": 1.0}, names), "celltypist": None},
        "1": {"top_markers": [{"gene": g} for g in ["LYZ", "S100A8", "S100A9", "CD14", "FCN1", "VCAN", "CST3", "TYROBP"]],
              "signatures": _sig({"mono_CD14": 1.5, "neutrophil": 0.4}, names), "celltypist": None},
        "2": {"top_markers": ["MS4A1", "CD79A", "CD79B", "BANK1", "CD19", "TCL1A", "IGHD", "FCER2"],
              "signatures": _sig({"B": 1.3}, names), "celltypist": None},
        "3": {"top_markers": ["NKG7", "GNLY", "KLRD1", "PRF1", "GZMB", "FCGR3A", "KLRF1", "CTSW"],
              "signatures": _sig({"NK": 1.4, "CD8_T": 0.5}, names), "celltypist": None},
        "4": {"top_markers": ["PPBP", "PF4", "GNG11", "TUBB1", "GP9", "CLU"],
              "signatures": _sig({"platelet": 3.0}, names), "celltypist": None},
    }
    if celltypist:
        cl["0"]["celltypist"] = {"majority_voting": {"Tcm/Naive helper T cells": 0.85, "Tem/Trm cytotoxic T cells": 0.15}}
        cl["1"]["celltypist"] = {"majority_voting": {"Classical monocytes": 0.95, "DC2": 0.05}}
        cl["2"]["celltypist"] = {"majority_voting": {"Naive B cells": 0.9, "Memory B cells": 0.1}}
        cl["3"]["celltypist"] = {"majority_voting": {"CD16+ NK cells": 1.0}}
        cl["4"]["celltypist"] = {"majority_voting": {"Megakaryocytes/platelets": 0.7, "Mystery cells": 0.3}}
    return cl


def test_synthetic_consensus_without_celltypist(ref, sig_sets):
    recs = cons.annotate(synthetic_clusters(sig_sets), sig_sets, ref)
    by = {r["cluster"]: r for r in recs}
    assert [r["cluster"] for r in recs] == ["0", "1", "2", "3", "4"]
    assert by["0"]["label_broad"] == "T cell" and by["0"]["cl_id"] == "CL:0000084"
    assert by["1"]["label_broad"] == "monocyte" and by["1"]["label_fine"] == "CD14 monocyte"
    assert by["2"]["label_broad"] == "B cell"
    assert by["3"]["label_broad"] == "NK cell" and by["3"]["cl_id"] == "CL:0000623"
    assert by["4"]["label_broad"] == "platelet" and by["4"]["cl_id"] == "CL:0000233"
    for r in recs:
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["status"] in ("consensus", "low_confidence", "conflict")
        v = json.loads(r["votes_json"])
        assert v["sources"] == ["signature", "markers"]
        assert set(v) >= {"signature", "markers", "celltypist", "broad_scores", "fine_scores", "source_top", "weights"}
        assert r["cl_id"] == ref.by_fine[r["label_fine"]].cl_id
    assert all(r["status"] == "consensus" for r in recs), [(r["cluster"], r["confidence"], r["status"]) for r in recs]


def test_celltypist_votes_refine_fine_labels_and_unmapped_labels_are_kept(ref, sig_sets):
    recs = cons.annotate(synthetic_clusters(sig_sets, celltypist=True), sig_sets, ref)
    by = {r["cluster"]: r for r in recs}
    assert by["0"]["label_fine"] == "CD4 T cell" and by["0"]["cl_id"] == "CL:0000624"
    assert by["2"]["label_fine"] == "naive B cell" and by["2"]["cl_id"] == "CL:0000788"
    assert by["1"]["label_fine"] == "CD14 monocyte"
    assert by["3"]["label_broad"] == "NK cell"
    v4 = json.loads(by["4"]["votes_json"])
    assert v4["sources"] == ["signature", "celltypist", "markers"]
    assert v4["celltypist_unmapped"] == {"Mystery cells": 0.3}
    assert v4["celltypist"]["platelet"] == pytest.approx(0.7)
    assert all(r["status"] == "consensus" for r in recs)
    # celltypist alone is also a valid source set (no signatures)
    recs2 = cons.annotate(synthetic_clusters(sig_sets, celltypist=True), {}, ref)
    assert json.loads(recs2[0]["votes_json"])["sources"] == ["celltypist", "markers"]
    assert {r["cluster"]: r["label_broad"] for r in recs2} == {r["cluster"]: r["label_broad"] for r in recs}


def test_conflict_low_confidence_and_unassigned(ref, sig_sets):
    cl = synthetic_clusters(sig_sets, celltypist=True)
    # strong celltypist vote against strong signature + marker evidence -> conflict
    cl["0"]["celltypist"] = {"majority_voting": {"Classical monocytes": 0.9, "T cells": 0.1}}
    # a cluster with nothing: no markers, no signal
    cl["5"] = {"top_markers": [], "signatures": _sig({}, list(sig_sets)), "celltypist": None}
    # weak, ambiguous evidence: a couple of shared markers only
    cl["6"] = {"top_markers": ["S100A8", "S100A9", "ACTB"], "signatures": _sig({}, list(sig_sets)),
               "celltypist": {"majority_voting": {"Mystery cells": 1.0}}}
    recs = {r["cluster"]: r for r in cons.annotate(cl, sig_sets, ref)}
    assert recs["0"]["status"] == "conflict"
    assert set(json.loads(recs["0"]["votes_json"])["source_top"].values()) == {"T cell", "monocyte"}
    assert recs["5"]["label_broad"] == "Unassigned" and recs["5"]["label_fine"] == "Unassigned"
    assert recs["5"]["cl_id"] == "" and recs["5"]["confidence"] == 0.0 and recs["5"]["status"] == "low_confidence"
    assert recs["6"]["status"] == "low_confidence" and recs["6"]["confidence"] < 0.6
    assert recs["6"]["label_broad"] != "Unassigned"
    # min_confidence is honoured
    strict = {r["cluster"]: r for r in cons.annotate(cl, sig_sets, ref, min_confidence=0.99)}
    for c in ("1", "2", "3", "4"):
        assert strict[c]["status"] == ("low_confidence" if strict[c]["confidence"] < 0.99 else "consensus"), strict[c]
    assert any(strict[c]["status"] == "low_confidence" for c in ("1", "2", "3", "4"))
    # only the two sources that are present can be strong: the conflict must involve both
    assert strict["0"]["status"] == "conflict"


def test_deterministic_and_order_independent(ref, sig_sets):
    cl = synthetic_clusters(sig_sets, celltypist=True)
    a = cons.annotate(cl, sig_sets, ref)
    b = cons.annotate(copy.deepcopy(cl), sig_sets, ref)
    assert a == b
    keys = list(cl)
    random.Random(3).shuffle(keys)
    shuffled = {k: cl[k] for k in keys}
    c = {r["cluster"]: r for r in cons.annotate(shuffled, dict(reversed(list(sig_sets.items()))), ref)}
    assert [r["cluster"] for r in cons.annotate(shuffled, sig_sets, ref)] == keys
    for r in a:
        assert c[r["cluster"]] == r
    df = cons.consensus_table(a)
    assert list(df.columns) == cons.CONSENSUS_COLUMNS
    assert len(df) == len(cl)
    assert df["confidence"].between(0, 1).all()


def test_signature_coverage_and_sign_rules(ref):
    sets = {"T": {"category": "lineage", "cell_type": "T cell", "n_genes_set": 13, "n_genes_found": 13},
            "weak": {"category": "lineage", "cell_type": "NK cell", "n_genes_set": 13, "n_genes_found": 1},
            "interferon": {"category": "state", "cell_type": None, "n_genes_set": 15, "n_genes_found": 15},
            "unknown": {"category": "lineage", "cell_type": "not a label", "n_genes_set": 5, "n_genes_found": 5}}
    cl = {"a": {"top_markers": [], "signatures": {"T": 2.0, "weak": 2.0, "interferon": 5.0, "unknown": 9.0}},
          "b": {"top_markers": [], "signatures": {"T": -0.5, "weak": -0.5, "interferon": -1.0, "unknown": -1.0}},
          "c": {"top_markers": [], "signatures": {"T": -0.6, "weak": -0.6, "interferon": -1.0, "unknown": -1.0}}}
    votes, available, covered = cons.signature_votes(cl, sets, ref)
    assert covered == {"T cell", "NK cell"}
    assert available
    assert votes["a"]["T cell"] > 0.5
    assert votes["a"]["NK cell"] == pytest.approx(votes["a"]["T cell"] / 7, abs=1e-3)  # coverage 1/7
    assert "interferon" not in str(votes) and "not a label" not in str(votes)
    assert votes["b"] == {} and votes["c"] == {}          # negative scores never vote
    # a single cluster cannot be z-scored: no signature votes, markers decide
    one = {"a": cl["a"]}
    v1, _, _ = cons.signature_votes(one, sets, ref)
    assert v1["a"] == {}
    recs = cons.annotate({"a": {"top_markers": ["CD3D", "CD3E", "CD2", "TRAC", "IL7R"], "signatures": cl["a"]["signatures"]}}, sets, ref)
    assert recs[0]["label_broad"] == "T cell"


def test_marker_votes_rank_weighting(ref):
    genes = ["X%d" % i for i in range(12)] + ["CD3D", "CD3E", "CD2"]   # all three T markers beyond rank 10
    cl = {"k": {"top_markers": genes}}
    v = cons.marker_votes(cl, ref)["k"]
    assert v["T cell"] == pytest.approx(0.3)                          # 3 * 0.5 / 5
    cl2 = {"k": {"top_markers": ["CD3D", "CD3E", "CD2", "TRAC", "IL7R", "LTB"]}}
    assert cons.marker_votes(cl2, ref)["k"]["T cell"] == 1.0
    cl3 = {"k": {"top_markers": ["PPBP", "PF4"]}}                       # tiny gene universe: judged on what exists
    assert cons.marker_votes(cl3, ref)["k"]["platelet"] == 1.0


def test_annotate_bundle_layout(ref, sig_sets):
    bundle = {"clusters": synthetic_clusters(sig_sets), "signature_sets": sig_sets}
    recs = cons.annotate_bundle(bundle, ref)
    assert {r["cluster"]: r["label_broad"] for r in recs} == {"0": "T cell", "1": "monocyte", "2": "B cell",
                                                              "3": "NK cell", "4": "platelet"}
