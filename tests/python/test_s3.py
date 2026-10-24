"""End-to-end tests for stage S3 on the real fixture (docs/agents-spec.md sections 4-5).

Runs ``cellwarden run --until S3`` once (module scope) on tests/fixtures/stage_inputs
(S0-S2 take about a minute; S3 a few seconds) and checks the S3 files, the
bundle (no per-cell data), the CellTypist skip note, the consensus against the
planted cell types (purity >= 0.9 clusters), idempotence of the consensus step
and runner skipping. Skipped when Rscript/Seurat or scanpy are missing.

Set ``CELLWARDEN_S3_TEST_RUN=/path/to/run`` to reuse a finished run directory
(must contain S2/object.rds) instead of running S0-S2 again.

Run from the repo root: PYTHONPATH=python python3 -m pytest -q tests/python/test_s3.py
"""
from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cellwarden import runner as R
from cellwarden.annotation import consensus as cons
from cellwarden.stages import s3_celltypist, s3_consensus

REPO = Path(__file__).resolve().parents[2]
INPUTS = REPO / "tests" / "fixtures" / "stage_inputs"
CONF = REPO / "tests" / "integration" / "stage_configs"
RSCRIPT = os.environ.get("CELLWARDEN_RSCRIPT", "Rscript")
HAVE_CELLTYPIST = importlib.util.find_spec("celltypist") is not None

PLANTED = {"T": "T cell", "B": "B cell", "Mono": "monocyte", "NK": "NK cell", "Platelet": "platelet"}
S3_FILES = ["markers.tsv.gz", "cluster_summary.json", "signatures.tsv.gz", "cell_clusters.tsv.gz", "s3_r_summary.json",
            "celltypist_note.json", "evidence_bundle.json", "annotation_consensus.tsv"]


def _have_r() -> bool:
    if shutil.which(RSCRIPT) is None:
        return False
    r = subprocess.run([RSCRIPT, "-e", 'cat(requireNamespace("Seurat", quietly=TRUE))'], capture_output=True, text=True)
    return r.returncode == 0 and "TRUE" in r.stdout


def _have_py() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("anndata", "scanpy"))


pytestmark = pytest.mark.skipif(not (_have_r() and _have_py()), reason="needs Rscript with Seurat and python with scanpy")


def _write_conf(work: Path) -> Path:
    conf = work / "conf"
    conf.mkdir(parents=True, exist_ok=True)
    for f in ("cpu.yaml", "qc_policy.yaml"):
        shutil.copy(CONF / f, conf / f)
    rows = list(csv.DictReader(open(INPUTS / "samples.csv", newline="")))
    with open(conf / "samples.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            r["path"] = os.path.relpath(INPUTS / r["path"], conf)
            w.writerow(r)
    return conf / "cpu.yaml"


def _cellwarden(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO / "python") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-m", "cellwarden", *args], capture_output=True, text=True, env=env)


def _run_until_s3(cfg: Path, rd: Path) -> subprocess.CompletedProcess:
    # --no-gates: this is an S3 test; the gates (G1 before S2) are covered by their own tests and would
    # stop the run (exit 3, human review) when no LLM is configured.
    return _cellwarden("run", "--config", str(cfg), "--run-dir", str(rd), "--until", "S3", "--no-gates")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    reuse = os.environ.get("CELLWARDEN_S3_TEST_RUN")
    if reuse and (Path(reuse) / "S2" / "object.rds").exists():
        rd = Path(reuse)
        cfg = Path(yaml.safe_load((rd / "config" / "config.resolved.yaml").read_text())["config_file"])
    else:
        work = tmp_path_factory.mktemp("s3run")
        cfg = _write_conf(work)
        rd = work / "run"
    r = _run_until_s3(cfg, rd)
    if r.returncode != 0:
        logs = "\n".join(f"--- {p.name}\n" + p.read_text()[-3000:] for p in sorted((rd / "logs").glob("*.log")))
        pytest.fail(f"cellwarden run --until S3 failed:\n{r.stdout}\n{r.stderr}\n{logs}")
    return rd


@pytest.fixture(scope="module")
def s3(run_dir) -> Path:
    return run_dir / "S3"


@pytest.fixture(scope="module")
def bundle(s3) -> dict:
    return json.loads((s3 / "evidence_bundle.json").read_text())


@pytest.fixture(scope="module")
def consensus(s3) -> pd.DataFrame:
    return pd.read_csv(s3 / "annotation_consensus.tsv", sep="\t", dtype={"cluster": str, "cl_id": str}, keep_default_na=False)


@pytest.fixture(scope="module")
def pure_clusters(s3) -> dict[str, str]:
    """cluster -> planted type for clusters with purity >= 0.9 over non-injected (and platelet) cells."""
    truth = pd.read_csv(INPUTS / "truth.tsv", sep="\t", dtype=str)
    truth = truth[truth["injected"].isin(["none", "platelet"])]
    cc = pd.read_csv(s3 / "cell_clusters.tsv.gz", sep="\t", dtype=str).set_index("cell_id")["cluster"]
    truth = truth[truth["cell_id"].isin(cc.index)]
    tab = pd.crosstab(cc.loc[truth["cell_id"]].values, truth["cell_type"].values)
    out = {}
    for c in tab.index:
        row = tab.loc[c]
        if row.sum() >= 10 and row.max() / row.sum() >= 0.9:
            out[str(c)] = row.idxmax()
    return out


# ----------------------------------------------------------------- files

def test_all_s3_files_exist_and_runner_state_is_done(run_dir, s3):
    for f in S3_FILES:
        assert (s3 / f).exists(), f
    st = json.loads((run_dir / "state.json").read_text())
    assert st["stages"]["S3"]["status"] == "done"
    assert set(st["stages"]["S3"]["fingerprint_parts"]["config"]) == set(R.STAGE_CONFIG_KEYS["S3"])
    assert "S2/object.rds|S2/object.qs2" in st["stages"]["S3"]["fingerprint_parts"]["inputs"]
    assert not (s3 / "exchange" / "counts.h5ad").exists()
    log = (run_dir / "logs" / "S3.log").read_text()
    assert "S3 (R) done" in log and "consensus:" in log


def test_markers_table(s3, bundle):
    m = pd.read_csv(s3 / "markers.tsv.gz", sep="\t", dtype={"cluster": str})
    assert list(m.columns) == ["cluster", "gene", "avg_log2fc", "pct_in", "pct_out", "p_adj", "rank"]
    assert set(m["cluster"]) == set(bundle["clusters"])
    for c, g in m.groupby("cluster"):
        assert len(g) <= 50
        assert g["rank"].tolist() == list(range(1, len(g) + 1))
        assert (g["avg_log2fc"].diff().dropna() <= 1e-9).all()
        assert (g["avg_log2fc"] > 0).all()
        assert g["p_adj"].between(0, 1).all() and g["pct_in"].between(0, 100).all()


def test_cluster_summary_keys(s3, bundle):
    cs = json.loads((s3 / "cluster_summary.json").read_text())
    assert set(cs) == set(bundle["clusters"])
    n = 0
    for c, s in cs.items():
        assert set(s) == {"n_cells", "pct_cells", "per_sample", "median_genes", "median_umis", "median_pct_mito", "top_markers"}
        assert sum(s["per_sample"].values()) == s["n_cells"]
        assert set(s["per_sample"]) <= {"donor_A", "donor_B"}
        assert len(s["top_markers"]) <= 20
        assert s["top_markers"] == [t["gene"] for t in bundle["clusters"][c]["top_markers"][:20]]
        n += s["n_cells"]
    assert n == bundle["n_cells"]


def test_signatures_table(s3, bundle):
    sg = pd.read_csv(s3 / "signatures.tsv.gz", sep="\t", dtype={"cluster": str})
    assert list(sg.columns) == ["cluster", "signature", "category", "cell_type", "mean_score", "median_score",
                                "pct_positive", "n_genes_set", "n_genes_found"]
    sets = yaml.safe_load((REPO / "R" / "stages" / "data" / "signatures.yaml").read_text())["sets"]
    assert set(sg["signature"]) == set(sets)
    assert len(sg) == len(sets) * bundle["n_clusters"]
    assert set(bundle["signature_sets"]) == set(sets)
    assert sg.loc[sg["n_genes_found"] == 0, "mean_score"].isna().all()
    assert bundle["signature_sets"]["T"]["cell_type"] == "T cell"


def test_bundle_has_no_per_cell_data_and_the_metadata_hint(bundle, s3):
    n_cells = bundle["n_cells"]
    assert n_cells > 100 and bundle["n_clusters"] >= 4
    text = (s3 / "evidence_bundle.json").read_text()
    assert "donor_A_" not in text          # no barcodes anywhere
    assert "cell_id" not in text
    assert len(text) < 200_000
    for c, ev in bundle["clusters"].items():
        assert set(ev) >= {"n_cells", "pct_cells", "per_sample", "median_genes", "median_umis", "median_pct_mito",
                           "top_markers", "signatures", "signature_detail", "celltypist", "consensus"}
        assert ev["n_cells"] < n_cells
        assert ev["consensus"]["status"] in ("consensus", "low_confidence", "conflict")
    assert bundle["sample_metadata_hint"] == {"donor": ["D1", "D2"], "condition": ["control"], "batch": ["B1", "B2"]}
    assert bundle["samples"] == ["donor_A", "donor_B"]
    assert "FindAllMarkers" in bundle["markers"]["method"] or "presto" in bundle["markers"]["method"]
    assert bundle["annotation"]["min_confidence"] == 0.6
    assert bundle["annotation"]["celltypist_model"] == "Immune_All_Low.pkl"


def test_celltypist_note(s3, bundle):
    note = json.loads((s3 / "celltypist_note.json").read_text())
    assert note["model"] == "Immune_All_Low.pkl"
    assert note["models_dir"].endswith(os.path.join("refs", "celltypist"))
    assert bundle["celltypist"]["status"] == note["status"]
    if not HAVE_CELLTYPIST:
        assert note["status"] == "skipped"
        assert "not installed" in note["reason"]
        assert not (s3 / "celltypist.tsv.gz").exists()
        assert all(ev["celltypist"] is None for ev in bundle["clusters"].values())
    else:
        assert note["status"] in ("ok", "skipped")
        if note["status"] == "ok":
            ct = pd.read_csv(s3 / "celltypist.tsv.gz", sep="\t")
            assert list(ct.columns) == ["cell_id", "predicted_label", "majority_voting", "conf_score"]
            assert len(ct) == bundle["n_cells"]


def test_celltypist_off_is_a_clean_skip(tmp_path):
    rd = tmp_path / "r"
    (rd / "S3").mkdir(parents=True)
    cfg = {"annotation": {"celltypist": "off", "celltypist_model": "X.pkl"}, "config_dir": str(tmp_path)}
    note = s3_celltypist.run(rd, cfg)
    assert note["status"] == "skipped" and "off" in note["reason"]
    assert json.loads((rd / "S3" / "celltypist_note.json").read_text())["status"] == "skipped"
    assert s3_celltypist.models_dir_for(cfg) == (tmp_path / "refs" / "celltypist").resolve()
    assert s3_celltypist.models_dir_for(cfg, "/x/y") == Path("/x/y")
    assert s3_celltypist.models_dir_for({"annotation": {"models_dir": "/m"}}) == Path("/m")


# ----------------------------------------------------------------- consensus

def test_consensus_recovers_planted_types(consensus, pure_clusters, bundle):
    assert list(consensus.columns) == cons.CONSENSUS_COLUMNS
    assert set(consensus["cluster"]) == set(bundle["clusters"])
    assert len(pure_clusters) >= 4, pure_clusters
    assert {"T", "B", "Mono", "NK"} <= set(pure_clusters.values())
    by = consensus.set_index("cluster")
    for c, planted in pure_clusters.items():
        assert by.loc[c, "label_broad"] == PLANTED[planted], (c, planted, by.loc[c].to_dict())
        assert by.loc[c, "cl_id"].startswith("CL:")
        assert by.loc[c, "status"] == "consensus", (c, by.loc[c].to_dict())
        assert by.loc[c, "confidence"] >= 0.6
    assert consensus["confidence"].between(0, 1).all()
    for v in consensus["votes_json"]:
        d = json.loads(v)
        assert d["sources"] == ["signature", "markers"] or "celltypist" in d["sources"]
    for c, ev in bundle["clusters"].items():
        assert ev["consensus"]["label_broad"] == by.loc[c, "label_broad"]
    assert bundle["consensus_summary"]["n_consensus"] == int((consensus["status"] == "consensus").sum())


def test_consensus_step_alone_is_idempotent(run_dir, s3):
    before = (s3 / "annotation_consensus.tsv").read_bytes()
    bundle_before = json.loads((s3 / "evidence_bundle.json").read_text())
    cfg = yaml.safe_load((run_dir / "config" / "config.resolved.yaml").read_text())
    s3_consensus.run(run_dir, cfg)
    assert (s3 / "annotation_consensus.tsv").read_bytes() == before
    after = json.loads((s3 / "evidence_bundle.json").read_text())
    after.pop("created_utc"); bundle_before.pop("created_utc")
    assert after == bundle_before


def test_rerun_skips_s3_and_annotation_change_reruns_only_s3(run_dir):
    cfg_file = Path(yaml.safe_load((run_dir / "config" / "config.resolved.yaml").read_text())["config_file"])
    r = _run_until_s3(cfg_file, run_dir)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "S3   Markers and cell-type evidence — skipped" in r.stdout, r.stdout
    # a changed annotation setting reruns S3 only (the edited copy sits next to the original so the
    # relative sample paths stay valid)
    edited = cfg_file.with_name("cpu_edited_s3.yaml")
    edited.write_text(cfg_file.read_text() + "\nannotation:\n  min_confidence: 0.7\n")
    r = _run_until_s3(edited, run_dir)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "S2   Normalise, PCA, batch correction, clustering — skipped" in r.stdout and "S3   Markers and cell-type evidence — running" in r.stdout, r.stdout
    assert json.loads((run_dir / "S3" / "evidence_bundle.json").read_text())["annotation"]["min_confidence"] == 0.7
