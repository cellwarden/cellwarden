"""S1b when the engine is scfast: pass-through of the columns S1a already wrote.

docs/scfast-integration.md, "S1b doublets and pre-clustering". The pass-through tests
need no R. The end-to-end test runs S0 and S1a (``R/stages/run_stage.R``, engine scfast)
and S1b/S1c on the stage fixture, and is skipped when ``scfastR`` or Rscript is missing
(point ``R_LIBS`` at the scfastR library).
"""

import gzip
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cellwarden import config as C
from cellwarden.stages import read_tsv_str, s1b_doublets, s1c_qc

from test_s1b_s1c import Result, inproc, make_run, write_gz  # noqa: F401  (inproc is a fixture)

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "stage_inputs"
CONF = REPO / "tests" / "integration" / "stage_configs"
S1B = ("doublet_score", "doublet_call", "pre_cluster")


def _set_engine(rd: Path, engine: str) -> None:
    f = rd / "config" / "config.resolved.yaml"
    cfg = yaml.safe_load(f.read_text())
    cfg["engine"] = engine
    f.write_text(yaml.safe_dump(cfg))


def _s1a_like(rd: Path, a) -> pd.DataFrame:
    """Turn the run's partial qc_metrics into what S1a writes on the scfast path; drop the h5ad."""
    m = read_tsv_str(rd / "S1" / "qc_metrics.tsv.gz")
    m["doublet_score"] = [f"{0.01 * (i % 30):.6g}" for i in range(len(m))]
    m["doublet_call"] = ["TRUE" if i % 17 == 0 else "FALSE" for i in range(len(m))]
    m["pre_cluster"] = [str(i % 4) for i in range(len(m))]
    write_gz(m, rd / "S1" / "qc_metrics.tsv.gz")
    (rd / "S1" / "metrics_counts.h5ad").unlink()
    return m


@pytest.mark.parametrize("engine", ["scfast", "auto"])
def test_passthrough_leaves_table_untouched(tmp_path, inproc, engine):
    rd, a = make_run(tmp_path, extra={"engine": engine})
    _s1a_like(rd, a)
    before = (rd / "S1" / "qc_metrics.tsv.gz").read_bytes()
    r = inproc("cellwarden.stages.s1b_doublets", rd)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (rd / "S1" / "qc_metrics.tsv.gz").read_bytes() == before
    assert not list(rd.rglob("*.h5ad"))
    assert "pass-through" in (r.stdout + r.stderr)
    # S1c is unchanged and reads the table as is
    r = inproc("cellwarden.stages.s1c_qc", rd)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (rd / "S1" / "qc_decisions.tsv.gz").exists()


def test_passthrough_records_method(tmp_path, inproc):
    rd, a = make_run(tmp_path, extra={"engine": "scfast", "doublets": {"method": "none"}})
    _s1a_like(rd, a)
    r = inproc("cellwarden.stages.s1b_doublets", rd)
    assert r.returncode == 0
    assert "doublets.method=none" in (r.stdout + r.stderr)


def test_engine_seurat_does_not_pass_through(tmp_path, inproc):
    """With engine seurat the existing behaviour holds: the h5ad is required and the columns recomputed."""
    rd, a = make_run(tmp_path, extra={"engine": "seurat", "doublets": {"method": "none"}})
    # write the S1b columns with a marker value but keep the h5ad
    t = read_tsv_str(rd / "S1" / "qc_metrics.tsv.gz")
    t["doublet_score"], t["doublet_call"], t["pre_cluster"] = "0.5", "TRUE", "marker"
    write_gz(t, rd / "S1" / "qc_metrics.tsv.gz")
    r = inproc("cellwarden.stages.s1b_doublets", rd)
    assert r.returncode == 0, r.stdout + r.stderr
    out = read_tsv_str(rd / "S1" / "qc_metrics.tsv.gz")
    assert (out["doublet_call"] == "FALSE").all() and (out["doublet_score"] == "NA").all()
    assert "marker" not in set(out["pre_cluster"])


def test_missing_columns_and_no_h5ad_is_an_error(tmp_path, inproc):
    """engine auto/scfast but S1a did not write the columns (it resolved to seurat): the normal path needs the h5ad."""
    rd, a = make_run(tmp_path, extra={"engine": "auto"})
    (rd / "S1" / "metrics_counts.h5ad").unlink()
    r = inproc("cellwarden.stages.s1b_doublets", rd)
    assert r.returncode == 1
    assert "metrics_counts.h5ad" in r.stderr and "not found" in r.stderr


def test_partial_columns_are_not_a_passthrough(tmp_path, inproc):
    rd, a = make_run(tmp_path, extra={"engine": "scfast", "doublets": {"method": "none"}})
    t = read_tsv_str(rd / "S1" / "qc_metrics.tsv.gz")
    t["pre_cluster"] = "x"
    write_gz(t, rd / "S1" / "qc_metrics.tsv.gz")
    r = inproc("cellwarden.stages.s1b_doublets", rd)
    assert r.returncode == 0, r.stdout + r.stderr
    out = read_tsv_str(rd / "S1" / "qc_metrics.tsv.gz")
    assert set(S1B) <= set(out.columns) and "x" not in set(out["pre_cluster"])


# ------------------------------------------------------------------ end to end with R

def _rscript():
    return os.environ.get("RSCRIPT") or shutil.which("Rscript")


def _has_scfastr(rscript) -> bool:
    if not rscript:
        return False
    r = subprocess.run([rscript, "-e", "quit(status = !requireNamespace('scfastR', quietly = TRUE))"],
                       capture_output=True, text=True)
    return r.returncode == 0


@pytest.fixture(scope="module")
def r_run(tmp_path_factory):
    rscript = _rscript()
    if not _has_scfastr(rscript):
        pytest.skip("scfastR (R) is not installed")
    work = tmp_path_factory.mktemp("scfast_e2e")
    conf = work / "conf"
    conf.mkdir()
    shutil.copy(CONF / "qc_policy.yaml", conf)
    cfg = yaml.safe_load((CONF / "cpu.yaml").read_text())
    cfg.update({"engine": "scfast", "run_dir": str(work / "run"), "samples": "samples.csv"})
    (conf / "config.yaml").write_text(yaml.safe_dump(cfg))
    sheet = pd.read_csv(FIXTURES / "samples.csv", dtype=str, keep_default_na=False)
    sheet["path"] = [str(FIXTURES / p) for p in sheet["path"]]
    sheet.to_csv(conf / "samples.csv", index=False)
    res = C.prepare_run_dir(C.load_config(conf / "config.yaml"))
    rd = Path(res["run_dir"])
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"), CELLWARDEN_THREADS="2")
    for st in ("S0", "S1a"):
        p = subprocess.run([rscript, str(REPO / "R" / "stages" / "run_stage.R"), "--stage", st, "--run-dir", str(rd)],
                           capture_output=True, text=True, env=env, timeout=900)
        assert p.returncode == 0, f"{st}\n{p.stdout}\n{p.stderr}"
    return rd, env


def test_e2e_s1a_writes_columns_and_no_h5ad(r_run):
    rd, _ = r_run
    assert not list(rd.rglob("*.h5ad"))
    m = read_tsv_str(rd / "S1" / "qc_metrics.tsv.gz")
    assert list(m.columns[-3:]) == list(S1B)
    assert m["doublet_call"].isin(["TRUE", "FALSE"]).all()
    assert (m["pre_cluster"] != "").all()


def test_e2e_s1b_passthrough_s1c_and_doublet_recall(r_run):
    rd, env = r_run
    before = (rd / "S1" / "qc_metrics.tsv.gz").read_bytes()
    for mod in ("cellwarden.stages.s1b_doublets", "cellwarden.stages.s1c_qc"):
        p = subprocess.run([sys.executable, "-m", mod, "--run-dir", str(rd)], capture_output=True, text=True,
                           env=env, timeout=900)
        assert p.returncode == 0, p.stdout + p.stderr
        if mod.endswith("s1b_doublets"):
            assert (rd / "S1" / "qc_metrics.tsv.gz").read_bytes() == before
            assert "pass-through" in p.stdout + p.stderr
    truth = pd.read_csv(FIXTURES / "truth.tsv", sep="\t", dtype=str).set_index("cell_id")
    m = read_tsv_str(rd / "S1" / "qc_metrics.tsv.gz").set_index("cell_id")
    inj = truth.loc[m.index, "injected"]
    dbl = inj == "doublet"
    assert dbl.sum() == 20
    called = m["doublet_call"] == "TRUE"
    assert called[dbl].mean() >= 0.5, f"recall {called[dbl].mean():.2f}"
    assert called[inj == "none"].mean() <= 0.10
    with gzip.open(rd / "S1" / "qc_decisions.tsv.gz", "rt") as f:
        dec = pd.read_csv(f, sep="\t", dtype=str).set_index("cell_id")
    assert len(dec) == len(m)
    assert dec.loc[dbl[dbl].index, "reasons"].str.contains("DOUBLET").mean() >= 0.5
