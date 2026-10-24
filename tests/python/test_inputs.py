"""Command-line data inputs: cellwarden run a.h5 b.rds c.h5ad (python/cellwarden/inputs.py)."""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import pytest
import yaml

from cellwarden import config as C
from cellwarden import inputs as I
from cellwarden import runner as R

REPO = Path(__file__).resolve().parents[2]
STAGE_INPUTS = REPO / "tests" / "fixtures" / "stage_inputs"


def _touch(p: Path, text: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _h5(p: Path) -> Path:
    """A minimal 10x v3 h5 (only the group detection looks at)."""
    h5py = pytest.importorskip("h5py")
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "w") as f:
        f.create_group("matrix")
    return p


def _mtx_dir(p: Path) -> Path:
    _touch(p / "matrix.mtx.gz")
    return p


def test_detect_format_by_extension_and_folder(tmp_path):
    assert I.detect_format(_mtx_dir(tmp_path / "m")) == "10x_mtx"
    assert I.detect_format(_touch(tmp_path / "a.rds")) == "seurat"
    assert I.detect_format(_touch(tmp_path / "a.qs")) == "seurat"
    assert I.detect_format(_touch(tmp_path / "a.loom")) == "loom"
    assert I.detect_format(_touch(tmp_path / "a.tsv.gz")) == "table"
    assert I.detect_format(_touch(tmp_path / "matrix.mtx")) == "10x_mtx"
    assert I.detect_format(_touch(tmp_path / "notes.pdf")) is None
    assert C.detect_format(tmp_path / "a.rds") == "seurat"  # config keeps the old entry point


def test_detect_format_h5_by_content(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "v3.h5", "w") as f:
        f.create_group("matrix")
    with h5py.File(tmp_path / "v2.h5", "w") as f:
        g = f.create_group("GRCh38")
        for k in ("barcodes", "data", "indptr"):
            g.create_dataset(k, data=[0])
    with h5py.File(tmp_path / "anndata.h5", "w") as f:
        f.create_group("obs")
        f.create_group("var")
    assert I.detect_format(tmp_path / "v3.h5") == "10x_h5"
    assert I.detect_format(tmp_path / "v2.h5") == "10x_h5"
    assert I.detect_format(tmp_path / "anndata.h5") == "h5ad"


def test_sample_ids_from_names_and_folders(tmp_path):
    a = _h5(tmp_path / "donorA" / "outs" / "filtered_feature_bc_matrix.h5")
    b = _touch(tmp_path / "donor B.rds")
    c = _mtx_dir(tmp_path / "donorC" / "filtered_feature_bc_matrix")
    d = _touch(tmp_path / "other" / "donorA.rds")
    rows, sheet = I.collect_inputs([a, b, c, d])
    assert sheet is None
    assert [r["sample_id"] for r in rows] == ["donorA", "donor_B", "donorC", "donorA_2"]
    assert [r["format"] for r in rows] == ["10x_h5", "seurat", "10x_mtx", "seurat"]
    assert all(Path(r["path"]).is_absolute() for r in rows)


def test_folder_expansion(tmp_path):
    run = tmp_path / "cellranger"
    _h5(run / "outs" / "filtered_feature_bc_matrix.h5")
    _h5(run / "outs" / "raw_feature_bc_matrix.h5")
    rows, _ = I.collect_inputs([run])
    assert len(rows) == 1 and rows[0]["path"].endswith("outs/filtered_feature_bc_matrix.h5")
    batch = tmp_path / "batch"
    _touch(batch / "s1.h5ad")
    _touch(batch / "s2.rds")
    _mtx_dir(batch / "s3")
    _touch(batch / "README.txt.bak")
    rows, _ = I.collect_inputs([batch])
    assert [r["sample_id"] for r in rows] == ["s1", "s2", "s3"]
    with pytest.raises(C.ConfigError, match="no single-cell data"):
        I.collect_inputs([_touch(tmp_path / "empty" / "x.bak").parent])
    with pytest.raises(C.ConfigError, match="does not exist"):
        I.collect_inputs([tmp_path / "missing.h5"])


def test_samples_sheet_is_used_as_is_and_must_be_alone(tmp_path):
    sheet = _touch(tmp_path / "s.csv", "sample_id,path,condition\nA,a.rds,ctrl\n")
    _touch(tmp_path / "a.rds")
    rows, got = I.collect_inputs([sheet])
    assert rows == [] and got == sheet
    with pytest.raises(C.ConfigError, match="only input"):
        I.collect_inputs([sheet, tmp_path / "a.rds"])
    counts = _touch(tmp_path / "counts.csv", "gene,c1,c2\nCD3E,1,0\n")
    assert not I.is_samples_sheet(counts)


def test_meta_parsing():
    assert I.parse_meta(["condition=control", "batch = B1"]) == {"condition": "control", "batch": "B1"}
    for bad in ("nokey", "1x=a", "sample_id=a"):
        with pytest.raises(C.ConfigError):
            I.parse_meta([bad])


def test_default_out_dir_and_overwrite_guard(tmp_path):
    rows = [{"sample_id": "donorA"}]
    assert I.default_out_dir(rows, today=dt.date(2026, 10, 23), cwd=tmp_path) == tmp_path / "cellwarden_donorA_20261023"
    assert I.default_out_dir(rows, name="my study", today=dt.date(2026, 10, 23), cwd=tmp_path).name == "cellwarden_my_study_20261023"
    busy = _touch(tmp_path / "busy" / "thesis.docx").parent
    with pytest.raises(C.ConfigError, match="not empty"):
        I.check_out_dir(busy)
    I.check_out_dir(tmp_path / "new")


def test_synthesized_config_loads_and_is_deterministic(tmp_path):
    rows, _ = I.collect_inputs([STAGE_INPUTS / "donor_A", STAGE_INPUTS / "donor_B.h5ad"])
    out = tmp_path / "run"
    p1 = I.synthesize_config(out, rows=rows, meta={"condition": "control"})
    cfg = C.load_config(p1)
    assert [s["sample_id"] for s in cfg["samples_resolved"]] == ["donor_A", "donor_B"]
    assert "condition" in cfg["sample_columns"] and Path(cfg["run_dir"]) == out
    first = (out / "config" / I.INPUT_SAMPLES).read_text()
    I.synthesize_config(out, rows=rows, meta={"condition": "control"})
    assert (out / "config" / I.INPUT_SAMPLES).read_text() == first  # same inputs → same sheet → resume works


def test_synthesized_config_keeps_user_config_with_absolute_paths(tmp_path):
    proj = tmp_path / "proj"
    policy = _touch(proj / "policy.yaml", yaml.safe_dump({"hard_limits": {"min_genes": 150}}))
    user = _touch(proj / "config.yaml", yaml.safe_dump({"project": "p", "qc_policy": "policy.yaml", "n_hvg": 3000,
                                                        "run_dir": "results"}))
    rows, _ = I.collect_inputs([STAGE_INPUTS / "donor_A"])
    p = I.synthesize_config(tmp_path / "out", rows=rows, config_path=user)
    raw = yaml.safe_load(p.read_text())
    assert raw["qc_policy"] == str(policy.resolve()) and raw["n_hvg"] == 3000
    assert raw["run_dir"] == str(tmp_path / "out")


def test_sheet_paths_are_relative_to_the_sheet(tmp_path):
    sheet = tmp_path / "meta" / "samples.csv"
    sheet.parent.mkdir()
    with open(sheet, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "path", "donor"])
        w.writerow(["A", str(Path("..") / "data" / "donor_A"), "D1"])
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "donor_A").symlink_to(STAGE_INPUTS / "donor_A")
    p = I.synthesize_config(tmp_path / "out", sheet=sheet)
    cfg = C.load_config(p)
    assert cfg["samples_resolved"][0]["format_detected"] == "10x_mtx"


def test_cli_parses_inputs_and_requires_something(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = R._parser().parse_args(["run", str(STAGE_INPUTS / "donor_A"), "--out", str(tmp_path / "o"), "--meta", "batch=B1"])
    path, rd = R._run_config_path(a)
    assert rd is None and Path(path).name == I.INPUT_CONFIG
    assert "batch" in (tmp_path / "o" / "config" / I.INPUT_SAMPLES).read_text().splitlines()[0]
    with pytest.raises(C.ConfigError, match="give data files"):
        R._run_config_path(R._parser().parse_args(["run"]))
    a = R._parser().parse_args(["run", str(STAGE_INPUTS / "donor_B.h5ad")])
    path, _ = R._run_config_path(a)
    assert Path(path).parents[1].name.startswith("cellwarden_donor_B_")
