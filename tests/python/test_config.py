"""Tests for cellwarden.config (docs/stages-spec.md section 4)."""

import json
from pathlib import Path

import pytest
import yaml

from cellwarden import config as C
from cellwarden import qc

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "pbmc_like"


def make_project(tmp_path, cfg_extra=None, samples=None, policy=True):
    """A config dir with config.yaml, samples.csv, qc_policy.yaml and two sample inputs."""
    proj = tmp_path / "proj"
    (proj / "data" / "S1").mkdir(parents=True)
    for f in ("matrix.mtx", "features.tsv", "barcodes.tsv"):
        (proj / "data" / "S1" / f).write_text("x")
    (proj / "data" / "S2.h5ad").write_text("x")
    (proj / "samples.csv").write_text(
        samples
        if samples is not None
        else "sample_id,path,format,donor,batch\nS1,data/S1,,D1,B1\nS2,data/S2.h5ad,h5ad,D2,B2\n"
    )
    cfg = {"project": "demo", "run_dir": "runs/demo", "samples": "samples.csv"}
    if policy:
        (proj / "qc_policy.yaml").write_text("hard_limits:\n  min_genes: 250\n")
        cfg["qc_policy"] = "qc_policy.yaml"
    cfg.update(cfg_extra or {})
    (proj / "config.yaml").write_text(yaml.safe_dump(cfg))
    return proj


def test_defaults_and_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    proj = make_project(tmp_path)
    cfg = C.load_config(proj / "config.yaml")
    assert cfg["seed"] == 42 and cfg["n_hvg"] == 2000 and cfg["device"] == "auto"
    assert cfg["resources"] == {"threads": 8, "max_memory_gb": 64, "gpu_mem_cap_gb": 20, "nice": 10}
    assert cfg["doublets"]["method"] == "scrublet" and cfg["qc"]["review_action"] == "keep"
    assert cfg["integration"] == {"method": "none", "batch_key": "sample_id", "engine": "auto",
                                  "harmony": {"theta": 2, "lambda": None, "sigma": 0.1, "nclust": None, "max_iter": 10}}
    assert cfg["pcs"] == "auto" and cfg["n_neighbors"] == 20 and cfg["clustering"]["resolution"] == 0.5
    assert cfg["object_format"] == "rds" and cfg["normalization"] == "lognorm"
    # Relative paths resolve against the config file's directory, not the cwd.
    assert cfg["run_dir"] == str(proj / "runs" / "demo")
    assert cfg["samples_source"] == str(proj / "samples.csv")
    assert cfg["qc_policy_source"] == str(proj / "qc_policy.yaml")
    s = {r["sample_id"]: r for r in cfg["samples_resolved"]}
    assert s["S1"]["path"] == str(proj / "data" / "S1") and s["S1"]["format_detected"] == "10x_mtx"
    assert s["S2"]["format"] == "h5ad"
    assert cfg["sample_columns"] == ["sample_id", "path", "format", "donor", "batch"]
    assert cfg["policy_hash"] == qc.policy_from_dict({"hard_limits": {"min_genes": 250}}).hash


def test_sample_paths_relative_to_config_dir(tmp_path):
    proj = make_project(tmp_path, {"samples": "sheets/samples.csv"})
    (proj / "sheets").mkdir()
    (proj / "sheets" / "samples.csv").write_text("sample_id,path\nS2,data/S2.h5ad\n")
    cfg = C.load_config(proj / "config.yaml")
    assert cfg["samples_resolved"][0]["path"] == str(proj / "data" / "S2.h5ad")
    assert cfg["config_dir"] == str(proj)
    out = C.prepare_run_dir(cfg)
    assert out["config_dir"] == str(proj)


def test_env_expansion_and_override(tmp_path, monkeypatch):
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setenv("SCRATCH_X", str(tmp_path / "scratch"))
    proj = make_project(tmp_path, {"run_dir": "${SCRATCH_X}/$USER/cellwarden/demo"})
    cfg = C.load_config(proj / "config.yaml")
    assert cfg["run_dir"] == str(tmp_path / "scratch" / "alice" / "cellwarden" / "demo")
    monkeypatch.chdir(tmp_path)
    cfg = C.load_config(proj / "config.yaml", run_dir="elsewhere/r1")
    assert cfg["run_dir"] == str(tmp_path / "elsewhere" / "r1")


def test_unset_env_var_is_problem(tmp_path, monkeypatch):
    monkeypatch.delenv("NOPE_NOT_SET", raising=False)
    proj = make_project(tmp_path, {"run_dir": "/x/$NOPE_NOT_SET/y"})
    with pytest.raises(C.ConfigError, match="NOPE_NOT_SET"):
        C.load_config(proj / "config.yaml")


def test_all_problems_reported(tmp_path):
    proj = make_project(
        tmp_path,
        {
            "seed": -1,
            "device": "tpu",
            "normalization": "scran",
            "n_hvg": "many",
            "pcs": 1,
            "n_neighbors": 1.5,
            "object_format": "h5",
            "resources": {"threads": 0, "max_memory_gb": -4, "gpu_mem_cap_gb": "x", "nice": 25},
            "doublets": {"method": "magic"},
            "qc": {"review_action": "maybe"},
            "integration": {"method": "scvi", "batch_key": ""},
            "clustering": {"resolution": 0},
        },
    )
    with pytest.raises(C.ConfigError) as e:
        C.load_config(proj / "config.yaml")
    text = "\n".join(e.value.problems)
    for key in ("seed", "device", "normalization", "n_hvg", "pcs", "n_neighbors", "object_format",
                "resources.threads", "resources.max_memory_gb", "resources.gpu_mem_cap_gb", "resources.nice",
                "doublets.method", "qc.review_action", "integration.method", "integration.batch_key",
                "clustering.resolution"):
        assert key in text, key
    assert len(e.value.problems) >= 16
    assert "invalid configuration" in str(e.value)


def test_bool_is_not_int(tmp_path):
    proj = make_project(tmp_path, {"seed": True, "resources": {"threads": True}})
    with pytest.raises(C.ConfigError) as e:
        C.load_config(proj / "config.yaml")
    assert any("seed" in p for p in e.value.problems)
    assert any("resources.threads" in p for p in e.value.problems)


def test_sct_gpu_conflict(tmp_path):
    proj = make_project(tmp_path, {"normalization": "sct", "device": "gpu"})
    with pytest.raises(C.ConfigError, match="sct runs on CPU only"):
        C.load_config(proj / "config.yaml")


def test_samples_validation(tmp_path):
    proj = make_project(
        tmp_path,
        samples=(
            "sample_id,path,format\n"
            "S1,data/S1,,\n"          # wrong field count
            "S2,data/S2.h5ad,h5ad\n"
            "S2,data/S2.h5ad,h5ad\n"  # duplicate id
            "S3,data/missing.h5,\n"  # missing path
            "S4,data/S2.h5ad,parquet\n"  # bad format
            "bad id,data/S2.h5ad,\n"  # bad characters
            ",data/S2.h5ad,\n"       # empty id
            "S5,,\n"                # empty path
        ),
    )
    with pytest.raises(C.ConfigError) as e:
        C.load_config(proj / "config.yaml")
    text = "\n".join(e.value.problems)
    assert "line 2 has 4 fields" in text
    assert "'S2' appears more than once" in text
    assert "does not exist" in text and "missing.h5" in text
    assert "'parquet'" in text
    assert "'bad id'" in text
    assert "sample_id is empty" in text
    assert "path is empty" in text


def test_samples_required_columns(tmp_path):
    proj = make_project(tmp_path, samples="id,file\nS1,data/S1\n")
    with pytest.raises(C.ConfigError) as e:
        C.load_config(proj / "config.yaml")
    text = "\n".join(e.value.problems)
    assert "'sample_id'" in text and "'path'" in text


def test_samples_undetectable_format(tmp_path):
    proj = make_project(tmp_path, samples="sample_id,path\nS1,data/thing.bin\n")
    (proj / "data" / "thing.bin").write_text("x")
    with pytest.raises(C.ConfigError, match="cannot detect the format"):
        C.load_config(proj / "config.yaml")


def test_missing_files_and_policy_errors(tmp_path):
    proj = make_project(tmp_path, {"samples": "nope.csv", "qc_policy": "nope.yaml"})
    with pytest.raises(C.ConfigError) as e:
        C.load_config(proj / "config.yaml")
    text = "\n".join(e.value.problems)
    assert "nope.csv" in text and "nope.yaml" in text
    proj2 = make_project(tmp_path / "b")
    (proj2 / "qc_policy.yaml").write_text("hard_limits:\n  min_genes: -5\nadaptive:\n  n_mads: 0\n")
    with pytest.raises(C.ConfigError) as e:
        C.load_config(proj2 / "config.yaml")
    assert sum(p.startswith("qc_policy:") for p in e.value.problems) == 2


def test_harmony_batch_key_must_be_column(tmp_path):
    proj = make_project(tmp_path, {"integration": {"method": "harmony", "batch_key": "lane"}})
    with pytest.raises(C.ConfigError, match="batch_key 'lane'"):
        C.load_config(proj / "config.yaml")
    proj = make_project(tmp_path / "ok", {"integration": {"method": "harmony", "batch_key": "batch"}})
    assert C.load_config(proj / "config.yaml")["integration"]["batch_key"] == "batch"


def test_unknown_keys_are_warnings(tmp_path):
    proj = make_project(tmp_path, {"llm": {"provider": "anthropic"}, "resources": {"threads": 2, "gpus": 1},
                                   "run_id": "mine"})
    cfg = C.load_config(proj / "config.yaml")
    w = "\n".join(cfg["_warnings"])
    assert "'llm'" in w and "resources.gpus" in w and "'run_id'" in w
    assert cfg["llm"] == {"default": {"provider": "anthropic"}}  # the flat shorthand is folded into default
    assert "run_id" not in cfg


def test_no_run_dir(tmp_path):
    proj = make_project(tmp_path, {"run_dir": None})
    with pytest.raises(C.ConfigError, match="run_dir is not set"):
        C.load_config(proj / "config.yaml")


def test_prepare_run_dir(tmp_path):
    proj = make_project(tmp_path)
    cfg = C.load_config(proj / "config.yaml")
    out = C.prepare_run_dir(cfg)
    rd = Path(cfg["run_dir"])
    cdir = rd / "config"
    for f in ("config.yaml", "samples.csv", "qc_policy.yaml", "config.resolved.yaml", "hashes.json"):
        assert (cdir / f).is_file(), f
    assert (rd / "logs").is_dir()
    assert (cdir / "config.yaml").read_bytes() == (proj / "config.yaml").read_bytes()
    assert (cdir / "samples.csv").read_bytes() == (proj / "samples.csv").read_bytes()
    res = yaml.safe_load((cdir / "config.resolved.yaml").read_text())
    assert res == out
    assert res["run_id"].startswith("demo-") and len(res["config_hash"]) == 64
    assert res["run_dir"] == str(rd) and Path(res["run_dir"]).is_absolute()
    assert res["samples"] == str(cdir / "samples.csv") and res["qc_policy"] == str(cdir / "qc_policy.yaml")
    assert Path(res["repo_root"]) == REPO
    assert not any(k.startswith("_") for k in res)
    hashes = json.loads((cdir / "hashes.json").read_text())
    assert hashes["samples.csv"] == C.sha256_file(cdir / "samples.csv")
    assert hashes["config_hash"] == res["config_hash"]
    assert C.load_resolved(rd) == res

    # Stable run_id and config_hash on rerun; config_hash changes with settings.
    again = C.prepare_run_dir(C.load_config(proj / "config.yaml"))
    assert again["run_id"] == res["run_id"] and again["config_hash"] == res["config_hash"]
    raw = yaml.safe_load((proj / "config.yaml").read_text())
    raw["clustering"] = {"resolution": 0.8}
    (proj / "config.yaml").write_text(yaml.safe_dump(raw))
    changed = C.prepare_run_dir(C.load_config(proj / "config.yaml"))
    assert changed["run_id"] == res["run_id"] and changed["config_hash"] != res["config_hash"]
    # A different run dir gets a different run_id.
    other = C.prepare_run_dir(C.load_config(proj / "config.yaml", run_dir=tmp_path / "other"))
    assert other["run_id"] != res["run_id"]


def test_builtin_default_policy_written(tmp_path):
    proj = make_project(tmp_path, policy=False)
    cfg = C.load_config(proj / "config.yaml")
    assert cfg["qc_policy_source"] == "builtin-default"
    out = C.prepare_run_dir(cfg)
    pol = qc.load_policy(out["qc_policy"])
    assert pol.hash == qc.policy_from_dict(None).hash == out["policy_hash"]


def test_load_resolved_missing(tmp_path):
    with pytest.raises(C.ConfigError, match="not found"):
        C.load_resolved(tmp_path)


def test_detect_format(tmp_path):
    import h5py

    h5 = tmp_path / "a.h5"
    with h5py.File(h5, "w") as f:
        f.create_group("matrix")
    other = tmp_path / "b.h5"
    with h5py.File(other, "w") as f:
        f.create_group("x")
    assert C.detect_format(h5) == "10x_h5"
    assert C.detect_format(other) is None
    assert C.detect_format(FIXTURE) == "10x_mtx"
    for name, fmt in (("x.h5ad", "h5ad"), ("x.rds", "seurat"), ("x.qs2", "seurat"), ("x.loom", "loom"),
                      ("x.csv", "table"), ("x.tsv.gz", "table"), ("x.txt", "table")):
        p = tmp_path / name
        p.write_text("")
        assert C.detect_format(p) == fmt, name


def test_engine_default_and_values(tmp_path):
    proj = make_project(tmp_path)
    assert C.load_config(proj / "config.yaml")["engine"] == "auto"
    for v in ("auto", "scfast", "seurat"):
        p = make_project(tmp_path / v, {"engine": v})
        assert C.load_config(p / "config.yaml")["engine"] == v


def test_engine_invalid_value(tmp_path):
    proj = make_project(tmp_path, {"engine": "rapids"})
    with pytest.raises(C.ConfigError, match="engine must be one of"):
        C.load_config(proj / "config.yaml")


def test_engine_scfast_gpu_conflict(tmp_path):
    proj = make_project(tmp_path, {"engine": "scfast", "device": "gpu"})
    with pytest.raises(C.ConfigError, match="engine: scfast has no GPU backend"):
        C.load_config(proj / "config.yaml")
    # auto and seurat are fine with device: gpu; scfast is fine with cpu/auto
    for extra in ({"engine": "auto", "device": "gpu"}, {"engine": "seurat", "device": "gpu"},
                  {"engine": "scfast", "device": "cpu"}, {"engine": "scfast", "device": "auto"}):
        p = make_project(tmp_path / (extra["engine"] + extra["device"]), extra)
        C.load_config(p / "config.yaml")


def test_engine_in_stage_fingerprints_and_example():
    from cellwarden import runner as R

    for st in ("S1a", "S1b", "S2"):
        assert "engine" in R.STAGE_CONFIG_KEYS[st]
    assert yaml.safe_load((REPO / "config" / "examples" / "config.yaml").read_text())["engine"] == "auto"


# engine_gpu (docs/scfast-integration.md "Pipeline wiring")


def test_engine_gpu_default_values_and_fingerprints(tmp_path):
    from cellwarden import runner as R

    proj = make_project(tmp_path)
    assert C.load_config(proj / "config.yaml")["engine_gpu"] == "auto"
    for v in ("auto", "scfast", "rapids"):
        p = make_project(tmp_path / v, {"engine_gpu": v, "device": "gpu"})
        assert C.load_config(p / "config.yaml")["engine_gpu"] == v
    proj = make_project(tmp_path / "bad", {"engine_gpu": "cuda"})
    with pytest.raises(C.ConfigError, match="engine_gpu must be one of"):
        C.load_config(proj / "config.yaml")
    for st in ("S1a", "S1b", "S2"):
        assert "engine_gpu" in R.STAGE_CONFIG_KEYS[st]
    assert yaml.safe_load((REPO / "config" / "examples" / "config.yaml").read_text())["engine_gpu"] == "auto"


def test_engine_gpu_conflicts(tmp_path):
    # engine_gpu: scfast lifts the scfast + device: gpu error (scfast then runs on CUDA)
    p = make_project(tmp_path / "ok", {"engine": "scfast", "device": "gpu", "engine_gpu": "scfast"})
    C.load_config(p / "config.yaml")
    p = make_project(tmp_path / "rapids", {"engine": "scfast", "device": "gpu", "engine_gpu": "rapids"})
    with pytest.raises(C.ConfigError, match="engine: scfast has no GPU backend"):
        C.load_config(p / "config.yaml")
    p = make_project(tmp_path / "seurat", {"engine": "seurat", "engine_gpu": "scfast"})
    with pytest.raises(C.ConfigError, match="engine_gpu: scfast needs engine: auto or scfast"):
        C.load_config(p / "config.yaml")


def test_integration_engine_and_harmony_params(tmp_path):
    proj = make_project(tmp_path, {"integration": {"method": "harmony", "batch_key": "batch", "engine": "scfast",
                                                   "harmony": {"theta": 1.5, "max_iter": 20}}})
    cfg = C.load_config(proj / "config.yaml")
    assert cfg["integration"]["engine"] == "scfast" and cfg["integration"]["harmony"]["theta"] == 1.5
    bad = make_project(tmp_path / "bad", {"integration": {"method": "harmony", "batch_key": "batch", "engine": "python",
                                                         "harmony": {"sigma": 0, "nclust": 0, "foo": 1}}})
    with pytest.raises(C.ConfigError) as e:
        C.load_config(bad / "config.yaml")
    msg = str(e.value)
    assert "integration.engine" in msg and "integration.harmony.sigma" in msg
    assert "integration.harmony.nclust" in msg and "unknown key(s) ['foo']" in msg


def test_cli_integrate_override(tmp_path):
    proj = make_project(tmp_path, {})
    cfg = C.load_config(proj / "config.yaml", overrides={"integration": {"method": "harmony", "batch_key": "batch"}})
    assert cfg["integration"]["method"] == "harmony" and cfg["integration"]["engine"] == "auto"
