"""S1b (doublets + pre-clustering) and S1c (QC decisions) on a synthetic run directory.

The synthetic data mimic what S1a writes: ``S1/metrics_counts.h5ad`` (exchange
full mode) and a partial ``S1/qc_metrics.tsv.gz``. They contain three cell
types, injected heterotypic doublets (sum of two cells' counts), low-quality
cells (few genes, high mitochondrial fraction) and a sample too small for
Scrublet.
"""

import gzip
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml

from cellwarden import config as C
from cellwarden import qc
from cellwarden import runner as R
from cellwarden.exchange import read_exchange, write_exchange
from cellwarden.stages import read_tsv_str, s1b_doublets, s1c_qc

REPO = Path(__file__).resolve().parents[2]
R_COLUMNS = ["cell_id", "sample_id", "n_genes", "n_umis", "pct_mito", "pct_ribo", "pct_hb", "log10_genes_per_umi",
             "protected_Platelet", "protected_Erythrocyte"]
SAMPLES = {"SA": 320, "SB": 280, "SC": 60}  # SC: fewer than 100 cells, Scrublet skipped
N_DOUBLETS = 16
N_LOWQ = 8
N_HIGHMT = 8


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    n_genes = 600
    names = [f"GENE{i}" for i in range(n_genes)]
    mt = [f"MT-CO{i}" for i in range(1, 14)]
    special = ["PPBP", "PF4", "GNG11", "HBB", "HBA1", "HBA2", "RPL3", "RPS6"]
    names[: len(mt)] = mt
    names[len(mt): len(mt) + len(special)] = special
    names[50] = "GENE_with_underscore"
    base = rng.gamma(2.0, 1.2, n_genes)
    base[: len(mt)] = 1.5
    base[len(mt): len(mt) + 6] = 0.02
    mu = np.tile(base, (3, 1))
    for k in range(3):
        mk = rng.choice(np.arange(100, n_genes), 40, replace=False)
        mu[k, mk] *= 10.0

    blocks, obs = [], []
    for s, n in SAMPLES.items():
        truth = rng.integers(0, 3, n)
        lam = mu[truth] * rng.gamma(12.0, 1 / 12.0, size=(n, 1)) * rng.gamma(20, 0.05, size=(n, n_genes))
        X = rng.poisson(lam).astype(np.int64)
        kind = np.array(["singlet"] * n, dtype=object)
        if n >= 100:
            for d in range(N_DOUBLETS):
                i = d
                while True:
                    j = rng.integers(N_DOUBLETS + N_LOWQ + N_HIGHMT, n)
                    if truth[j] != truth[i]:
                        break
                X[i] = X[i] + X[j]
                kind[i] = "doublet"
            lo = slice(N_DOUBLETS, N_DOUBLETS + N_LOWQ)
            X[lo] = rng.binomial(X[lo], 0.08)
            kind[lo] = "lowq"
            hm = slice(N_DOUBLETS + N_LOWQ, N_DOUBLETS + N_LOWQ + N_HIGHMT)
            X[hm, : len(mt)] += rng.poisson(120, size=(N_HIGHMT, len(mt)))
            kind[hm] = "highmt"
        blocks.append(X)
        obs.append(pd.DataFrame({"sample_id": s, "truth": [f"T{t}" for t in truth], "kind": kind},
                                index=[f"{s}_BC{i:05d}-1" for i in range(n)]))
    counts = sp.csr_matrix(np.vstack(blocks).astype(np.int32))
    obs = pd.concat(obs)
    a = ad.AnnData(X=counts.astype(np.float32), obs=obs, var=pd.DataFrame(index=names))
    a.layers["counts"] = counts
    tot = np.asarray(counts.sum(1)).ravel()
    a.X = sp.csr_matrix(np.log1p(counts.multiply(1e4 / tot[:, None]))).astype(np.float32)
    return a


def partial_metrics(a):
    """What S1a writes (spec 3.2), computed in Python."""
    c = sp.csr_matrix(a.layers["counts"])
    tot = np.asarray(c.sum(1)).ravel().astype(float)
    ng = np.asarray((c > 0).sum(1)).ravel()
    names = pd.Index(a.var_names)

    def frac(mask):
        return np.asarray(c[:, np.flatnonzero(mask)].sum(1)).ravel() / tot

    df = pd.DataFrame(
        {
            "cell_id": a.obs_names,
            "sample_id": a.obs["sample_id"].to_numpy(),
            "n_genes": ng,
            "n_umis": tot.astype(int),
            "pct_mito": 100 * frac(names.str.startswith("MT-")),
            "pct_ribo": 100 * frac(names.str.match(r"^RP[SL]")),
            "pct_hb": 100 * frac(names.str.match(r"^HB[ABDEGMQZ]\d*$")),
            "log10_genes_per_umi": np.log10(ng) / np.log10(tot),
            "protected_Platelet": frac(names.isin(["PPBP", "PF4", "GNG11"])),
            "protected_Erythrocyte": frac(names.isin(["HBB", "HBA1", "HBA2"])),
        }
    )
    return df


def write_gz(df, path):
    buf = io.StringIO()
    df.to_csv(buf, sep="\t", index=False, float_format="%.6g")
    with gzip.open(path, "wt") as f:
        f.write(buf.getvalue())


def make_run(tmp_path, extra=None, data=None):
    proj = tmp_path / "proj"
    proj.mkdir(parents=True)
    for s in SAMPLES:
        (proj / f"{s}.h5ad").write_text("placeholder")
    (proj / "samples.csv").write_text("sample_id,path\n" + "".join(f"{s},{s}.h5ad\n" for s in SAMPLES))
    cfg = {"project": "synthetic", "run_dir": "run", "samples": "samples.csv", "seed": 11,
           "resources": {"threads": 2, "max_memory_gb": 8, "gpu_mem_cap_gb": 4, "nice": 0}}
    cfg.update(extra or {})
    (proj / "config.yaml").write_text(yaml.safe_dump(cfg))
    res = C.prepare_run_dir(C.load_config(proj / "config.yaml"))
    rd = Path(res["run_dir"])
    a = data if data is not None else synthetic()
    (rd / "S1").mkdir()
    write_exchange(a, rd / "S1" / "metrics_counts.h5ad", "full", run_id=res["run_id"], stage="S1a",
                   config_hash=res["config_hash"])
    write_gz(partial_metrics(a), rd / "S1" / "qc_metrics.tsv.gz")
    return rd, a


NUMBA_CACHE = {}


@pytest.fixture(scope="session", autouse=True)
def _shared_numba_cache(tmp_path_factory):
    # One numba cache for every stage process in this module, so JIT compilation happens once.
    NUMBA_CACHE["dir"] = str(tmp_path_factory.mktemp("numba"))
    old = os.environ.get("NUMBA_CACHE_DIR")
    os.environ["NUMBA_CACHE_DIR"] = NUMBA_CACHE["dir"]
    yield
    if old is None:
        os.environ.pop("NUMBA_CACHE_DIR", None)
    else:
        os.environ["NUMBA_CACHE_DIR"] = old


def stage(module, rd, extra_env=None):
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"), OMP_NUM_THREADS="2")
    env.pop("CELLWARDEN_ALLOW_CPU_AS_GPU", None)
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-m", module, "--run-dir", str(rd)], capture_output=True, text=True,
                          env=env, timeout=900)


class Result:
    def __init__(self, returncode, stdout, stderr):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


@pytest.fixture
def inproc(capfd, monkeypatch):
    """Run a stage's main() in this process (numba JIT stays warm, unlike a subprocess)."""
    monkeypatch.delenv("CELLWARDEN_ALLOW_CPU_AS_GPU", raising=False)
    mods = {"cellwarden.stages.s1b_doublets": s1b_doublets, "cellwarden.stages.s1c_qc": s1c_qc}

    def call(module, rd, extra_env=None):
        for k, v in (extra_env or {}).items():
            monkeypatch.setenv(k, v)
        capfd.readouterr()
        code = mods[module].main(["--run-dir", str(rd)])
        out, err = capfd.readouterr()
        for k in extra_env or {}:
            monkeypatch.delenv(k)
        return Result(code, out, err)

    return call


@pytest.fixture(scope="module")
def done(tmp_path_factory):
    """S1b then S1c on the default synthetic data (run once for the module)."""
    rd, a = make_run(tmp_path_factory.mktemp("s1"))
    before = read_tsv_str(rd / "S1" / "qc_metrics.tsv.gz")
    r1 = stage("cellwarden.stages.s1b_doublets", rd)
    assert r1.returncode == 0, r1.stdout + r1.stderr
    after_b = read_tsv_str(rd / "S1" / "qc_metrics.tsv.gz")
    b_bytes = (rd / "S1" / "qc_metrics.tsv.gz").read_bytes()
    assert (rd / "S1" / "metrics_counts.h5ad").exists()  # S1b never deletes it
    r2 = stage("cellwarden.stages.s1c_qc", rd)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    return {"rd": rd, "a": a, "before": before, "after_b": after_b, "b_bytes": b_bytes, "log_b": r1.stdout + r1.stderr,
            "log_c": r2.stdout + r2.stderr}


# --------------------------------------------------------------------------
# S1b
# --------------------------------------------------------------------------


def test_s1b_preserves_rows_and_r_columns(done):
    before, after = done["before"], done["after_b"]
    assert list(after.columns) == list(before.columns) + ["doublet_score", "doublet_call", "pre_cluster"]
    pd.testing.assert_frame_equal(after[list(before.columns)], before)  # byte-identical strings, same order
    assert len(after) == sum(SAMPLES.values())


def test_s1b_doublet_columns(done):
    m = done["after_b"].set_index("cell_id")
    kind = done["a"].obs["kind"]
    assert set(m["doublet_call"]) <= {"TRUE", "FALSE"}
    sc_ = m.loc[m.sample_id == "SC"]
    assert (sc_["doublet_score"] == "NA").all() and (sc_["doublet_call"] == "FALSE").all()
    assert "SC: 60 cells < 100" in done["log_b"]
    big = m.loc[m.sample_id != "SC"]
    score = big["doublet_score"].astype(float)
    assert score.notna().all() and ((score >= 0) & (score <= 1)).all()
    k = kind.reindex(big.index)
    call = big["doublet_call"] == "TRUE"
    # Injected heterotypic doublets score higher and are mostly called.
    assert score[k == "doublet"].median() > score[k == "singlet"].quantile(0.9)
    assert call[k == "doublet"].mean() >= 0.5
    assert call[k == "singlet"].mean() <= 0.1


def test_s1b_pre_clusters_recover_types(done):
    m = done["after_b"].set_index("cell_id")
    obs = done["a"].obs
    good = obs.index[obs["kind"] == "singlet"]
    pc = m.loc[good, "pre_cluster"]
    assert pc.str.fullmatch(r"\d+").all()
    tab = pd.crosstab(pc, obs.loc[good, "truth"])
    purity = tab.max(axis=1).sum() / tab.values.sum()
    assert purity >= 0.9, tab
    assert 2 <= m["pre_cluster"].nunique() <= 12
    assert not (done["rd"] / "S1" / ".s1b_tmp").exists()


def test_s1b_deterministic_rerun(inproc, tmp_path):
    rd, _ = make_run(tmp_path, {"doublets": {"method": "none"}})
    assert inproc("cellwarden.stages.s1b_doublets", rd).returncode == 0
    first = (rd / "S1" / "qc_metrics.tsv.gz").read_bytes()
    r = inproc("cellwarden.stages.s1b_doublets", rd)  # rerun on its own output: columns replaced, same bytes
    assert r.returncode == 0, r.stderr
    assert "recomputing" in r.stdout
    assert (rd / "S1" / "qc_metrics.tsv.gz").read_bytes() == first
    m = read_tsv_str(rd / "S1" / "qc_metrics.tsv.gz")
    assert (m["doublet_score"] == "NA").all() and (m["doublet_call"] == "FALSE").all()
    assert m["pre_cluster"].nunique() >= 2


def test_s1b_scdblfinder_not_implemented(inproc, tmp_path):
    rd, _ = make_run(tmp_path, {"doublets": {"method": "scdblfinder"}})
    r = inproc("cellwarden.stages.s1b_doublets", rd)
    assert r.returncode == 1
    assert "scDblFinder" in r.stderr or "scdblfinder" in r.stderr
    assert "runs in R" in r.stderr


def test_s1b_scdblfinder_columns_from_r_are_kept(inproc, tmp_path):
    rd, a = make_run(tmp_path, {"doublets": {"method": "scdblfinder"}})
    m = partial_metrics(a)
    m["doublet_score"] = 0.25
    m["doublet_call"] = np.where(np.arange(len(m)) < 5, "TRUE", "FALSE")
    write_gz(m, rd / "S1" / "qc_metrics.tsv.gz")
    before = read_tsv_str(rd / "S1" / "qc_metrics.tsv.gz")
    r = inproc("cellwarden.stages.s1b_doublets", rd)
    assert r.returncode == 0, r.stderr
    after = read_tsv_str(rd / "S1" / "qc_metrics.tsv.gz")
    pd.testing.assert_frame_equal(after[list(before.columns)], before)
    assert "pre_cluster" in after.columns


def test_s1b_row_order_and_cell_mismatch(inproc, tmp_path):
    rd, a = make_run(tmp_path, {"doublets": {"method": "none"}})
    m = partial_metrics(a).sample(frac=1, random_state=1)  # table order differs from the h5ad
    write_gz(m, rd / "S1" / "qc_metrics.tsv.gz")
    assert inproc("cellwarden.stages.s1b_doublets", rd).returncode == 0
    out = read_tsv_str(rd / "S1" / "qc_metrics.tsv.gz")
    assert list(out["cell_id"]) == list(m["cell_id"])
    write_gz(m.iloc[:-3], rd / "S1" / "qc_metrics.tsv.gz")
    r = inproc("cellwarden.stages.s1b_doublets", rd)
    assert r.returncode == 1 and "differ" in r.stderr


def test_s1b_missing_input(inproc, tmp_path):
    rd, _ = make_run(tmp_path)
    (rd / "S1" / "metrics_counts.h5ad").unlink()
    r = inproc("cellwarden.stages.s1b_doublets", rd)
    assert r.returncode == 1 and "S1a must run first" in r.stderr


def test_s1b_device_gpu_without_gpu(inproc, tmp_path):
    from cellwarden import gpu_core

    if gpu_core.gpu_available()[0]:
        pytest.skip("a real GPU is present")
    rd, _ = make_run(tmp_path, {"device": "gpu", "doublets": {"method": "none"}})
    r = inproc("cellwarden.stages.s1b_doublets", rd)
    assert r.returncode == 1 and "no usable GPU" in r.stderr
    r = inproc("cellwarden.stages.s1b_doublets", rd, {"CELLWARDEN_ALLOW_CPU_AS_GPU": "1"})
    assert r.returncode == 0, r.stderr
    assert "cpu-as-gpu" in r.stdout


# --------------------------------------------------------------------------
# S1c
# --------------------------------------------------------------------------


def test_s1c_outputs(done):
    rd = done["rd"]
    dec = read_tsv_str(rd / "S1" / "qc_decisions.tsv.gz")
    assert list(dec.columns) == ["cell_id", "sample_id", "decision", "reasons", "policy_hash", "decided_by"]
    assert list(dec["cell_id"]) == list(done["after_b"]["cell_id"])
    assert set(dec["decision"]) <= {"keep", "exclude", "review"}
    assert (dec["decided_by"] == "rules").all()
    pol = qc.load_policy(rd / "config" / "qc_policy.yaml")
    assert (dec["policy_hash"] == pol.hash).all()
    assert (dec.loc[dec.decision == "keep", "reasons"] == "").all()

    s = json.loads((rd / "S1" / "qc_summary.json").read_text())
    for k in ("thresholds", "counts", "counts_per_sample", "counts_per_reason", "protected_populations",
              "triggered_review_rules", "policy_hash", "run_id", "config_hash"):
        assert k in s, k
    assert s["policy_hash"] == pol.hash
    assert s["run_id"] == C.load_resolved(rd)["run_id"]
    assert set(s["thresholds"]["per_sample"]) == set(SAMPLES)
    assert s["thresholds"]["per_sample"]["SC"]["adaptive_applied"] is False
    assert sum(s["counts"].values()) == len(dec)
    # The exchange file is deleted once S1c has written its outputs.
    assert not (rd / "S1" / "metrics_counts.h5ad").exists()
    assert s["deleted_exchange"]["path"] == "S1/metrics_counts.h5ad"


def test_s1c_flags_planted_cells(done):
    dec = read_tsv_str(done["rd"] / "S1" / "qc_decisions.tsv.gz").set_index("cell_id")
    m = done["after_b"].set_index("cell_id")
    kind = done["a"].obs["kind"]
    lowq = kind.index[kind == "lowq"]
    assert (dec.loc[lowq, "decision"] != "keep").all()
    assert dec.loc[lowq, "reasons"].str.contains("LOW_GENES|LOW_UMIS|ADAPT_LOW").all()
    highmt = kind.index[kind == "highmt"]
    assert (dec.loc[highmt, "decision"] != "keep").all()
    assert dec.loc[highmt, "reasons"].str.contains("HIGH_MT").all()
    called = m.index[m["doublet_call"] == "TRUE"]
    assert (dec.loc[called, "decision"] == "exclude").all()
    assert dec.loc[called, "reasons"].str.contains("DOUBLET").all()
    singlets = kind.index[kind == "singlet"]
    assert (dec.loc[singlets, "decision"] == "keep").mean() > 0.85


def test_s1c_matches_engine(done):
    rd = done["rd"]
    pol = qc.load_policy(rd / "config" / "qc_policy.yaml")
    met = qc.read_metrics(rd / "S1" / "qc_metrics.tsv.gz")
    want = qc.decide(met, qc.compute_thresholds(met, pol), pol).table
    got = read_tsv_str(rd / "S1" / "qc_decisions.tsv.gz")
    pd.testing.assert_frame_equal(got.reset_index(drop=True), want.astype(str).reset_index(drop=True))


def test_s1c_needs_s1b(inproc, tmp_path):
    rd, _ = make_run(tmp_path)
    r = inproc("cellwarden.stages.s1c_qc", rd)
    assert r.returncode == 1 and "S1b" in r.stderr
    assert (rd / "S1" / "metrics_counts.h5ad").exists()


def test_s1c_bad_policy_keeps_exchange(inproc, tmp_path):
    rd, _ = make_run(tmp_path, {"doublets": {"method": "none"}})
    assert inproc("cellwarden.stages.s1b_doublets", rd).returncode == 0
    (rd / "config" / "qc_policy.yaml").write_text("hard_limits:\n  min_genes: -1\n")
    r = inproc("cellwarden.stages.s1c_qc", rd)
    assert r.returncode == 1 and "min_genes" in r.stderr
    assert (rd / "S1" / "metrics_counts.h5ad").exists()
    assert not (rd / "S1" / "qc_decisions.tsv.gz").exists()


# --------------------------------------------------------------------------
# Through the runner: dummy R stages, real Python stages
# --------------------------------------------------------------------------


def test_runner_with_real_python_stages(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    a = synthetic(seed=3)
    write_exchange(a, src / "metrics_counts.h5ad", "full", run_id="x", stage="S1a")
    write_gz(partial_metrics(a), src / "qc_metrics.tsv.gz")

    def r_dummy(sid, outputs, copy=()):
        code = "import os,pathlib,shutil,sys\nrd=pathlib.Path(os.environ['CELLWARDEN_RUN_DIR'])\n(rd/'S1').mkdir(exist_ok=True)\n"
        for o in outputs:
            code += f"p=rd/{o!r}; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('x')\n"
        for name in copy:
            code += f"shutil.copy({str(src / name)!r}, rd/'S1'/{name!r})\n"
        return lambda ctx: [sys.executable, "-c", code]

    real = {s.id: s for s in R.default_stages()}
    stages = [
        R.Stage("S0", "R", r_dummy("S0", ["S0/object.rds", "S0/ingest_report.json"]), real["S0"].config_keys,
                real["S0"].inputs, real["S0"].outputs, external_inputs=real["S0"].external_inputs),
        R.Stage("S1a", "R", r_dummy("S1a", [], ["metrics_counts.h5ad", "qc_metrics.tsv.gz"]), real["S1a"].config_keys,
                real["S1a"].inputs, real["S1a"].outputs, real["S1a"].transient),
        real["S1b"],
        real["S1c"],
        R.Stage("S2", "R", r_dummy("S2", ["S2/object.rds", "S2/summary.json"]), real["S2"].config_keys,
                real["S2"].inputs, real["S2"].outputs),
    ]
    proj = tmp_path / "proj"
    proj.mkdir()
    for s in SAMPLES:
        (proj / f"{s}.h5ad").write_text("placeholder")
    (proj / "samples.csv").write_text("sample_id,path\n" + "".join(f"{s},{s}.h5ad\n" for s in SAMPLES))
    (proj / "config.yaml").write_text(yaml.safe_dump({
        "project": "e2e", "run_dir": "run", "samples": "samples.csv", "doublets": {"method": "none"},
        "resources": {"threads": 2, "max_memory_gb": 8, "gpu_mem_cap_gb": 4, "nice": 0}}))
    out = io.StringIO()
    code = R.main(["run", "--config", str(proj / "config.yaml")], stages=stages, out=out)
    assert code == 0, out.getvalue()
    rd = proj / "run"
    st = json.loads((rd / "state.json").read_text())
    assert all(st["stages"][s]["status"] == "done" for s in ("S0", "S1a", "S1b", "S1c", "S2"))
    assert (rd / "S1" / "qc_decisions.tsv.gz").exists() and not (rd / "S1" / "metrics_counts.h5ad").exists()
    assert "S1b: done" in (rd / "logs" / "S1b.log").read_text()
    # Changing only the clustering resolution reruns only S2.
    raw = yaml.safe_load((proj / "config.yaml").read_text())
    raw["clustering"] = {"resolution": 1.0}
    (proj / "config.yaml").write_text(yaml.safe_dump(raw))
    out = io.StringIO()
    assert R.main(["run", "--config", str(proj / "config.yaml")], stages=stages, out=out) == 0
    text = out.getvalue()
    assert text.count("— skipped (") == 4 and "S2   Normalise, PCA, batch correction, clustering — running" in text
