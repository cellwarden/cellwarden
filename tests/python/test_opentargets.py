"""Tests for cellwarden.opentargets (client, cache, offline mode, summary) and the S4 Open Targets step.

Never touch the network: ``requests.post`` is replaced by a fake that serves
the recorded responses in tests/fixtures/opentargets/ (routing by GraphQL
operation name and variables) or simulates failures.

    PYTHONPATH=python python3 -m pytest -q tests/python/test_opentargets.py
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import pytest
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from cellwarden.opentargets import (  # noqa: E402
    OpenTargetsClient,
    OpenTargetsError,
    cache_key,
    configure,
    search_entity,
    target_by_symbol,
)
from cellwarden.opentargets import client as C  # noqa: E402
from cellwarden.opentargets import queries as Q  # noqa: E402
from cellwarden.opentargets.summary import build_summary, read_dge_table  # noqa: E402
from cellwarden.stages import s4_opentargets  # noqa: E402

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "opentargets"
GENES = json.loads((FIX / "manifest.json").read_text())["genes"]


# ------------------------------------------------------------------ fake network
class FakeResponse:
    def __init__(self, body, status=200):
        self._body = body
        self.status_code = status
        self.text = json.dumps(body) if not isinstance(body, str) else body

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


def route(query: str, variables: dict) -> Path | None:
    m = re.search(r"^query\s+(\w+)", query, re.M)
    op = m.group(1) if m else ""
    if op == "TargetSearch":
        p = FIX / "target_search" / f"{variables['text']}.json"
        return p if p.exists() else FIX / "target_search" / f"{variables['text'].upper()}.json"
    if op == "TargetEvidence":
        return FIX / "target_evidence" / f"{variables['id']}.json"
    if op == "EntitySearch":
        ent = variables["entities"][0]
        return FIX / "entity_search" / f"{ent}__{variables['text'].replace(' ', '_')}.json"
    return None


class FakeNet:
    """Serves fixture files; counts calls; can be scripted to fail."""

    def __init__(self, fail_first: int = 0, fail_status: int = 503, exc: type | None = None, errors=None):
        self.calls: list[dict] = []
        self.fail_first = fail_first
        self.fail_status = fail_status
        self.exc = exc
        self.errors = errors

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "payload": json, "headers": headers, "timeout": timeout})
        assert url == Q.ENDPOINT
        assert headers["Content-Type"] == "application/json"
        if self.fail_first > 0:
            self.fail_first -= 1
            if self.exc is not None:
                raise self.exc("simulated")
            return FakeResponse({"errors": [{"message": "unavailable"}]}, status=self.fail_status)
        if self.errors is not None:
            return FakeResponse({"errors": self.errors})
        p = route(json["query"], json["variables"])
        if p is None or not p.exists():
            return FakeResponse({"data": {"search": {"total": 0, "hits": []}, "target": None}})
        return FakeResponse(__import__("json").loads(p.read_text()))


@pytest.fixture
def net(monkeypatch):
    fake = FakeNet()
    monkeypatch.setattr(requests, "post", fake.post)
    return fake


@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(requests, "post", boom)


def client(tmp_path, net=None, **kw):
    sleeps: list[float] = []
    c = OpenTargetsClient(tmp_path / "cache", sleep=sleeps.append, backoff=0.5, **kw)
    c._sleeps = sleeps  # type: ignore[attr-defined]
    return c


# ------------------------------------------------------------------ client
def test_target_by_symbol_shape(tmp_path, net):
    c = client(tmp_path)
    t = c.target_by_symbol("MS4A1")
    assert t["ensembl_id"] == "ENSG00000156738" and t["approved_symbol"] == "MS4A1"
    assert t["symbol_match"] == "exact" and t["lookup"] is None if "lookup" in t else True
    assert isinstance(t["tractability"], list) and {"modality", "id", "value"} <= set(t["tractability"][0])
    assert "AB" in t["tractability_summary"]
    kd = t["known_drugs"]
    assert kd and {"drug", "drug_id", "phase", "disease", "disease_id", "mechanism", "evidence_id"} <= set(kd[0])
    assert any(d["drug"] == "RITUXIMAB" and d["phase"] == 4 for d in kd)
    assert kd[0]["evidence_id"].startswith("OT:ENSG00000156738:drug:")
    ad = t["associated_diseases"]
    assert len(ad) == 10 and {"efo_id", "name", "score", "evidence_id"} <= set(ad[0])
    assert ad[0]["evidence_id"] == f"OT:ENSG00000156738:disease:{ad[0]['efo_id']}"
    # two requests: search then target
    ops = [re.search(r"query (\w+)", x["payload"]["query"]).group(1) for x in net.calls]
    assert ops == ["TargetSearch", "TargetEvidence"]
    assert net.calls[0]["timeout"] == 30.0


def test_symbol_resolution_prefers_exact_match(tmp_path, net):
    c = client(tmp_path)
    # CD3E's search has decoys CD3D / CD3G after the exact hit; PPBP has a pseudogene decoy
    assert c.resolve_symbol("CD3E")["ensembl_id"] == "ENSG00000198851"
    assert c.resolve_symbol("ppbp")["ensembl_id"] == "ENSG00000163736"
    assert c.resolve_symbol("NOSUCHGENE1") == {}
    assert c.target_by_symbol("NOSUCHGENE1") == {}
    assert c.target_by_ensembl("ENSG00000000000") == {}


def test_all_fixture_genes_resolve(tmp_path, net):
    c = client(tmp_path)
    for sym, ens in GENES.items():
        t = c.target_by_symbol(sym)
        assert t["ensembl_id"] == ens and t["approved_symbol"] == sym
        assert len(t["associated_diseases"]) == 10
    assert c.target_by_symbol("PF4")["known_drugs"] == []


def test_search_entity(tmp_path, net):
    c = client(tmp_path)
    hits = c.search_entity("systemic lupus erythematosus", "disease")
    assert hits[0] == {"id": "EFO_0002690", "name": "systemic lupus erythematosus", "score": 93.7, "entity": "disease",
                       "description": hits[0]["description"]}
    assert all({"id", "name", "score", "entity"} <= set(h) for h in hits)
    assert net.calls[-1]["payload"]["variables"] == {"text": "systemic lupus erythematosus", "entities": ["disease"], "size": 10}
    with pytest.raises(OpenTargetsError):
        c.search_entity("x", "gene")


def test_cache_hit_and_key(tmp_path, net):
    c = client(tmp_path)
    c.target_by_symbol("LYZ")
    n = len(net.calls)
    assert n == 2
    files = sorted((tmp_path / "cache").glob("*.json"))
    assert len(files) == 2
    assert all(re.fullmatch(r"[0-9a-f]{64}\.json", f.name) for f in files)
    k = cache_key(Q.TARGET_SEARCH, {"text": "LYZ"})
    rec = json.loads((tmp_path / "cache" / f"{k}.json").read_text())
    assert rec["query"] == Q.TARGET_SEARCH and rec["variables"] == {"text": "LYZ"} and "data" in rec
    assert rec["fetched_utc"].endswith("Z")
    # second client, same cache dir: no network
    c2 = client(tmp_path)
    t = c2.target_by_symbol("LYZ")
    assert t["approved_symbol"] == "LYZ"
    assert len(net.calls) == n and c2.cache_hits == 2 and c2.network_calls == 0
    # key depends on query text and variables
    assert cache_key(Q.TARGET_SEARCH, {"text": "LYZ"}) != cache_key(Q.TARGET_SEARCH, {"text": "lyz"})
    assert cache_key(Q.TARGET_SEARCH, {"text": "LYZ"}) != cache_key(Q.TARGET_EVIDENCE, {"text": "LYZ"})
    # refresh ignores the cache and rewrites it
    c3 = client(tmp_path, refresh=True)
    c3.target_by_symbol("LYZ")
    assert len(net.calls) == n + 2 and c3.cache_hits == 0


def test_offline_reads_only_cache(tmp_path, net, monkeypatch):
    c = client(tmp_path)
    c.target_by_symbol("NKG7")
    c.search_entity("systemic lupus erythematosus", "disease")
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    off = client(tmp_path, offline=True)
    assert off.target_by_symbol("NKG7")["ensembl_id"] == "ENSG00000105374"
    assert off.search_entity("systemic lupus erythematosus", "disease")[0]["id"] == "EFO_0002690"
    assert off.target_by_symbol("PPBP") is None           # not cached -> miss, no network
    assert off.target_by_ensembl("ENSG00000163736") is None
    assert len(off.offline_misses) == 2 and off.offline_misses[0]["variables"] == {"text": "PPBP"}
    assert off.network_calls == 0 and off.stats()["offline"] is True


def test_retry_with_backoff_then_success(tmp_path, monkeypatch):
    fake = FakeNet(fail_first=2, fail_status=503)
    monkeypatch.setattr(requests, "post", fake.post)
    c = client(tmp_path, retries=3)
    t = c.target_by_ensembl("ENSG00000163737")
    assert t["approved_symbol"] == "PF4"
    assert len(fake.calls) == 3 and c._sleeps == [0.5, 1.0]


def test_retry_on_connection_error(tmp_path, monkeypatch):
    fake = FakeNet(fail_first=1, exc=requests.ConnectionError)
    monkeypatch.setattr(requests, "post", fake.post)
    c = client(tmp_path, retries=3)
    assert c.target_by_ensembl("ENSG00000163737")["approved_symbol"] == "PF4"
    assert len(fake.calls) == 2


def test_gives_up_after_retries(tmp_path, monkeypatch):
    fake = FakeNet(fail_first=10, exc=requests.Timeout)
    monkeypatch.setattr(requests, "post", fake.post)
    c = client(tmp_path, retries=3)
    with pytest.raises(OpenTargetsError, match="after 4 attempt"):
        c.target_by_ensembl("ENSG00000163737")
    assert len(fake.calls) == 4 and c._sleeps == [0.5, 1.0, 2.0]
    assert not list((tmp_path / "cache").glob("*.json"))      # failures are not cached


def test_graphql_errors_and_bad_status(tmp_path, monkeypatch):
    fake = FakeNet(errors=[{"message": "Cannot query field 'nope'"}])
    monkeypatch.setattr(requests, "post", fake.post)
    with pytest.raises(OpenTargetsError, match="Cannot query field"):
        client(tmp_path).target_by_ensembl("ENSG00000163737")
    fake = FakeNet(fail_first=1, fail_status=400)
    monkeypatch.setattr(requests, "post", fake.post)
    with pytest.raises(OpenTargetsError, match="HTTP 400"):
        client(tmp_path, retries=3).target_by_ensembl("ENSG00000163737")
    assert len(fake.calls) == 1     # 4xx other than 429 is not retried


def test_module_level_functions(tmp_path, net):
    C._DEFAULT = None
    with pytest.raises(OpenTargetsError, match="not configured"):
        target_by_symbol("PF4")
    configure(tmp_path / "cache", sleep=lambda s: None)
    assert target_by_symbol("PF4")["ensembl_id"] == "ENSG00000163737"
    assert search_entity("systemic lupus erythematosus", "disease")[0]["id"] == "EFO_0002690"
    C._DEFAULT = None


# ------------------------------------------------------------------ summary
DGE_COLS = ["gene", "baseMean", "log2fc", "lfcSE", "pvalue", "padj", "direction"]


def write_dge(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write("\t".join(DGE_COLS) + "\n")
        for r in rows:
            fh.write("\t".join("" if v is None else str(v) for v in r) + "\n")


def make_run(tmp_path, *, ensembl=True, cfg_extra=None):
    """A run dir with one contrast, two groups (one skipped) and a gene map."""
    rd = tmp_path / "run"
    (rd / "config").mkdir(parents=True)
    (rd / "S4" / "dge").mkdir(parents=True)
    symbols = ["PPBP", "PF4", "CD3E", "MS4A1", "LYZ", "NKG7", "NOSUCHGENE1", "FILLER1", "FILLER2"]
    gm = {}
    for s in symbols:
        gm[s] = {"symbol": s, "ensembl_id": GENES.get(s, "") if ensembl else ""}
    keys = {s: (GENES[s] if (ensembl and s in GENES) else s) for s in symbols}
    with open(rd / "S4" / "gene_map.tsv", "w") as fh:
        fh.write("gene\tsymbol\tensembl_id\n")
        for s in symbols:
            fh.write(f"{keys[s]}\t{s}\t{gm[s]['ensembl_id']}\n")
    rows = [
        (keys["MS4A1"], 120.0, 2.5, 0.3, 1e-9, 1e-7, "up"),
        (keys["CD3E"], 80.0, 1.9, 0.3, 1e-8, 5e-7, "up"),
        (keys["NOSUCHGENE1"], 10.0, 1.2, 0.4, 1e-4, 0.01, "up"),
        (keys["FILLER1"], 5.0, 0.9, 0.4, 2e-4, 0.02, "up"),
        (keys["PPBP"], 300.0, -3.1, 0.4, 1e-10, 1e-8, "down"),
        (keys["PF4"], 200.0, -2.7, 0.4, 2e-9, 5e-8, "down"),
        (keys["LYZ"], 500.0, -1.0, 0.2, 1e-5, 0.004, "down"),
        (keys["NKG7"], 50.0, 0.1, 0.3, 0.5, 0.9, "ns"),
        (keys["FILLER2"], 3.0, 0.4, 0.5, None, None, "ns"),
    ]
    write_dge(rd / "S4" / "dge" / "treated_vs_control__B.tsv.gz", rows)
    dge_summary = {
        "stage": "S4", "thresholds": {"padj": 0.05, "lfc": 0.5},
        "contrasts": [{"name": "treated_vs_control", "numerator": "treated", "denominator": "control", "groups": [
            {"group": "B", "file": "S4/dge/treated_vs_control__B.tsv.gz", "n_up": 4, "n_down": 3, "n_tested": 9, "skipped_reason": None},
            {"group": "NK", "file": None, "n_up": 0, "n_down": 0, "n_tested": 0,
             "skipped_reason": "1 treated and 2 control pseudobulk sample(s) with >= 20 cells; min_replicates is 2 per side"},
        ]}],
    }
    (rd / "S4" / "dge_summary.json").write_text(json.dumps(dge_summary))
    cfg = {"run_id": "t-1", "config_hash": "abc", "config_dir": str(tmp_path),
           "opentargets": {"enabled": True, "max_genes_per_direction": 3, "cache_dir": str(tmp_path / "cache"),
                           "offline": False, "timeout_s": 5, "retries": 1}}
    if cfg_extra:
        cfg["opentargets"].update(cfg_extra)
    (rd / "config" / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    return rd, cfg


def test_read_dge_table(tmp_path):
    rd, _ = make_run(tmp_path)
    rows = read_dge_table(rd / "S4" / "dge" / "treated_vs_control__B.tsv.gz")
    assert len(rows) == 9 and rows[0]["gene"] == GENES["MS4A1"] and rows[0]["log2fc"] == 2.5
    assert rows[-1]["padj"] is None and rows[-1]["direction"] == "ns"
    bad = tmp_path / "bad.tsv.gz"
    with gzip.open(bad, "wt") as fh:
        fh.write("gene\tlog2fc\n")
    with pytest.raises(ValueError, match="missing column"):
        read_dge_table(bad)


def test_summary_per_direction_ranked_and_capped(tmp_path, net):
    rd, cfg = make_run(tmp_path)
    out = s4_opentargets.run(rd, cfg)
    assert (rd / "S4" / "opentargets_summary.json").exists()
    assert out["stage"] == "S4" and out["step"] == "opentargets" and out["enabled"] is True
    assert out["gene_id_source"] == "ensembl_id" and out["run_id"] == "t-1"
    assert out["max_genes_per_direction"] == 3 and out["thresholds"] == {"padj": 0.05, "lfc": 0.5}
    c = out["contrasts"][0]
    assert c["contrast"] == "treated_vs_control" and c["numerator"] == "treated"
    b, nk = c["groups"]
    assert nk["skipped_reason"] and nk["directions"]["up"]["genes"] == [] and nk["directions"]["down"]["n_de"] == 0
    up = b["directions"]["up"]
    assert up["n_de"] == 4 and up["n_listed"] == 3                   # capped at max_genes_per_direction
    assert [g["symbol"] for g in up["genes"]] == ["MS4A1", "CD3E", "NOSUCHGENE1"]   # ordered by padj
    assert [g["rank"] for g in up["genes"]] == [1, 2, 3]
    assert all(g["direction"] == "up" for g in up["genes"])
    down = b["directions"]["down"]
    assert [g["symbol"] for g in down["genes"]] == ["PPBP", "PF4", "LYZ"] and down["n_de"] == 3
    assert down["genes"][0]["log2fc"] == -3.1 and down["genes"][0]["direction"] == "down"
    # evidence: has_record + pointer into targets; direction never comes from Open Targets
    g = up["genes"][0]
    assert g["has_record"] is True and g["target"] == GENES["MS4A1"] and g["ensembl_id"] == GENES["MS4A1"]
    t = out["targets"][GENES["MS4A1"]]
    assert t["approved_symbol"] == "MS4A1" and t["lookup"] == "ensembl_id" and t["known_drugs"]
    assert "direction" not in t
    assert set(out["targets"]) == {GENES[s] for s in ("MS4A1", "CD3E", "PPBP", "PF4", "LYZ")}
    # no-record genes listed; NKG7 (ns) never queried
    assert out["genes_without_record"] == ["NOSUCHGENE1"]
    assert up["genes"][2]["has_record"] is False and up["genes"][2]["target"] is None
    assert up["n_with_record"] == 2 and down["n_with_record"] == 3
    assert GENES["NKG7"] not in out["targets"]
    cnt = out["counts"]
    assert cnt["genes_listed"] == 6 and cnt["genes_unique"] == 6 and cnt["genes_with_record"] == 5
    assert cnt["genes_without_record"] == 1 and cnt["groups"] == 2 and cnt["groups_with_tables"] == 1
    assert cnt["network_calls"] == 6 and cnt["known_drug_rows"] > 0
    # ensembl-id genes are looked up directly: one request each, no symbol search
    ops = [re.search(r"query (\w+)", x["payload"]["query"]).group(1) for x in net.calls]
    assert ops.count("TargetSearch") == 1 and ops.count("TargetEvidence") == 5
    # evidence ids are unique and well formed
    ids = [d["evidence_id"] for t in out["targets"].values() for d in t["known_drugs"] + t["associated_diseases"]]
    assert all(i.startswith("OT:ENSG") for i in ids)


def test_summary_symbol_only_genes(tmp_path, net):
    rd, cfg = make_run(tmp_path, ensembl=False)
    out = s4_opentargets.run(rd, cfg)
    assert out["gene_id_source"] == "symbol"
    t = out["targets"]["MS4A1"]
    assert t["ensembl_id"] == GENES["MS4A1"] and t["lookup"] == "symbol" and t["symbol_match"] == "exact"
    g = out["contrasts"][0]["groups"][0]["directions"]["up"]["genes"][0]
    assert g["gene"] == "MS4A1" and g["ensembl_id"] == GENES["MS4A1"]


def test_summary_offline_misses_and_disabled(tmp_path, net):
    rd, cfg = make_run(tmp_path)
    # warm the cache for two genes only
    c = OpenTargetsClient(tmp_path / "cache", sleep=lambda s: None)
    c.target_by_ensembl(GENES["MS4A1"])
    c.target_by_ensembl(GENES["PPBP"])
    n = len(net.calls)
    cfg["opentargets"]["offline"] = True
    out = s4_opentargets.run(rd, cfg)
    assert len(net.calls) == n and out["offline"] is True
    assert set(out["targets"]) == {GENES["MS4A1"], GENES["PPBP"]}
    assert out["genes_not_resolved_offline"] == sorted([GENES["CD3E"], GENES["PF4"], GENES["LYZ"], "NOSUCHGENE1"])
    assert out["counts"]["genes_not_resolved_offline"] == 4 and out["counts"]["network_calls"] == 0
    g = out["contrasts"][0]["groups"][0]["directions"]["up"]["genes"][1]
    assert g["symbol"] == "CD3E" and g["has_record"] is False and g["lookup_status"] == "offline-miss"
    # disabled: summary still written, no lookups
    cfg["opentargets"]["offline"] = False
    cfg["opentargets"]["enabled"] = False
    out = s4_opentargets.run(rd, cfg)
    assert out["enabled"] is False and out["targets"] == {} and len(net.calls) == n
    assert out["contrasts"][0]["groups"][0]["directions"]["up"]["genes"][0]["lookup_status"] == "not-queried"


def test_summary_records_errors_without_failing(tmp_path, monkeypatch):
    fake = FakeNet(fail_first=100, exc=requests.ConnectionError)
    monkeypatch.setattr(requests, "post", fake.post)
    rd, cfg = make_run(tmp_path)
    cfg["opentargets"]["retries"] = 0
    out = s4_opentargets.run(rd, cfg)
    assert out["targets"] == {} and len(out["errors"]) == 6
    assert out["errors"][0]["gene"] and "unreachable" in out["errors"][0]["error"]
    assert out["counts"]["genes_with_error"] == 6


def test_stage_cli(tmp_path, net, capsys):
    rd, cfg = make_run(tmp_path)
    assert s4_opentargets.main(["--run-dir", str(rd)]) == 0
    out = json.loads((rd / "S4" / "opentargets_summary.json").read_text())
    assert out["counts"]["genes_with_record"] == 5
    assert "5 with an Open Targets record" in capsys.readouterr().out
    (rd / "S4" / "dge_summary.json").unlink()
    assert s4_opentargets.main(["--run-dir", str(rd)]) == 1
    assert "dge_summary.json not found" in capsys.readouterr().err


def test_s4_driver_skip_dge(tmp_path, net):
    from cellwarden.stages import s4

    rd, cfg = make_run(tmp_path)
    assert s4.main(["--run-dir", str(rd), "--skip-dge"]) == 0
    assert (rd / "S4" / "opentargets_summary.json").exists()


def test_config_validates_contrasts_against_sample_sheet(tmp_path):
    """dge.contrasts must name values of the sample sheet's condition column; cache_dir becomes absolute."""
    from cellwarden import config as C

    proj = tmp_path / "proj"
    (proj / "data" / "A").mkdir(parents=True)
    for f in ("matrix.mtx", "features.tsv", "barcodes.tsv"):
        (proj / "data" / "A" / f).write_text("x")
    (proj / "data" / "B.h5ad").write_text("x")
    (proj / "samples.csv").write_text("sample_id,path,format,condition,batch\nA,data/A,,control,B1\nB,data/B.h5ad,h5ad,treated,B2\n")

    def load(contrasts, **dge):
        cfg = {"project": "demo", "run_dir": "runs/demo", "samples": "samples.csv",
               "dge": {"contrasts": contrasts, **dge}, "opentargets": {"cache_dir": "refs/ot"}}
        (proj / "config.yaml").write_text(yaml.safe_dump(cfg))
        return C.load_config(proj / "config.yaml")

    good = [{"name": "treated_vs_control", "numerator": "treated", "denominator": "control", "covariates": ["batch"]}]
    cfg = load(good)
    assert cfg["dge"]["contrasts"] == good and cfg["dge"]["method"] == "findmarkers" and cfg["dge"]["test"] == "MAST"
    assert cfg["opentargets"]["cache_dir"] == str(proj / "refs" / "ot")
    with pytest.raises(C.ConfigError, match="'disease' is not a value of samples.csv column 'condition'"):
        load([{"name": "x", "numerator": "disease", "denominator": "control"}])
    with pytest.raises(C.ConfigError, match="'donor' is not a samples.csv column"):
        load([{"name": "x", "numerator": "treated", "denominator": "control", "covariates": ["donor"]}], method="deseq2")
    with pytest.raises(C.ConfigError, match="column 'group' is not a samples.csv column"):
        load(good, condition_col="group", method="pseudobulk")
    # findmarkers also takes cell-level columns (checked by S4) and lists / 'rest' on either side
    load([{"name": "b", "column": "cell_type_broad", "numerator": ["B cell", "plasma cell"], "denominator": "rest"}])
    load([{"name": "t", "column": "condition", "numerator": "treated", "denominator": "rest", "within": "none"}])
    with pytest.raises(C.ConfigError, match="numerator cannot be 'rest'"):
        load([{"name": "x", "numerator": "rest", "denominator": "control"}])
    with pytest.raises(C.ConfigError, match="dge.test"):
        load(good, test="magic")
    with pytest.raises(C.ConfigError, match="used twice"):
        load(good + good)
    with pytest.raises(C.ConfigError, match="dge.method"):
        load(good, method="edger")


def test_no_network_in_tests(tmp_path, no_network):
    """Sanity: an unpatched call path would be caught (every test above patches requests.post)."""
    c = client(tmp_path)
    with pytest.raises(AssertionError, match="network access attempted"):
        c.target_by_ensembl("ENSG00000163737")


def test_shape_target_reads_the_current_api():
    """drugAndClinicalCandidates (one row per drug with its diseases) and tractability.label, as served
    since October 2026; each drug's first disease comes before further ones, no pair twice."""
    from cellwarden.opentargets.client import shape_target

    t = {"id": "ENSG1", "approvedSymbol": "MS4A1", "tractability": [{"modality": "AB", "label": "GO CC high conf", "value": True}],
         "drugAndClinicalCandidates": {"count": 2, "rows": [
             {"maxClinicalStage": "PHASE_2", "drug": {"id": "D2", "name": "two"}, "diseases": [{"disease": {"id": "X", "name": "x"}}]},
             {"maxClinicalStage": "APPROVAL", "drug": {"id": "D1", "name": "one",
                                                       "mechanismsOfAction": {"rows": [{"mechanismOfAction": "CD20 binder"}]}},
              "diseases": [{"disease": None}, {"disease": {"id": "X", "name": "x"}}, {"disease": {"id": "X", "name": "x"}},
                           {"disease": {"id": "Y", "name": "y"}}]}]},
         "associatedDiseases": {"count": 0, "rows": []}}
    r = shape_target(t, n_drugs=10)
    assert [(d["drug"], d["disease_id"], d["phase"]) for d in r["known_drugs"]] == [("one", "X", 4), ("two", "X", 2), ("one", "Y", 4)]
    assert r["known_drugs"][0]["mechanism"] == "CD20 binder" and r["known_drugs"][0]["status"] == "approval"
    assert r["known_drugs"][0]["evidence_id"] == "OT:ENSG1:drug:D1:X" and r["known_drugs_count"] == 2
    assert r["tractability"] == [{"modality": "AB", "id": "GO CC high conf", "value": True}] and r["tractability_summary"] == ["AB"]
    assert len(shape_target(t, n_drugs=2)["known_drugs"]) == 2
