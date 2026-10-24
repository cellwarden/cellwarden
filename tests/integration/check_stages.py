#!/usr/bin/env python3
"""Checks for the S0-S2 end-to-end test (docs/stages-spec.md section 6).

Called by tests/integration/run_stages.sh, one subcommand per step. Each
subcommand prints what it found and exits 0 (pass) or 1 (fail).

    check_stages.py state      RUN                      all stages done
    check_stages.py ingest     RUN INPUTS               S0 report vs fixture manifest
    check_stages.py metrics    RUN INPUTS               S1 metrics vs an independent computation
    check_stages.py qc-truth   RUN TRUTH                injected populations handled as expected
    check_stages.py qc-same    RUN_A RUN_B              QC decisions identical
    check_stages.py s2         RUN TRUTH OUT_TSV --device cpu|gpu
                                                        export S2 object (Rscript), check it,
                                                        purity of clusters vs planted types
    check_stages.py ari        TSV_A TSV_B              ARI between two cluster exports
    check_stages.py snapshot   RUN OUT_JSON             save state.json
    check_stages.py skipped    SNAPSHOT RUN             nothing reran since the snapshot
    check_stages.py reran      SNAPSHOT RUN --stages S2 only the listed stages reran

Thresholds can be overridden with environment variables (see THRESH below).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ["S0", "S1a", "S1b", "S1c", "S2"]
THRESH = {
    "lowq_excluded": float(os.environ.get("MIN_LOWQ_EXCLUDED", 0.90)),
    # Deliberately low: scrublet ranks the planted doublets well (AUC about 0.9) but its automatic
    # threshold only calls 20-40% of them in ~280-cell samples. The measured rate is printed.
    "doublet_excluded": float(os.environ.get("MIN_DOUBLET_EXCLUDED", 0.25)),
    "platelet_review": float(os.environ.get("MIN_PLATELET_REVIEW", 0.80)),
    "none_kept": float(os.environ.get("MIN_NONE_KEPT", 0.85)),
    "purity": float(os.environ.get("MIN_PURITY", 0.90)),
    "ari": float(os.environ.get("MIN_ARI", 0.70)),
}
RSCRIPT = os.environ.get("RSCRIPT", "Rscript")

QC_METRIC_COLUMNS = ["cell_id", "sample_id", "n_genes", "n_umis", "pct_mito", "pct_ribo",
                     "pct_hb", "log10_genes_per_umi", "protected_Platelet",
                     "protected_Erythrocyte", "doublet_score", "doublet_call", "pre_cluster"]
QC_DECISION_COLUMNS = ["cell_id", "sample_id", "decision", "reasons", "policy_hash", "decided_by"]


class Check:
    """Collects pass/fail lines and turns them into an exit code."""

    def __init__(self) -> None:
        self.failed = 0

    def ok(self, cond: bool, msg: str) -> bool:
        print(f"  [{'ok' if cond else 'FAIL'}] {msg}")
        if not cond:
            self.failed += 1
        return cond

    def info(self, msg: str) -> None:
        print(f"  [info] {msg}")

    def done(self) -> int:
        print(f"  -> {'PASS' if not self.failed else f'FAIL ({self.failed} problem(s))'}")
        return 1 if self.failed else 0


# ------------------------------------------------------------------ state.json
def load_state(run: Path) -> dict:
    with open(run / "state.json") as fh:
        return json.load(fh)


def stage_entries(state: dict) -> dict[str, dict]:
    """Per-stage dicts, whether state.json nests them under "stages" or not.

    Accepts {"stages": {"S0": {...}}}, {"S0": {...}} or {"stages": [{"id": "S0", ...}]}.
    """
    st = state.get("stages", state)
    if isinstance(st, list):
        st = {e.get("id") or e.get("stage") or e.get("name"): e for e in st}
    return {k: v for k, v in st.items() if k in STAGES and isinstance(v, dict)}


def _field(entry: dict, *needles: str):
    for k in entry:
        lk = k.lower()
        if any(n in lk for n in needles):
            return entry[k]
    return None


def start_time(entry: dict):
    return _field(entry, "start")


def end_time(entry: dict):
    return _field(entry, "end", "finish")


def fingerprint(entry: dict):
    return _field(entry, "fingerprint", "hash")


def cmd_state(a) -> int:
    c = Check()
    run = Path(a.run)
    c.ok((run / "state.json").exists(), "state.json exists")
    if c.failed:
        return c.done()
    ent = stage_entries(load_state(run))
    for s in STAGES:
        e = ent.get(s)
        if not c.ok(e is not None, f"{s} present in state.json"):
            continue
        status = e.get("status")
        wall = _field(e, "wall")
        rss = _field(e, "rss", "mem")
        c.ok(status == "done", f"{s} status={status} wall={wall} peak_rss={rss} exit={_field(e, 'exit')}")
        c.ok(fingerprint(e) not in (None, ""), f"{s} has an input fingerprint")
        c.ok(start_time(e) is not None and end_time(e) is not None, f"{s} has start and end times")
    for s in STAGES:
        c.ok((run / "logs" / f"{s}.log").exists(), f"logs/{s}.log exists")
    for f in ["config/config.yaml", "config/samples.csv", "config/qc_policy.yaml",
              "config/config.resolved.yaml", "config/hashes.json"]:
        c.ok((run / f).exists(), f"{f} exists")
    return c.done()


def cmd_snapshot(a) -> int:
    Path(a.out).write_text(json.dumps(load_state(Path(a.run)), indent=2))
    ent = stage_entries(load_state(Path(a.run)))
    for s in STAGES:
        e = ent.get(s, {})
        print(f"  {s:4s} status={e.get('status')} start={start_time(e)} end={end_time(e)}")
    return 0


def _compare(snap_path: str, run: str, expect_rerun: set[str]) -> int:
    c = Check()
    before = stage_entries(json.loads(Path(snap_path).read_text()))
    after = stage_entries(load_state(Path(run)))
    for s in STAGES:
        b, n = before.get(s, {}), after.get(s, {})
        changed = (start_time(b), end_time(b)) != (start_time(n), end_time(n))
        c.ok(n.get("status") == "done", f"{s} status={n.get('status')}")
        if s in expect_rerun:
            c.ok(changed, f"{s} reran (start {start_time(b)} -> {start_time(n)})")
            c.info(f"{s} fingerprint {'changed' if fingerprint(b) != fingerprint(n) else 'UNCHANGED'}")
        else:
            c.ok(not changed, f"{s} skipped (start {start_time(b)}{' -> ' + str(start_time(n)) if changed else ''})")
            c.ok(fingerprint(b) == fingerprint(n), f"{s} fingerprint unchanged")
    return c.done()


def cmd_skipped(a) -> int:
    return _compare(a.snapshot, a.run, set())


def cmd_reran(a) -> int:
    return _compare(a.snapshot, a.run, set(a.stages.split(",")))


# ------------------------------------------------------------------ inputs
def read_inputs(inputs: Path) -> pd.DataFrame:
    """Independent per-cell metrics straight from the fixture input files."""
    import anndata as ad
    import scipy.io as sio

    frames = []
    a_dir = inputs / "donor_A"
    with gzip.open(a_dir / "matrix.mtx.gz") as fh, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)          # scipy spmatrix default change
        m_a = sio.mmread(fh).T.tocsr()                             # cells x genes
    g_a = pd.read_csv(a_dir / "features.tsv.gz", sep="\t", header=None)[1].tolist()
    b_a = pd.read_csv(a_dir / "barcodes.tsv.gz", sep="\t", header=None)[0].tolist()
    b = ad.read_h5ad(inputs / "donor_B.h5ad")
    for sid, m, genes, bcs in [("donor_A", m_a, g_a, b_a),
                               ("donor_B", b.layers["counts"].tocsr(), list(b.var_names), list(b.obs_names))]:
        genes = np.array(genes)
        nu = np.asarray(m.sum(1)).ravel().astype(float)
        ng = np.asarray((m > 0).sum(1)).ravel()

        def frac(mask):
            return np.asarray(m[:, mask].sum(1)).ravel() / nu

        mt = np.char.startswith(genes.astype(str), "MT-")
        ribo = np.array([g.startswith(("RPS", "RPL")) for g in genes])
        hb = np.isin(genes, ["HBB", "HBA1", "HBA2"])
        frames.append(pd.DataFrame({
            "cell_id": [f"{sid}_{x}" for x in bcs], "sample_id": sid,
            "n_genes": ng, "n_umis": nu, "pct_mito": 100 * frac(mt), "pct_ribo": 100 * frac(ribo),
            "pct_hb": 100 * frac(hb), "log10_genes_per_umi": np.log10(ng) / np.log10(nu),
            "protected_Platelet": frac(np.isin(genes, ["PPBP", "PF4", "GNG11"])),
            "protected_Erythrocyte": frac(hb),
        }))
    return pd.concat(frames, ignore_index=True).set_index("cell_id")


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""], dtype={"reasons": str})


def cmd_sheets(a) -> int:
    """SHEET:BASE pairs resolve to the same samples, files and metadata."""
    import csv
    c = Check()
    seen = []
    for spec in a.sheets:
        sheet, base = spec.rsplit(":", 1)
        rows = list(csv.DictReader(open(sheet, newline="")))
        for r in rows:
            r["path"] = os.path.realpath(os.path.join(base, r["path"]))
            c.ok(os.path.exists(r["path"]), f"{sheet}: {r['sample_id']} -> {r['path']}")
        seen.append((sheet, rows))
    for sheet, rows in seen[1:]:
        c.ok(rows == seen[0][1], f"{sheet} matches {seen[0][0]}")
    return c.done()


def cmd_ingest(a) -> int:
    c = Check()
    run, inputs = Path(a.run), Path(a.inputs)
    man = json.loads((inputs / "manifest.json").read_text())
    rp = run / "S0" / "ingest_report.json"
    if not c.ok(rp.exists(), "S0/ingest_report.json exists"):
        return c.done()
    obj = [p for p in (run / "S0" / "object.rds", run / "S0" / "object.qs2") if p.exists()]
    c.ok(bool(obj), "S0 object exists")
    rep = json.loads(rp.read_text())
    print("  ingest_report.json:\n" + "\n".join("    " + ln for ln in json.dumps(rep, indent=2).splitlines()[:60]))
    samples = rep.get("samples", {})
    if isinstance(samples, list):
        samples = {s.get("sample_id"): s for s in samples}
    for sid, exp in man["samples"].items():
        s = samples.get(sid)
        if not c.ok(s is not None, f"{sid} in report"):
            continue
        c.ok(s.get("format") == exp["format"], f"{sid} format {s.get('format')} (expected {exp['format']}"
             f"{', auto-detected' if sid == 'donor_A' else ''})")
        c.ok(s.get("cells", s.get("n_cells")) == exp["n_cells"], f"{sid} cells {s.get('cells', s.get('n_cells'))} == {exp['n_cells']}")
        c.ok(s.get("genes", s.get("n_genes")) == exp["n_genes"], f"{sid} genes {s.get('genes', s.get('n_genes'))} == {exp['n_genes']}")
        umis = s.get("total_umis", s.get("umis"))
        c.ok(umis is not None and int(umis) == exp["total_umis"], f"{sid} total UMIs {umis} == {exp['total_umis']}")
    tot = rep.get("totals", {})
    c.ok(tot.get("genes", tot.get("n_genes")) == man["n_genes_union"],
         f"merged genes {tot.get('genes', tot.get('n_genes'))} == union {man['n_genes_union']} "
         f"(genes missing from donor_B count as zero)")
    c.ok(tot.get("cells", tot.get("n_cells")) == sum(v["n_cells"] for v in man["samples"].values()),
         f"merged cells {tot.get('cells', tot.get('n_cells'))}")
    key = rep.get("gene_key", tot.get("gene_key"))
    c.info(f"gene key chosen: {key} (both samples carry Ensembl IDs: donor_A in features.tsv column 1, "
           f"donor_B in var['gene_ids'])")
    return c.done()


def cmd_metrics(a) -> int:
    c = Check()
    run = Path(a.run)
    p = run / "S1" / "qc_metrics.tsv.gz"
    if not c.ok(p.exists(), "S1/qc_metrics.tsv.gz exists"):
        return c.done()
    m = read_tsv(p)
    missing = [k for k in QC_METRIC_COLUMNS if k not in m.columns]
    c.ok(not missing, f"columns present (missing: {missing or 'none'}); all: {list(m.columns)}")
    c.ok(m["cell_id"].is_unique, f"{len(m)} rows, cell_id unique")
    ref = read_inputs(Path(a.inputs))
    c.ok(set(m["cell_id"]) == set(ref.index), "cell_ids are <sample_id>_<barcode> for every input cell")
    m = m.set_index("cell_id")
    common = ref.index.intersection(m.index)
    if not len(common):
        return c.done()
    for col, tol in [("n_genes", 0), ("n_umis", 0), ("pct_mito", 1e-3), ("pct_ribo", 1e-3),
                     ("pct_hb", 1e-3), ("log10_genes_per_umi", 1e-4),
                     ("protected_Platelet", 1e-5), ("protected_Erythrocyte", 1e-5)]:
        if col not in m.columns:
            continue
        d = np.abs(m.loc[common, col].astype(float) - ref.loc[common, col]).max()
        extra = " (HBP1/HBEGF must not count as haemoglobin)" if col == "pct_hb" else ""
        c.ok(d <= tol, f"{col} matches independent computation (max abs diff {d:.3g}){extra}")
    if "sample_id" in m.columns:
        c.ok((m.loc[common, "sample_id"] == ref.loc[common, "sample_id"]).all(), "sample_id matches")
    if "doublet_score" in m.columns:
        c.ok(m["doublet_score"].notna().all(), "doublet_score has no missing values")
    if "pre_cluster" in m.columns:
        c.info(f"pre_cluster: {m['pre_cluster'].nunique()} clusters")
    if not (run / "S1" / "metrics_counts.h5ad").exists():
        c.info("S1/metrics_counts.h5ad deleted after S1c, as specified")
    else:
        c.ok(False, "S1/metrics_counts.h5ad should be deleted once S1c is done")
    return c.done()


# ------------------------------------------------------------------ QC decisions
def load_decisions(run: Path) -> pd.DataFrame:
    d = read_tsv(run / "S1" / "qc_decisions.tsv.gz")
    d["reasons"] = d["reasons"].fillna("")
    return d


def cmd_qc_truth(a) -> int:
    c = Check()
    run = Path(a.run)
    p = run / "S1" / "qc_decisions.tsv.gz"
    if not c.ok(p.exists(), "S1/qc_decisions.tsv.gz exists"):
        return c.done()
    d = load_decisions(run)
    missing = [k for k in QC_DECISION_COLUMNS if k not in d.columns]
    c.ok(not missing, f"decision columns present (missing: {missing or 'none'})")
    c.ok(set(d["decision"]) <= {"keep", "exclude", "review"}, f"decision values {sorted(set(d['decision']))}")
    c.ok((d.loc[d.decision == "keep", "reasons"] == "").all(), "keep rows have empty reasons")
    c.ok((d.loc[d.decision != "keep", "reasons"] != "").all(), "exclude/review rows have reasons")
    if "decided_by" in d.columns:
        c.ok((d["decided_by"] == "rules").all(), "decided_by == rules")
    if "policy_hash" in d.columns:
        c.ok(d["policy_hash"].nunique() == 1, f"one policy_hash ({d['policy_hash'].iloc[0]})")
    truth = pd.read_csv(a.truth, sep="\t")
    c.ok(set(truth.cell_id) == set(d.cell_id), f"decisions cover every input cell ({len(d)} rows)")
    x = truth.merge(d, on="cell_id", how="left")
    x["reasons"] = x["reasons"].fillna("")

    print("  decision by injected population:")
    tab = pd.crosstab(x["injected"], x["decision"])
    print("    " + tab.to_string().replace("\n", "\n    "))
    codes = Counter(r for rs in x["reasons"] for r in rs.split(";") if r)
    c.info("reason codes: " + ", ".join(f"{k}={v}" for k, v in sorted(codes.items())))

    def rate(mask, cond):
        n = int(mask.sum())
        return (float(cond[mask].mean()) if n else float("nan")), n

    r, n = rate(x.injected == "lowq", x.decision == "exclude")
    c.ok(r >= THRESH["lowq_excluded"], f"low-quality cells excluded: {r:.0%} of {n} (need >= {THRESH['lowq_excluded']:.0%})")
    hm = x.loc[x.injected == "lowq", "reasons"].str.contains("HIGH_MT").mean()
    c.info(f"low-quality cells with HIGH_MT: {hm:.0%}")
    r, n = rate(x.injected == "doublet", x.decision == "exclude")
    dbl_called = x.loc[x.injected == "doublet", "reasons"].str.contains(r"(?:^|;)DOUBLET(?:;|$)").mean()
    c.ok(r >= THRESH["doublet_excluded"],
         f"injected doublets excluded: {r:.0%} of {n} (need >= {THRESH['doublet_excluded']:.0%}; "
         f"called DOUBLET by scrublet: {dbl_called:.0%})")
    fp = x.loc[x.injected == "none", "reasons"].str.contains(r"(?:^|;)DOUBLET(?:;|$)").mean()
    c.info(f"non-injected cells called DOUBLET: {fp:.1%}")
    plt_mask = x.injected == "platelet"
    r, n = rate(plt_mask, (x.decision == "review") & x.reasons.str.contains("PROTECTED_POP:Platelet", regex=False))
    c.ok(r >= THRESH["platelet_review"],
         f"platelets marked review with PROTECTED_POP:Platelet: {r:.0%} of {n} (need >= {THRESH['platelet_review']:.0%})")
    c.info("platelet reasons: " + ", ".join(f"{k}={v}" for k, v in Counter(x.loc[plt_mask, "reasons"]).most_common(4)))
    r, n = rate(x.injected == "none", x.decision == "keep")
    c.ok(r >= THRESH["none_kept"], f"non-injected cells kept: {r:.0%} of {n} (need >= {THRESH['none_kept']:.0%})")
    nr = x.loc[(x.injected == "none") & (x.decision == "review"), "reasons"]
    if len(nr):
        c.info(f"non-injected cells in review: {len(nr)} ({', '.join(f'{k}={v}' for k, v in Counter(nr).most_common(3))})")

    sp = run / "S1" / "qc_summary.json"
    if c.ok(sp.exists(), "S1/qc_summary.json exists"):
        s = json.loads(sp.read_text())
        c.info(f"qc_summary.json keys: {sorted(s)}")
        c.ok(any("threshold" in k for k in s), "qc_summary has thresholds")
        c.ok(any("hash" in k for k in s), "qc_summary has the policy hash")
    return c.done()


def cmd_qc_same(a) -> int:
    c = Check()
    da, db = load_decisions(Path(a.run_a)), load_decisions(Path(a.run_b))
    cols = ["cell_id", "sample_id", "decision", "reasons", "policy_hash"]
    da = da[cols].sort_values("cell_id").reset_index(drop=True)
    db = db[cols].sort_values("cell_id").reset_index(drop=True)
    c.ok(len(da) == len(db), f"same number of cells ({len(da)} / {len(db)})")
    if len(da) == len(db):
        diff = (da != db).any(axis=1)
        c.ok(not diff.any(), f"decisions, reasons and policy hash identical ({int(diff.sum())} rows differ)")
        if diff.any():
            print(pd.concat([da[diff].add_suffix("_a"), db[diff].add_suffix("_b")], axis=1).head(10).to_string())
    ma = read_tsv(Path(a.run_a) / "S1" / "qc_metrics.tsv.gz").set_index("cell_id").sort_index()
    mb = read_tsv(Path(a.run_b) / "S1" / "qc_metrics.tsv.gz").set_index("cell_id").sort_index()
    if "doublet_score" in ma and "doublet_score" in mb:
        dd = np.abs(ma["doublet_score"] - mb["doublet_score"]).max()
        c.info(f"doublet_score max abs diff between runs: {dd:.3g}")
    if "pre_cluster" in ma and "pre_cluster" in mb:
        c.info(f"pre_cluster ARI between runs: {ari(ma['pre_cluster'].astype(str), mb['pre_cluster'].astype(str)):.3f}")
    return c.done()


# ------------------------------------------------------------------ S2
R_EXPORT = r"""
suppressPackageStartupMessages(library(SeuratObject))
a <- commandArgs(TRUE)
o <- if (grepl("\\.qs2$", a[1])) qs2::qs_read(a[1]) else readRDS(a[1])
md <- o[[]]
md <- data.frame(cell_id = colnames(o), ident = as.character(Idents(o)),
                 lapply(md, function(x) if (is.factor(x)) as.character(x) else x),
                 check.names = FALSE, stringsAsFactors = FALSE)
write.table(md, a[2], sep = "\t", quote = FALSE, row.names = FALSE, na = "")
assay <- DefaultAssay(o)
info <- list(
  class = class(o)[1], n_cells = ncol(o), n_features = nrow(o), assay = assay,
  layers = Layers(o[[assay]]), reductions = Reductions(o), graphs = Graphs(o),
  n_variable_features = length(VariableFeatures(o)),
  reduction_dims = lapply(setNames(Reductions(o), Reductions(o)), function(r) ncol(Embeddings(o, r))),
  merges = length(o@misc$cellwarden_exchange_merges)
)
writeLines(jsonlite::toJSON(info, auto_unbox = TRUE, pretty = TRUE), a[3])
"""


def purity(clusters: pd.Series, labels: pd.Series) -> tuple[float, pd.DataFrame]:
    tab = pd.crosstab(clusters, labels)
    return float(tab.max(axis=1).sum() / tab.values.sum()), tab


def ari(x, y) -> float:
    x = pd.Series(list(x)).astype(str)
    y = pd.Series(list(y)).astype(str)
    tab = pd.crosstab(x, y).values.astype(float)
    comb = lambda v: v * (v - 1) / 2  # noqa: E731
    n = tab.sum()
    sij, sa, sb = comb(tab).sum(), comb(tab.sum(1)).sum(), comb(tab.sum(0)).sum()
    exp = sa * sb / comb(n)
    mx = (sa + sb) / 2
    return 1.0 if mx == exp else float((sij - exp) / (mx - exp))


def cmd_s2(a) -> int:
    c = Check()
    run = Path(a.run)
    obj = [p for p in (run / "S2" / "object.rds", run / "S2" / "object.qs2") if p.exists()]
    if not c.ok(bool(obj), "S2 object exists"):
        return c.done()
    sp = run / "S2" / "summary.json"
    summ = json.loads(sp.read_text()) if c.ok(sp.exists(), "S2/summary.json exists") else {}
    print("  summary.json:\n" + "\n".join("    " + ln for ln in json.dumps(summ, indent=2).splitlines()[:50]))
    dev = summ.get("device")
    if isinstance(dev, dict):
        dev = dev.get("device")
    c.ok(dev == a.device, f"summary device = {dev} (expected {a.device})")
    c.ok(any(k in summ for k in ("plan", "memory_plan")) or isinstance(summ.get("device"), dict),
         "summary records the memory plan")
    c.ok(any("reason" in k for k in summ) or isinstance(summ.get("device"), dict), "summary records why")

    with tempfile.TemporaryDirectory() as tmp:
        rfile = Path(tmp) / "export.R"
        rfile.write_text(R_EXPORT)
        info_path = Path(tmp) / "info.json"
        r = subprocess.run([RSCRIPT, str(rfile), str(obj[0]), a.out_tsv, str(info_path)],
                           capture_output=True, text=True)
        if not c.ok(r.returncode == 0, f"Rscript export of {obj[0].name} (exit {r.returncode})"):
            print(r.stdout[-2000:], r.stderr[-3000:])
            return c.done()
        info = json.loads(info_path.read_text())
    print("  object:", json.dumps(info))
    c.ok(info["class"] == "Seurat", "object is a Seurat object")
    c.ok("pca" in info["reductions"], f"has pca (reductions: {info['reductions']})")
    c.ok("umap" in info["reductions"], "has umap")
    c.ok(len(info["graphs"]) >= 1, f"has graphs ({info['graphs']})")
    c.ok("data" in info["layers"] and "counts" in info["layers"], f"has counts and data layers ({info['layers']})")
    pcs = summ.get("pcs", summ.get("n_pcs"))
    c.info(f"PCs: summary={pcs}, pca dims in object={info['reduction_dims'].get('pca')}")
    if a.device == "gpu":
        c.ok(info["merges"] >= 1, f"exchange merge recorded in misc$cellwarden_exchange_merges ({info['merges']})")

    md = pd.read_csv(a.out_tsv, sep="\t", keep_default_na=False, na_values=[""])
    clus_col = "seurat_clusters" if "seurat_clusters" in md.columns else "ident"
    md["cluster"] = md[clus_col].astype(str)
    md[["cell_id", "cluster"]].to_csv(a.out_tsv, sep="\t", index=False)   # slim export for ARI
    dec = load_decisions(run)
    expected = set(dec.loc[dec.decision.isin(["keep", "review"]), "cell_id"])   # review_action: keep
    got = set(md.cell_id)
    c.ok(got == expected, f"S2 holds exactly the keep + review cells ({len(got)} vs {len(expected)}; "
         f"extra {len(got - expected)}, missing {len(expected - got)})")
    c.ok(md["cluster"].ne("").all() and md["cluster"].ne("NA").all(), "every cell has a cluster")
    for k in ("cells_in", "n_cells_in", "cells_out", "n_cells_out"):
        if k in summ:
            c.info(f"summary {k} = {summ[k]}")

    truth = pd.read_csv(a.truth, sep="\t")
    x = md.merge(truth, on="cell_id", how="left")
    n_clusters = x["cluster"].nunique()
    c.info(f"{n_clusters} clusters ({clus_col}); summary says {summ.get('clusters', summ.get('n_clusters'))}")
    core = x[x.injected == "none"]
    p, tab = purity(core["cluster"], core["cell_type"])
    print("  clusters x planted type (non-injected kept cells):")
    print("    " + tab.to_string().replace("\n", "\n    "))
    c.ok(p >= THRESH["purity"], f"purity {p:.3f} (need >= {THRESH['purity']})")
    majority = set(tab.idxmax(axis=1))
    c.ok(majority >= set(core["cell_type"]), f"every planted type is the majority of a cluster ({sorted(majority)})")
    pl = x[x.injected == "platelet"]
    if len(pl):
        top = pl["cluster"].value_counts()
        c.info(f"kept platelets: {len(pl)}, in cluster(s) {dict(top.head(3))}")
    print(f"  ARI vs planted types (non-injected): {ari(core['cluster'], core['cell_type']):.3f}")
    return c.done()


def cmd_ari(a) -> int:
    c = Check()
    x = pd.read_csv(a.tsv_a, sep="\t", dtype=str).set_index("cell_id")["cluster"]
    y = pd.read_csv(a.tsv_b, sep="\t", dtype=str).set_index("cell_id")["cluster"]
    common = x.index.intersection(y.index)
    c.ok(len(common) == len(x) == len(y), f"same cells in both runs ({len(x)} / {len(y)}, common {len(common)})")
    v = ari(x[common], y[common])
    c.info(f"clusters: {x.nunique()} vs {y.nunique()}")
    print("    " + pd.crosstab(x[common], y[common]).to_string().replace("\n", "\n    "))
    c.ok(v >= THRESH["ari"], f"ARI CPU vs GPU path = {v:.3f} (need >= {THRESH['ari']})")
    return c.done()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("state"); p.add_argument("run"); p.set_defaults(f=cmd_state)
    p = sub.add_parser("sheets"); p.add_argument("sheets", nargs="+", metavar="SHEET:BASE"); p.set_defaults(f=cmd_sheets)
    p = sub.add_parser("ingest"); p.add_argument("run"); p.add_argument("inputs"); p.set_defaults(f=cmd_ingest)
    p = sub.add_parser("metrics"); p.add_argument("run"); p.add_argument("inputs"); p.set_defaults(f=cmd_metrics)
    p = sub.add_parser("qc-truth"); p.add_argument("run"); p.add_argument("truth"); p.set_defaults(f=cmd_qc_truth)
    p = sub.add_parser("qc-same"); p.add_argument("run_a"); p.add_argument("run_b"); p.set_defaults(f=cmd_qc_same)
    p = sub.add_parser("s2"); p.add_argument("run"); p.add_argument("truth"); p.add_argument("out_tsv")
    p.add_argument("--device", required=True, choices=["cpu", "gpu"]); p.set_defaults(f=cmd_s2)
    p = sub.add_parser("ari"); p.add_argument("tsv_a"); p.add_argument("tsv_b"); p.set_defaults(f=cmd_ari)
    p = sub.add_parser("snapshot"); p.add_argument("run"); p.add_argument("out"); p.set_defaults(f=cmd_snapshot)
    p = sub.add_parser("skipped"); p.add_argument("snapshot"); p.add_argument("run"); p.set_defaults(f=cmd_skipped)
    p = sub.add_parser("reran"); p.add_argument("snapshot"); p.add_argument("run")
    p.add_argument("--stages", default="S2"); p.set_defaults(f=cmd_reran)
    a = ap.parse_args()
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
