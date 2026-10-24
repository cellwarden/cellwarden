"""Unit tests for the rules-based QC engine (cellwarden.qc, docs/stages-spec.md 3.4)."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from cellwarden import qc

SPEC_POLICY = """
hard_limits:
  min_genes: 200
  max_mito_pct: 25
  min_umis: 400
adaptive:
  n_mads: 3
  metrics: [log_genes, log_umis, mito_pct]   # mito: upper bound only
  min_cells_for_mad: 100                      # fewer cells -> hard limits only
doublets:
  exclude_called: true
protected_populations:
  - name: Platelet
    markers: [PPBP, PF4, GNG11]
    min_fraction: 0.01
  - name: Erythrocyte
    markers: [HBB, HBA1, HBA2]
    min_fraction: 0.05
review_triggers:
  max_cells_removed_per_sample_pct: 30
  max_cluster_removed_pct: 50
"""


def policy(**over):
    d = yaml.safe_load(SPEC_POLICY)
    for k, v in over.items():
        sec, _, key = k.partition("__")
        if key:
            d[sec][key] = v
        else:
            d[sec] = v
    return qc.policy_from_dict(d)


def sample(sample_id="A", n=200, seed=0, n_clusters=4):
    """Well-behaved cells: ~1000 genes, ~3160 UMIs, ~5% mito; nothing should be removed."""
    rng = np.random.default_rng(seed)
    lg = rng.normal(3.0, 0.05, n)
    lu = lg + 0.5 + rng.normal(0, 0.02, n)
    return pd.DataFrame(
        {
            "cell_id": [f"{sample_id}_c{i:04d}" for i in range(n)],
            "sample_id": sample_id,
            "n_genes": np.round(10**lg).astype(int),
            "n_umis": np.round(10**lu).astype(int),
            "pct_mito": np.clip(rng.normal(5, 1, n), 0, None),
            "pct_ribo": 20.0,
            "pct_hb": 0.0,
            "log10_genes_per_umi": lg / lu,
            "protected_Platelet": 0.0,
            "protected_Erythrocyte": 0.0,
            "doublet_score": rng.uniform(0, 0.1, n),
            "doublet_call": False,
            "pre_cluster": [str(i % n_clusters) for i in range(n)],
        }
    )


def run_all(m, pol):
    thr = qc.compute_thresholds(m, pol)
    return thr, qc.decide(m, thr, pol)


def row(dec, cell):
    t = dec.table.set_index("cell_id")
    return t.loc[cell, "decision"], t.loc[cell, "reasons"]


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


def test_load_policy_spec_example(tmp_path):
    p = tmp_path / "qc_policy.yaml"
    p.write_text(SPEC_POLICY)
    pol = qc.load_policy(p)
    assert pol.min_genes == 200 and pol.min_umis == 400 and pol.max_mito_pct == 25
    assert pol.n_mads == 3 and pol.min_cells_for_mad == 100
    assert pol.adaptive_metrics == ("log_genes", "log_umis", "mito_pct")
    assert pol.exclude_doublets is True
    assert [p.name for p in pol.protected_populations] == ["Platelet", "Erythrocyte"]
    assert pol.protected_populations[1].markers == ("HBB", "HBA1", "HBA2")
    assert pol.max_cells_removed_per_sample_pct == 30 and pol.max_cluster_removed_pct == 50
    assert len(pol.hash) == 64
    # The spec example equals the built-in defaults.
    assert pol.hash == qc.policy_from_dict(None).hash


def test_policy_hash_canonical():
    d = yaml.safe_load(SPEC_POLICY)
    a = qc.policy_from_dict(d)
    # Key order, int vs float spelling and metric list order do not matter.
    d2 = {k: d[k] for k in reversed(list(d))}
    d2["hard_limits"] = {"min_umis": 400.0, "max_mito_pct": 25.0, "min_genes": 200}
    d2["adaptive"] = dict(d["adaptive"], metrics=["mito_pct", "log_umis", "log_genes"])
    assert qc.policy_from_dict(d2).hash == a.hash
    # A value change does.
    assert policy(hard_limits__min_genes=250).hash != a.hash
    assert policy(adaptive__n_mads=2.5).hash != a.hash


def test_policy_defaults_and_partial():
    pol = qc.policy_from_dict({"hard_limits": {"min_genes": 300}})
    assert pol.min_genes == 300 and pol.min_umis == 400 and pol.max_mito_pct == 25
    assert pol.n_mads == 3 and pol.max_cluster_removed_pct == 50
    # A list is replaced, not merged: omitted protected_populations means none (as S1a in R).
    assert pol.protected_populations == ()
    assert [p.name for p in qc.policy_from_dict(None).protected_populations] == ["Platelet", "Erythrocyte"]
    assert qc.policy_from_dict({}).hash == qc.policy_from_dict(None).hash
    pol = qc.policy_from_dict({"protected_populations": []})
    assert pol.protected_populations == ()


def test_policy_null_hard_limit_disables():
    pol = policy(hard_limits__min_umis=None)
    m = sample(n=50)  # < min_cells_for_mad: hard limits only
    m.loc[0, "n_umis"] = 10
    _, dec = run_all(m, pol)
    assert row(dec, "A_c0000") == ("keep", "")


def test_policy_reports_all_problems():
    bad = {
        "hard_limits": {"min_genes": -1, "max_mito_pct": "x", "min_umis": 400},
        "adaptive": {"n_mads": 0, "metrics": ["log_genes", "bogus"], "min_cells_for_mad": 0},
        "doublets": {"exclude_called": "yes"},
        "protected_populations": [{"name": "P", "markers": [], "min_fraction": 2}, {"markers": ["A"]}],
        "review_triggers": {"max_cells_removed_per_sample_pct": 130, "max_cluster_removed_pct": 50, "extra": 1},
        "typo_section": {},
    }
    with pytest.raises(qc.QCError) as e:
        qc.policy_from_dict(bad)
    probs = "\n".join(e.value.problems)
    for needle in ("min_genes", "max_mito_pct", "n_mads", "bogus", "min_cells_for_mad", "exclude_called",
                   "markers", "min_fraction", "name", "max_cells_removed_per_sample_pct", "review_triggers.extra",
                   "typo_section"):
        assert needle in probs, needle
    assert len(e.value.problems) >= 12


def test_policy_duplicate_population():
    with pytest.raises(qc.QCError, match="duplicated"):
        qc.policy_from_dict({"protected_populations": [{"name": "P", "markers": ["A"]}, {"name": "P", "markers": ["B"]}]})


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------


def test_thresholds_scaled_mad():
    m = sample()
    pol = policy()
    thr = qc.compute_thresholds(m, pol)
    t = thr.per_sample["A"]
    assert t.adaptive_applied and t.n_cells == 200
    lg = np.log10(m["n_genes"].to_numpy(float))
    med = np.median(lg)
    mad = 1.4826 * np.median(np.abs(lg - med))
    b = t.bounds["log_genes"]
    assert b.median == pytest.approx(med) and b.mad == pytest.approx(mad)
    assert b.lower == pytest.approx(med - 3 * mad) and b.upper == pytest.approx(med + 3 * mad)
    mt = t.bounds["mito_pct"]
    assert mt.lower is None and mt.upper == pytest.approx(
        np.median(m.pct_mito) + 3 * 1.4826 * np.median(np.abs(m.pct_mito - np.median(m.pct_mito)))
    )
    assert qc.MAD_SCALE == 1.4826
    d = thr.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["per_sample"]["A"]["adaptive_natural_scale"]["min_genes"] == pytest.approx(10 ** b.lower)


def test_thresholds_per_sample_and_small_sample():
    m = pd.concat([sample("A", 200, seed=1), sample("B", 99, seed=2)], ignore_index=True)
    m.loc[m.sample_id == "B", "n_genes"] *= 2
    thr = qc.compute_thresholds(m, policy())
    assert set(thr.per_sample) == {"A", "B"}
    assert thr.per_sample["A"].adaptive_applied
    assert not thr.per_sample["B"].adaptive_applied
    assert "min_cells_for_mad" in thr.per_sample["B"].note


def test_thresholds_zero_mad_not_applied():
    m = sample()
    m["pct_mito"] = 0.0
    thr = qc.compute_thresholds(m, policy())
    b = thr.per_sample["A"].bounds["mito_pct"]
    assert not b.applied and "MAD is 0" in b.note
    m.loc[0, "pct_mito"] = 10.0  # would be flagged by any MAD>0 rule; hard limit 25 still passes
    _, dec = run_all(m, policy())
    assert row(dec, "A_c0000") == ("keep", "")


def test_thresholds_only_listed_metrics():
    pol = policy(adaptive__metrics=["mito_pct"])
    m = sample()
    m.loc[0, "n_genes"] = 5000
    thr, dec = run_all(m, pol)
    assert set(thr.per_sample["A"].bounds) == {"mito_pct"}
    assert row(dec, "A_c0000") == ("keep", "")


def test_thresholds_roundtrip_from_dict():
    m = sample()
    m.loc[0, "pct_mito"] = 15
    pol = policy()
    thr = qc.compute_thresholds(m, pol)
    thr2 = qc.Thresholds.from_dict(json.loads(json.dumps(thr.to_dict())))
    pd.testing.assert_frame_equal(qc.decide(m, thr, pol).table, qc.decide(m, thr2, pol).table)


# --------------------------------------------------------------------------
# Rule 1: hard limits
# --------------------------------------------------------------------------


def test_hard_limits_each_code():
    m = sample(n=60)  # small sample: hard limits only
    m.loc[0, "n_genes"] = 150
    m.loc[1, "n_umis"] = 300
    m.loc[2, "pct_mito"] = 30
    m.loc[3, ["n_genes", "n_umis", "pct_mito"]] = [100, 200, 40]
    m.loc[4, "n_genes"] = 200  # boundary: not below 200
    m.loc[5, "pct_mito"] = 25  # boundary: not above 25
    m.loc[6, "n_umis"] = 400
    _, dec = run_all(m, policy())
    assert row(dec, "A_c0000") == ("exclude", "LOW_GENES")
    assert row(dec, "A_c0001") == ("exclude", "LOW_UMIS")
    assert row(dec, "A_c0002") == ("exclude", "HIGH_MT")
    assert row(dec, "A_c0003") == ("exclude", "LOW_GENES;LOW_UMIS;HIGH_MT")
    for c in ("A_c0004", "A_c0005", "A_c0006"):
        assert row(dec, c) == ("keep", "")


def test_hard_limits_missing_values_fail():
    m = sample(n=60)
    m["pct_mito"] = m["pct_mito"].astype(float)
    m.loc[0, "pct_mito"] = np.nan
    m["n_genes"] = m["n_genes"].astype(float)
    m.loc[1, "n_genes"] = np.nan
    _, dec = run_all(m, policy())
    assert row(dec, "A_c0000") == ("exclude", "HIGH_MT")
    assert row(dec, "A_c0001") == ("exclude", "LOW_GENES")


def test_adaptive_never_loosens_hard_limit():
    # A low-quality sample whose adaptive lower bound is below the hard limit.
    rng = np.random.default_rng(3)
    m = sample(n=200)
    lg = rng.normal(np.log10(260), 0.15, 200)
    m["n_genes"] = np.round(10**lg).astype(int)
    m["n_umis"] = m["n_genes"] * 3
    thr = qc.compute_thresholds(m, policy())
    assert 10 ** thr.per_sample["A"].bounds["log_genes"].lower < 200
    _, dec = run_all(m, policy())
    t = dec.table.set_index("cell_id")
    below = m.loc[m.n_genes < 200, "cell_id"]
    assert len(below) > 0
    assert (t.loc[below, "decision"] != "keep").all()
    assert t.loc[below, "reasons"].str.contains("LOW_GENES").all()


# --------------------------------------------------------------------------
# Rule 2: adaptive bounds
# --------------------------------------------------------------------------


def test_adaptive_each_code():
    m = sample()
    med_g = int(np.median(m.n_genes))
    med_u = int(np.median(m.n_umis))
    m.loc[0, "n_genes"] = 500          # > hard 200, far below median 1000
    m.loc[1, "n_genes"] = 3000
    m.loc[2, "n_umis"] = 1500          # > hard 400
    m.loc[3, "n_umis"] = 20000
    m.loc[4, "pct_mito"] = 15          # < hard 25
    m.loc[5, "pct_mito"] = 0           # mito: upper bound only
    _, dec = run_all(m, policy())
    assert row(dec, "A_c0000") == ("exclude", "ADAPT_LOW_GENES")
    assert row(dec, "A_c0001") == ("exclude", "ADAPT_HIGH_GENES")
    assert row(dec, "A_c0002") == ("exclude", "ADAPT_LOW_UMIS")
    assert row(dec, "A_c0003") == ("exclude", "ADAPT_HIGH_UMIS")
    assert row(dec, "A_c0004") == ("exclude", "ADAPT_HIGH_MT")
    assert row(dec, "A_c0005") == ("keep", "")
    assert med_g > 800 and med_u > 2500
    # Unmodified cells are kept.
    assert (dec.table.decision.iloc[6:] == "keep").all()


def test_adaptive_per_sample_bounds():
    # 1,500 genes is an outlier in A (median ~1000) but normal in B (median ~1600).
    a = sample("A", seed=1)
    b = sample("B", seed=2)
    b["n_genes"] = (b["n_genes"] * 1.6).round().astype(int)
    b["n_umis"] = (b["n_umis"] * 1.6).round().astype(int)
    a.loc[0, "n_genes"] = 1600
    b.loc[0, "n_genes"] = 1600
    _, dec = run_all(pd.concat([a, b], ignore_index=True), policy())
    assert row(dec, "A_c0000") == ("exclude", "ADAPT_HIGH_GENES")
    assert row(dec, "B_c0000") == ("keep", "")


def test_small_sample_hard_limits_only():
    m = pd.concat([sample("A", 200, seed=1), sample("S", 99, seed=2)], ignore_index=True)
    m.loc[m.cell_id == "S_c0000", "n_genes"] = 5000
    m.loc[m.cell_id == "S_c0001", "pct_mito"] = 20
    m.loc[m.cell_id == "S_c0002", "n_genes"] = 150
    m.loc[m.cell_id == "A_c0000", "n_genes"] = 5000
    _, dec = run_all(m, policy())
    assert row(dec, "S_c0000") == ("keep", "")
    assert row(dec, "S_c0001") == ("keep", "")
    assert row(dec, "S_c0002") == ("exclude", "LOW_GENES")
    assert row(dec, "A_c0000") == ("exclude", "ADAPT_HIGH_GENES")


def test_min_cells_for_mad_boundary():
    pol = policy()
    m = sample(n=100)
    m.loc[0, "n_genes"] = 5000
    _, dec = run_all(m, pol)  # exactly 100 cells: adaptive applies
    assert row(dec, "A_c0000") == ("exclude", "ADAPT_HIGH_GENES")


# --------------------------------------------------------------------------
# Rule 3: doublets
# --------------------------------------------------------------------------


@pytest.mark.parametrize("val", [True, "TRUE", "True", 1])
def test_doublet_excluded(val):
    m = sample()
    m["doublet_call"] = m["doublet_call"].astype(object)
    m.loc[7, "doublet_call"] = val
    _, dec = run_all(m, policy())
    assert row(dec, "A_c0007") == ("exclude", "DOUBLET")


def test_doublet_not_excluded_when_disabled():
    m = sample()
    m.loc[7, "doublet_call"] = True
    _, dec = run_all(m, policy(doublets={"exclude_called": False}))
    assert row(dec, "A_c0007") == ("keep", "")


def test_doublet_column_missing_or_na_is_not_called():
    m = sample().drop(columns=["doublet_call"])
    _, dec = run_all(m, policy())
    assert (dec.table.decision == "keep").all()
    m = sample()
    m["doublet_call"] = pd.Series([np.nan] * len(m), dtype=object)
    _, dec = run_all(m, policy())
    assert (dec.table.decision == "keep").all()


def test_doublet_bad_values():
    m = sample()
    m["doublet_call"] = "maybe"
    with pytest.raises(qc.QCError, match="non-boolean"):
        run_all(m, policy())


def test_reason_order_all_rules():
    m = sample()
    m.loc[0, ["n_genes", "pct_mito", "doublet_call"]] = [150, 40, True]
    _, dec = run_all(m, policy())
    d, r = row(dec, "A_c0000")
    assert d == "exclude"
    codes = r.split(";")
    assert codes == [c for c in qc.HARD_CODES + qc.ADAPT_CODES + ("DOUBLET",) if c in codes]
    assert codes[:2] == ["LOW_GENES", "HIGH_MT"] and codes[-1] == "DOUBLET"
    assert "ADAPT_LOW_GENES" in codes and "ADAPT_HIGH_MT" in codes


# --------------------------------------------------------------------------
# Rule 4: protected populations
# --------------------------------------------------------------------------


def test_protected_adaptive_only_becomes_review():
    m = sample()
    m.loc[0, ["pct_mito", "protected_Platelet"]] = [15, 0.02]       # adaptive only + protected
    m.loc[1, ["n_genes", "protected_Platelet"]] = [150, 0.5]        # hard limit: stays excluded
    m.loc[2, ["pct_mito", "doublet_call", "protected_Platelet"]] = [15, True, 0.5]  # doublet: stays excluded
    m.loc[3, ["pct_mito", "protected_Platelet"]] = [15, 0.005]      # below min_fraction
    m.loc[4, ["pct_mito", "protected_Erythrocyte"]] = [15, 0.05]    # exactly min_fraction (>=)
    m.loc[5, ["pct_mito", "protected_Erythrocyte"]] = [15, 0.049]
    m.loc[6, ["pct_mito", "protected_Platelet", "protected_Erythrocyte"]] = [15, 0.1, 0.1]
    m.loc[7, "protected_Platelet"] = 0.5                            # protected, passes everything
    thr, dec = run_all(m, policy())
    assert row(dec, "A_c0000") == ("review", "ADAPT_HIGH_MT;PROTECTED_POP:Platelet")
    assert row(dec, "A_c0001")[0] == "exclude" and "PROTECTED" not in row(dec, "A_c0001")[1]
    assert row(dec, "A_c0002") == ("exclude", "ADAPT_HIGH_MT;DOUBLET")
    assert row(dec, "A_c0003") == ("exclude", "ADAPT_HIGH_MT")
    assert row(dec, "A_c0004") == ("review", "ADAPT_HIGH_MT;PROTECTED_POP:Erythrocyte")
    assert row(dec, "A_c0005") == ("exclude", "ADAPT_HIGH_MT")
    assert row(dec, "A_c0006") == ("review", "ADAPT_HIGH_MT;PROTECTED_POP:Platelet;PROTECTED_POP:Erythrocyte")
    assert row(dec, "A_c0007") == ("keep", "")

    losses = {p.name: p for p in qc.check_protected(m, thr, policy())}
    pl = losses["Platelet"]
    assert pl.n_protected == 5  # rows 0,1,2,6,7
    assert pl.n_flagged == 4 and pl.n_rescued == 2 and pl.n_lost == 2
    assert pl.would_be_lost is False
    assert pl.per_sample["A"]["n_lost"] == 2
    er = losses["Erythrocyte"]
    assert er.n_protected == 2 and er.n_rescued == 2 and er.n_lost == 0
    assert er.would_be_lost is True  # every protected cell flagged by rules 1-3
    json.dumps([p.to_dict() for p in losses.values()], allow_nan=False)


def test_protected_missing_column_is_error():
    m = sample().drop(columns=["protected_Erythrocyte"])
    with pytest.raises(qc.QCError, match="protected_Erythrocyte"):
        qc.compute_thresholds(m, policy())


# --------------------------------------------------------------------------
# Rule 5: review triggers
# --------------------------------------------------------------------------


def _lossy_sample():
    """Sample B: 70 hard failures, 10 adaptive-only failures, 130 good cells (80/210 = 38% lost)."""
    b = sample("B", n=210, seed=5)
    b.loc[:69, "n_genes"] = 100
    b.loc[70:79, "pct_mito"] = 15
    return b


def test_sample_loss_trigger():
    a = sample("A", seed=4)
    a.loc[0, "pct_mito"] = 15
    m = pd.concat([a, _lossy_sample()], ignore_index=True)
    pol = policy()
    thr = qc.compute_thresholds(m, pol)
    sim = qc.simulate_filter(m, thr, pol)
    assert sim.sample_loss_triggered == ["B"]
    assert sim.per_sample.loc["B", "n_excluded"] == 80
    assert sim.per_sample.loc["B", "n_fail_hard"] == 70
    assert sim.per_sample.loc["B", "n_fail_adaptive_only"] == 10
    assert sim.per_sample.loc["B", "pct_excluded"] == pytest.approx(100 * 80 / 210)
    dec = qc.decide(m, thr, pol)
    t = dec.table.set_index("cell_id")
    assert (t.loc[[f"B_c{i:04d}" for i in range(70)], "decision"] == "exclude").all()
    assert row(dec, "B_c0070") == ("review", "ADAPT_HIGH_MT;SAMPLE_LOSS")
    assert (t.loc[[f"B_c{i:04d}" for i in range(70, 80)], "decision"] == "review").all()
    assert row(dec, "A_c0000") == ("exclude", "ADAPT_HIGH_MT")  # other sample unaffected
    trig = [x for x in dec.triggered if x["rule"] == "SAMPLE_LOSS"]
    assert trig == [{"rule": "SAMPLE_LOSS", "sample_id": "B", "pct_excluded": pytest.approx(100 * 80 / 210),
                     "limit_pct": 30.0, "n_cells_to_review": 10}]


def test_sample_loss_threshold_is_strict():
    b = _lossy_sample()
    pol = policy(review_triggers={"max_cells_removed_per_sample_pct": 100 * 80 / 210, "max_cluster_removed_pct": 100})
    _, dec = run_all(b, pol)
    assert dec.triggered == []
    assert row(dec, "B_c0070") == ("exclude", "ADAPT_HIGH_MT")


def test_cluster_loss_trigger():
    m = sample("A", n=200, seed=6)  # 4 clusters x 50
    c3 = m.index[m.pre_cluster == "3"]
    m.loc[c3[:30], "pct_mito"] = 15  # 30/50 = 60% of cluster 3; 15% of the sample
    m.loc[c3[30], "n_genes"] = 150   # hard failure in the same cluster stays excluded
    pol = policy()
    thr = qc.compute_thresholds(m, pol)
    sim = qc.simulate_filter(m, thr, pol)
    assert sim.sample_loss_triggered == []
    assert sim.cluster_loss_triggered == ["3"]
    assert sim.per_cluster.loc["3", "n_excluded"] == 31
    assert list(sim.per_cluster.index) == ["0", "1", "2", "3"]
    dec = qc.decide(m, thr, pol)
    first = m.loc[c3[0], "cell_id"]
    assert row(dec, first) == ("review", "ADAPT_HIGH_MT;CLUSTER_LOSS")
    assert dec.table.set_index("cell_id").loc[m.loc[c3[:30], "cell_id"], "decision"].eq("review").all()
    assert row(dec, m.loc[c3[30], "cell_id"])[0] == "exclude"
    trig = dec.triggered
    assert len(trig) == 1 and trig[0]["rule"] == "CLUSTER_LOSS" and trig[0]["pre_cluster"] == "3"
    assert trig[0]["n_cells_to_review"] == 30


def test_triggers_use_losses_after_protection():
    # 30 adaptive-only failures in cluster 3, 10 of them protected: after rule 4
    # only 20/50 = 40% are lost, so CLUSTER_LOSS does not fire.
    m = sample("A", n=200, seed=6)
    c3 = m.index[m.pre_cluster == "3"]
    m.loc[c3[:30], "pct_mito"] = 15
    m.loc[c3[:10], "protected_Platelet"] = 0.2
    _, dec = run_all(m, policy())
    t = dec.table.set_index("cell_id")
    assert dec.triggered == []
    assert (t.loc[m.loc[c3[:10], "cell_id"], "reasons"] == "ADAPT_HIGH_MT;PROTECTED_POP:Platelet").all()
    assert (t.loc[m.loc[c3[10:30], "cell_id"], "decision"] == "exclude").all()


def test_both_triggers_and_protected_codes_combine():
    b = _lossy_sample()
    b["pre_cluster"] = "9"
    b.loc[60:89, "pre_cluster"] = "7"   # cluster 7: 10 hard + 10 adaptive-only of 30 = 67%
    b.loc[71, "protected_Platelet"] = 0.2
    _, dec = run_all(b, policy())
    rules = sorted(t["rule"] for t in dec.triggered)
    assert rules == ["CLUSTER_LOSS", "SAMPLE_LOSS"]
    assert row(dec, "B_c0070") == ("review", "ADAPT_HIGH_MT;SAMPLE_LOSS;CLUSTER_LOSS")
    assert row(dec, "B_c0071") == ("review", "ADAPT_HIGH_MT;PROTECTED_POP:Platelet;SAMPLE_LOSS;CLUSTER_LOSS")
    assert row(dec, "B_c0060")[0] == "exclude"


def test_no_pre_cluster_column():
    m = sample().drop(columns=["pre_cluster"])
    m.loc[0, "pct_mito"] = 15
    thr, dec = run_all(m, policy())
    assert qc.simulate_filter(m, thr, policy()).per_cluster is None
    assert row(dec, "A_c0000") == ("exclude", "ADAPT_HIGH_MT")


# --------------------------------------------------------------------------
# Output shape, summary and determinism
# --------------------------------------------------------------------------


def test_decisions_table_and_summary():
    m = pd.concat([sample("A", seed=1), _lossy_sample()], ignore_index=True)
    pol = policy()
    thr, dec = run_all(m, pol)
    t = dec.table
    assert list(t.columns) == ["cell_id", "sample_id", "decision", "reasons", "policy_hash", "decided_by"]
    assert list(t.cell_id) == list(m.cell_id)
    assert set(t.decision) <= {"keep", "exclude", "review"}
    assert (t.loc[t.decision == "keep", "reasons"] == "").all()
    assert (t.loc[t.decision != "keep", "reasons"] != "").all()
    assert (t.policy_hash == pol.hash).all() and (t.decided_by == "rules").all()
    s = dec.summary()
    json.dumps(s, allow_nan=False)
    assert s["policy_hash"] == pol.hash
    assert s["counts"]["keep"] + s["counts"]["exclude"] + s["counts"]["review"] == len(m)
    assert s["counts_per_sample"]["B"] == {"keep": 130, "exclude": 70, "review": 10}
    assert s["counts_per_reason"]["SAMPLE_LOSS"] == 10
    assert s["counts_per_reason_per_sample"]["B"]["LOW_GENES"] == 70
    assert set(s["thresholds"]["per_sample"]) == {"A", "B"}
    assert s["triggered_review_rules"][0]["rule"] == "SAMPLE_LOSS"
    assert [p["name"] for p in s["protected_populations"]] == ["Platelet", "Erythrocyte"]


def test_determinism_and_row_order_independence():
    m = pd.concat([sample("A", seed=1), _lossy_sample(), sample("C", 80, seed=9)], ignore_index=True)
    m.loc[3, "pct_mito"] = 15
    pol = policy()
    _, d1 = run_all(m, pol)
    _, d2 = run_all(m.copy(), qc.policy_from_dict(yaml.safe_load(SPEC_POLICY)))
    pd.testing.assert_frame_equal(d1.table, d2.table)
    assert json.dumps(d1.summary(), sort_keys=True) == json.dumps(d2.summary(), sort_keys=True)
    shuffled = m.sample(frac=1, random_state=0).reset_index(drop=True)
    _, d3 = run_all(shuffled, pol)
    a = d1.table.set_index("cell_id").sort_index()
    b = d3.table.set_index("cell_id").sort_index()
    pd.testing.assert_frame_equal(a, b)
    assert list(d3.table.cell_id) == list(shuffled.cell_id)


def test_duplicate_cells_rejected():
    m = sample()
    m.loc[1, "cell_id"] = m.loc[0, "cell_id"]
    with pytest.raises(qc.QCError, match="not unique"):
        qc.compute_thresholds(m, policy())


def test_read_metrics_roundtrip(tmp_path):
    m = sample()
    m["doublet_call"] = np.where(np.arange(len(m)) == 0, "TRUE", "FALSE")
    m.loc[1, "doublet_score"] = np.nan
    p = tmp_path / "qc_metrics.tsv.gz"
    m.to_csv(p, sep="\t", index=False, na_rep="NA")
    r = qc.read_metrics(p)
    assert pd.api.types.is_string_dtype(r.pre_cluster) and pd.api.types.is_string_dtype(r.cell_id)
    assert r.loc[0, "pre_cluster"] == "0"
    _, dec = run_all(r, policy())
    assert row(dec, "A_c0000") == ("exclude", "DOUBLET")
