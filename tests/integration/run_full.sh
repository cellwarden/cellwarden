#!/usr/bin/env bash
# End-to-end test through S5 (docs/agents-spec.md section 9).
#
#   tests/integration/run_full.sh [WORKDIR]
#
# Extends run_stages.sh: one `cellwarden run` on the two-sample fixture with
# `llm.default.provider: mock` and `opentargets.offline: true`, CPU path, from
# S0 through S5. S0-S2 run first (same checks as run_stages.sh); then every
# later stage of the runner's registry is run with `cellwarden resume --until`
# and checked:
#
#   S3      markers, cluster summary, signatures, evidence bundle, consensus
#   G1-G4   the gate's bundle/proposal/review/decision.json and report.html
#           (a gate that stops for a human, exit 3, is approved with
#           `cellwarden approve --by run_full` and the run resumed)
#   S4      dge_summary.json, opentargets_summary.json (offline; no contrasts
#           are configured because both fixture samples are controls)
#   S5      S5/umap.tsv.gz (one row per S2 cell), report.html parses and holds
#           every section, explorer.html holds the embedded JSON and a canvas
#   rerun   the same config again skips every stage
#
# A stage that is not registered in the runner yet (for example the gates
# before the coordinator wires them) is reported as SKIP, so this script
# passes on a partial tree. Every step runs even if an earlier one fails;
# steps that depend on a failed one are SKIP. Exit code: non-zero when any
# step FAILED (SKIP alone does not fail the script, unlike run_stages.sh).
#
# Environment: PYTHON, RSCRIPT, KEEP_WORKDIR=1; thresholds as in check_stages.py.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export RSCRIPT="${RSCRIPT:-Rscript}"
export CELLWARDEN_DOCTOR_OFFLINE="${CELLWARDEN_DOCTOR_OFFLINE:-1}"
INPUTS="$REPO/tests/fixtures/stage_inputs"
CHECK=("$PYTHON" "$HERE/check_stages.py")

if [[ $# -ge 1 ]]; then
  WORK="$1"; mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/cellwarden-full.XXXXXX")"
  CLEANUP_ON_PASS=1
fi
export PYTHONPATH="$REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$WORK/.numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$WORK/.mpl}"
export HOME_LEDGER="$WORK/ledger_total.json"
LOG="$WORK/logs"; mkdir -p "$LOG"
RUN="$WORK/run_full"

# Config: cpu.yaml from run_stages.sh plus the mock LLM, offline Open Targets and auto gates.
SRC_CONF="$HERE/stage_configs"
CONF="$WORK/stage_configs"
mkdir -p "$CONF"
cp "$SRC_CONF"/qc_policy.yaml "$CONF"/
"$PYTHON" - "$INPUTS/samples.csv" "$INPUTS" "$CONF" > "$CONF/samples.csv" <<'PY'
import csv, os, sys
sheet, inputs, conf = sys.argv[1:4]
rows = list(csv.DictReader(open(sheet, newline="")))
w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]), lineterminator="\n")
w.writeheader()
for r in rows:
    r["path"] = os.path.relpath(os.path.join(inputs, r["path"]), conf)
    w.writerow(r)
PY
"$PYTHON" - "$SRC_CONF/cpu.yaml" "$CONF/full.yaml" "$HOME_LEDGER" <<'PY'
import os, sys, yaml
src, dst, ledger = sys.argv[1:4]
cfg = yaml.safe_load(open(src))
cfg["project"] = "full_test"
cfg["engine"] = os.environ.get("CELLWARDEN_TEST_ENGINE", "auto")   # auto: Rust path when scfastR is installed
cfg["llm"] = {"default": {"provider": "mock", "model": "mock-model"},
              "budget": {"per_run_limit": 5, "total_limit": 30, "ledger_path": ledger}, "replay": False}
cfg["opentargets"] = {"enabled": True, "offline": True, "cache_dir": "refs/opentargets"}
cfg["gates"] = {"G1": "auto", "G2": "auto", "G3": "auto", "G4": "auto"}
cfg["report"] = {"explorer_max_cells": 100000, "expr_genes": 50, "expr_max_cells": 1000, "top_markers": 5}
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY

echo "cellwarden full pipeline S0-S5 end-to-end (mock LLM, offline Open Targets)"
echo "  repo:    $REPO"
echo "  workdir: $WORK"
echo

declare -a NAMES STATUS TIMES
declare -A RESULT
T_ALL=$(date +%s.%N)
record() { NAMES+=("$1  $2"); STATUS+=("$3"); TIMES+=("$4"); RESULT[$1]="$3"; }
run_step() {   # run_step ID "description" DEPENDS_ON -- cmd...
  local id="$1" desc="$2" deps="$3"; shift 4
  local d
  for d in $deps; do
    if [[ "${RESULT[$d]:-}" != "PASS" ]]; then
      record "$id" "$desc" "SKIP" "-"; echo "=== [$id] $desc: SKIP (needs $d)"; echo; return
    fi
  done
  echo "=== [$id] $desc"
  local t0 t1 rc
  t0=$(date +%s.%N)
  "$@" 2>&1 | tee "$LOG/$id.log"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s.%N)
  local st="PASS"
  if [[ $rc -eq 99 ]]; then st="SKIP"; elif [[ $rc -ne 0 ]]; then st="FAIL"; fi
  record "$id" "$desc" "$st" "$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1fs", b-a}')"
  echo "--- [$id] $st (exit $rc)"; echo
}

cellwarden() { "$PYTHON" -m cellwarden "$@"; }

# The runner's stage order after S2 (whatever is registered today).
registered() { "$PYTHON" -c 'from cellwarden import runner; print(" ".join(s.id for s in runner.STAGES))'; }
is_registered() { local s; for s in $(registered); do [[ "$s" == "$1" ]] && return 0; done; return 1; }

show_logs() {
  local f
  for f in "$RUN"/logs/*.log; do [[ -e "$f" ]] || continue; echo "----- tail of $f"; tail -n 30 "$f"; done
  [[ -f "$RUN/state.json" ]] && { echo "----- state.json"; cat "$RUN/state.json"; }
}

first_run() {   # S0-S2 (G1 sits before S2: a gate that stops for a human is approved and the run resumed)
  cellwarden run --config "$CONF/full.yaml" --run-dir "$RUN" --until S2
  local rc=$?
  [[ $rc -eq 3 ]] && { resume_until S2; return $?; }
  [[ $rc -eq 0 ]] || { echo "cellwarden run exited $rc"; show_logs; }
  return $rc
}

# Resume through STAGE. Exit 3 = a gate waits for a human: approve it as
# "run_full" and continue (at most 4 gates). Exit 99 = STAGE not registered.
resume_until() {
  local stage="$1" i rc gate
  is_registered "$stage" || { echo "$stage is not registered in the runner yet: SKIP"; return 99; }
  for i in 1 2 3 4 5; do
    cellwarden resume --run-dir "$RUN" --until "$stage" | tee "$WORK/resume_$stage.out"
    rc=${PIPESTATUS[0]}
    [[ $rc -eq 0 ]] && return 0
    if [[ $rc -eq 3 ]]; then
      gate="$(grep -oE -- '--stage [a-z-]+-review|--stage G[0-9]' "$WORK/resume_$stage.out" | head -1 | awk '{print $2}')"
      [[ -n "$gate" ]] || { echo "exit 3 but no '--stage <gate>' hint in the output"; show_logs; return 1; }
      echo ">>> gate $gate waits for a human: cellwarden approve --by run_full"
      cellwarden approve --run-dir "$RUN" --stage "$gate" --by run_full || { echo "cellwarden approve failed"; show_logs; return 1; }
      continue
    fi
    echo "cellwarden resume exited $rc"; show_logs; return $rc
  done
  echo "still not through $stage after 5 resumes"; return 1
}

check_s3() {
  local f miss=0
  for f in markers.tsv.gz cluster_summary.json signatures.tsv.gz evidence_bundle.json annotation_consensus.tsv; do
    [[ -s "$RUN/S3/$f" ]] || { echo "missing S3/$f"; miss=1; }
  done
  [[ $miss -eq 0 ]] || return 1
  "$PYTHON" - "$RUN" <<'PY'
import gzip, json, sys, csv
from pathlib import Path
rd = Path(sys.argv[1])
cs = json.load(open(rd / "S3" / "cluster_summary.json"))
s2 = json.load(open(rd / "S2" / "summary.json"))
n = sum(int(v.get("n_cells", 0)) for v in cs.values()) if isinstance(cs, dict) else None
print(f"cluster_summary: {len(cs)} clusters, {n} cells")
rows = list(csv.DictReader(gzip.open(rd / "S3" / "markers.tsv.gz", "rt"), delimiter="\t"))
need = {"cluster", "gene", "avg_log2fc", "pct_in", "pct_out", "p_adj", "rank"}
assert rows and need <= set(rows[0]), f"markers columns {list(rows[0]) if rows else None}"
print(f"markers: {len(rows)} rows, {len({r['cluster'] for r in rows})} clusters")
b = json.load(open(rd / "S3" / "evidence_bundle.json"))
txt = json.dumps(b)
assert "cell_id" not in txt, "evidence bundle must not hold per-cell data"
ann = list(csv.DictReader(open(rd / "S3" / "annotation_consensus.tsv"), delimiter="\t"))
assert ann and {"cluster", "label_broad", "confidence", "status"} <= set(ann[0]), list(ann[0]) if ann else None
print("consensus:", ", ".join(f"{r['cluster']}={r['label_broad']}({r['status']})" for r in ann))
PY
}

check_gate() {   # the four protocol files + report.html
  local g="$1" f miss=0
  is_registered "$g" || { echo "$g not registered: SKIP"; return 99; }
  for f in bundle.json proposal.json review.json decision.json report.html; do
    [[ -s "$RUN/gates/$g/$f" ]] || { echo "missing gates/$g/$f"; miss=1; }
  done
  [[ $miss -eq 0 ]] || return 1
  "$PYTHON" - "$RUN/gates/$g" "$g" <<'PY'
import json, sys
from pathlib import Path
d, g = Path(sys.argv[1]), sys.argv[2]
p = json.load(open(d / "proposal.json")); r = json.load(open(d / "review.json")); dec = json.load(open(d / "decision.json"))
assert p.get("gate") == g and "proposal" in p and "policy_check" in p, "proposal.json shape"
assert r.get("verdict") in ("approve", "request_changes", "escalate"), r.get("verdict")
assert dec.get("gate") == g and dec.get("mode") in ("auto", "review", "escalate") and dec.get("decided_by"), "decision.json shape"
print(f"{g}: mode {dec['mode']}, decided by {dec['decided_by']}, reviewer {r['verdict']}, fallback {p.get('fallback')}")
PY
}

check_s4() {
  [[ -s "$RUN/S4/dge_summary.json" ]] || { echo "missing S4/dge_summary.json"; return 1; }
  [[ -s "$RUN/S4/opentargets_summary.json" ]] || { echo "missing S4/opentargets_summary.json"; return 1; }
  "$PYTHON" - "$RUN" <<'PY'
import json, sys
from pathlib import Path
rd = Path(sys.argv[1])
d = json.load(open(rd / "S4" / "dge_summary.json"))
o = json.load(open(rd / "S4" / "opentargets_summary.json"))
print(f"dge_summary: {len(d.get('contrasts') or [])} contrast(s) (none configured: both fixture samples are controls)")
for c in d.get("contrasts") or []:
    for g in c.get("groups") or []:
        f = g.get("file")
        if f:
            assert (rd / f).exists() or Path(f).exists(), f"DGE table {f} missing"
assert o.get("offline") in (True, None) or o.get("mode") == "offline", "Open Targets must have run offline"
print("opentargets_summary keys:", sorted(o)[:8])
PY
}

check_s5() {
  local f
  for f in umap.tsv.gz report.html explorer.html export.json; do
    [[ -s "$RUN/S5/$f" ]] || { echo "missing S5/$f"; return 1; }
  done
  "$PYTHON" - "$RUN" <<'PY'
import gzip, json, re, sys
from html.parser import HTMLParser
from pathlib import Path
rd = Path(sys.argv[1])
s2 = json.load(open(rd / "S2" / "summary.json"))
rows = gzip.open(rd / "S5" / "umap.tsv.gz", "rt").read().splitlines()
hdr = rows[0].split("\t")
need = {"cell_id", "umap_1", "umap_2", "seurat_clusters", "sample_id"}
assert need <= set(hdr), hdr
n = len(rows) - 1
cells_out = s2.get("cells_out") or s2.get("n_cells_out") or (s2.get("cells") or {}).get("out")
print(f"umap.tsv.gz: {n} cells, columns {hdr}; S2 cells out: {cells_out}")
if cells_out is not None:
    assert n == int(cells_out), f"umap rows {n} != S2 cells {cells_out}"
assert "cell_type_broad" in hdr, "cell_type_broad column missing (S3 consensus or G3 annotation should be joined)"

class P(HTMLParser):
    VOID = {"meta", "link", "img", "br", "hr", "input", "rect", "line", "circle", "path", "source", "wbr"}
    def __init__(self):
        super().__init__(); self.ids = set(); self.ext = []; self.stack = []; self.err = []
    def handle_starttag(self, t, a):
        a = dict(a)
        if a.get("id"): self.ids.add(a["id"])
        for k in ("src", "href"):
            if (a.get(k) or "").startswith(("http:", "https:", "//")): self.ext.append(a[k])
        if t not in self.VOID: self.stack.append(t)
    def handle_startendtag(self, t, a):
        self.handle_starttag(t, a)
        if t not in self.VOID: self.stack.pop()
    def handle_endtag(self, t):
        if t in self.VOID: return
        if not self.stack or self.stack[-1] != t: self.err.append(t); return
        self.stack.pop()

html = (rd / "S5" / "report.html").read_text()
p = P(); p.feed(html); p.close()
assert not p.err and not p.stack, f"report.html does not parse cleanly: {p.err[:3]} {p.stack[:3]}"
assert not p.ext, f"report.html references external resources: {p.ext[:3]}"
for sec in ("summary", "resources", "qc", "clusters", "markers", "dge", "opentargets", "gates", "llm", "software"):
    assert sec in p.ids, f"report.html lacks section #{sec}"
assert "UMAP by cluster" in html and "<svg" in html, "no UMAP / plots in report.html"
assert "donor_A" in html and "donor_B" in html, "sample tables missing"
not_run = re.findall(r"([A-Za-z0-9 /]+): not run", html)
print(f"report.html: {len(html)//1024} kB, sections {sorted(p.ids & {'summary','qc','clusters','markers','dge','opentargets','gates','llm'})}, not run: {not_run}")
ex = (rd / "S5" / "explorer.html").read_text()
m = re.search(r'<script id="data" type="application/json">(.*?)</script>', ex, re.S)
assert m and "<canvas" in ex, "explorer.html lacks the data block or canvas"
data = json.loads(m.group(1))
assert data["n"] == n == data["n_total"] and len(data["x"]) == n, (data["n"], n)
assert "seurat_clusters" in data["cats"], list(data["cats"])
print(f"explorer.html: {len(ex)//1024} kB, {data['n']} cells, colour options {sorted(data['cats'])}, {len(data['genes'])} genes")
PY
}

rerun_same() {
  "${CHECK[@]}" snapshot "$RUN" "$WORK/state_before_rerun.json" || return 1
  cellwarden run --config "$CONF/full.yaml" --run-dir "$RUN" || { echo "rerun exited $?"; return 1; }
  "${CHECK[@]}" skipped "$WORK/state_before_rerun.json" "$RUN"
}

status_check() {
  local out rc
  out="$(cellwarden status --run-dir "$RUN" 2>&1)"; rc=$?
  echo "$out"
  [[ $rc -eq 0 ]] || return 1
  grep -q "S5" <<<"$out" || { echo "status does not mention S5"; return 1; }
}

run_step T0 "stage inputs reproducible"               ""   -- "$PYTHON" "$REPO/tests/fixtures/make_stage_inputs.py" --check
run_step T1 "cellwarden doctor (mock LLM, offline)"       ""   -- cellwarden doctor --config "$CONF/full.yaml"
run_step F1 "run S0-S2 (device cpu)"                  "T0" -- first_run
run_step F2 "state.json, logs, config copies"         "F1" -- "${CHECK[@]}" state "$RUN"
run_step F3 "QC decisions vs injected truth"          "F1" -- "${CHECK[@]}" qc-truth "$RUN" "$INPUTS/truth.tsv"
run_step F4 "S2 object, summary, purity"              "F1" -- "${CHECK[@]}" s2 "$RUN" "$INPUTS/truth.tsv" "$WORK/clusters.tsv" --device cpu
run_step G1 "gate G1 (QC agent, mock)"                "F1" -- resume_until G1
run_step G1c "G1: four files + report.html"           "G1" -- check_gate G1
run_step S3 "stage S3 (evidence, CellTypist, consensus)" "F1" -- resume_until S3
run_step S3c "S3 outputs"                             "S3" -- check_s3
run_step G2 "gate G2 (context agent, mock)"           "S3" -- resume_until G2
run_step G2c "G2: four files + report.html"           "G2" -- check_gate G2
run_step G3 "gate G3 (annotation agent, mock)"        "S3" -- resume_until G3
run_step G3c "G3: four files + report.html"           "G3" -- check_gate G3
run_step S4 "stage S4 (pseudobulk DGE, Open Targets offline)" "S3" -- resume_until S4
run_step S4c "S4 outputs"                             "S4" -- check_s4
run_step G4 "gate G4 (interpretation agent, mock)"    "S4" -- resume_until G4
run_step G4c "G4: four files + report.html"           "G4" -- check_gate G4
run_step S5 "stage S5 (export, report, explorer)"     "F1" -- resume_until S5
run_step S5c "S5: umap export, report.html, explorer.html" "S5" -- check_s5
run_step R0 "cellwarden status lists S5"                  "S5" -- status_check
run_step R1 "rerun same config skips every stage"     "S5" -- rerun_same

T_END=$(date +%s.%N)
echo "================= full pipeline summary ================="
printf "%-6s %-8s %s\n" "RESULT" "TIME" "STEP"
fail=0
for i in "${!NAMES[@]}"; do
  printf "%-6s %-8s %s\n" "${STATUS[$i]}" "${TIMES[$i]}" "${NAMES[$i]}"
  [[ "${STATUS[$i]}" == "FAIL" ]] && fail=1
done
printf "%-6s %-8s %s\n" "" "$(awk -v a="$T_ALL" -v b="$T_END" 'BEGIN{printf "%.1fs", b-a}')" "total"
echo "=========================================================="
if [[ $fail -ne 0 ]]; then
  echo "FULL PIPELINE TESTS FAILED (workdir kept: $WORK, logs: $LOG)"; exit 1
fi
echo "FULL PIPELINE TESTS PASSED (SKIP = stage not registered yet)"
if [[ "${CLEANUP_ON_PASS:-0}" == "1" && "${KEEP_WORKDIR:-0}" != "1" ]]; then rm -rf "$WORK"; else echo "workdir kept: $WORK"; fi
