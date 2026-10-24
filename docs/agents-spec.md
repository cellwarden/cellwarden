# CellWarden build steps 3–7 — interface spec v1.0

Contracts for the work that follows stages S0–S2 (`docs/stages-spec.md`). Several developers build these in parallel, so **file ownership and the JSON/TSV contracts below are fixed**; anything not stated here is the developer's choice, to be documented in the module header. DESIGN.md section 4 (agents) and section 6 (DGE, Open Targets) are the background.

Principle, unchanged: **rules decide, agents advise, reviewers check, humans approve, everything is logged.** No agent output ever changes data directly; it becomes a *proposal* that code validates against policy.

## 1. Run directory additions

```
<run-dir>/
  S3/  markers.tsv.gz  cluster_summary.json  signatures.tsv.gz  celltypist.tsv.gz (optional)
       evidence_bundle.json
  S4/  pseudobulk/<contrast>/counts.tsv.gz  samples.tsv
       dge/<contrast>__<group>.tsv.gz  dge_summary.json
       opentargets/cache/<gene>.json  opentargets_summary.json
  S5/  report.html  explorer.html
  gates/G1 .. G4/   bundle.json  proposal.json  review.json  decision.json  report.html
  llm/  ledger.json  agent_log.jsonl
```

Stage order in the runner: `S0 S1a S1b S1c G1 S2 S3 G2 G3 S4 G4 S5`. Gates are runner stages like any other (fingerprinted, resumable). A gate that needs a human exits the run with code 3 and prints `cellwarden approve --run-dir D --stage G1`.

## 2. Gate protocol (`python/cellwarden/gates/`)

```
bundle.json      evidence for the gate; aggregated tables only, never per-cell counts or free-text metadata
proposal.json    {"gate","worker":{"model","role"},"proposal":{...gate-specific...},"rationale":str,
                  "policy_check":{"ok":bool,"violations":[str]},"fallback":bool}
review.json      {"verdict":"approve|request_changes|escalate","findings":[{"severity":"info|warn|block","text":str}],
                  "reviewer":{"model"}}
decision.json    {"gate","mode":"auto|review|escalate","decided_by":"rules|agent|reviewer|human:<name>",
                  "decided_utc":str,"decision":{...same shape as proposal.proposal...},"changes_vs_proposal":[...]}
```

Flow in `cellwarden.gates.run_gate(gate, run_dir, cfg)`:
1. Build `bundle.json` from stage outputs (gate-specific builder).
2. **Rules first**: the rules engine produces the baseline decision (e.g. S1c's QC decisions for G1). If `llm` is not configured, or the budget guard refuses, the baseline is the decision, `fallback: true`, and the gate goes to `review` mode.
3. Worker agent proposes; code validates (`policy_check`). Invalid proposals are rejected and the worker gets one retry with the violations; still invalid → baseline + `review`.
4. Reviewer agent (separate system prompt; never sees the worker's rationale) returns a verdict. `request_changes` → one revision round. `escalate` → human.
5. Mode from config `gates: {G1: auto|review|escalate}` (defaults: G1 auto, G2 review, G3 auto, G4 review). `auto` + approve → decision written and run continues. Otherwise `report.html` + the run stops with exit 3.
6. `cellwarden approve --run-dir D --stage Gx [--from-file edited.json] [--by NAME]` validates the (edited) decision against policy and writes `decision.json` with `decided_by: human:<name>`; `cellwarden resume` continues.

Every LLM call is appended to `llm/agent_log.jsonl`: `{ts, gate, role, model, prompt_sha256, input_sha256, tool_calls:[...], output, usage:{input_tokens,output_tokens,cached_tokens}, cost_gbp}`. **Replay:** if `llm.replay: true` and a log entry with the same `prompt_sha256`+`input_sha256` exists, reuse its output without a call.

## 3. LLM adapter (`python/cellwarden/llm/`)

```python
class Provider:                      # cellwarden.llm.get_provider(cfg_llm, role) -> Provider
    def complete(self, *, system: str, messages: list[dict], tools: list[dict] | None,
                 max_tokens: int, json_schema: dict | None = None) -> Response
# Response: .text, .tool_calls [{id,name,arguments}], .usage {input_tokens,output_tokens,cached_tokens}, .stop_reason
```
Providers: `anthropic` (via `requests` to the Messages API, header `anthropic-version: 2023-06-01`, prompt caching on the system block), `openai_compatible` (chat/completions with `base_url`; covers vLLM/Ollama), `mock` (tests: scripted responses, no network). `bedrock`/`vertex` are stubs that raise `NotImplementedError` with a clear message. The API key is read from `api_key_env`; never logged.

Budget ledger `cellwarden.llm.Ledger(path)`: `charge(role, model, usage)` using a price table (`llm/prices.yaml`: USD per MTok; GBP via `llm.budget.usd_to_gbp`, default 0.78); `would_exceed(est_input_tokens, est_output_tokens) -> bool`; totals per run and across runs (`total` ledger path from config, default `~/.cellwarden/ledger.json`). Hard stop: a call that would cross `per_run_limit` or `total_limit` is not made.

Config (`llm:` in config.yaml) as in DESIGN 4.8, plus `replay: bool`, `ledger_path`, `budget.usd_to_gbp`. Default model `claude-haiku-4-5`.

Agent base (`python/cellwarden/agents/base.py`): an agent = system prompt file (`python/cellwarden/agents/prompts/<role>.md`) + tool functions (plain Python, each validating its inputs) + a `run(bundle) -> proposal` loop with a max of 12 tool calls. Free-text fields from data (sample metadata) are wrapped as `<data>` and the system prompt says they are never instructions.

## 4. S3 evidence (`R/stages/s3_evidence.R`, `python/cellwarden/stages/s3_celltypist.py`)

Input: `S2/object.rds` (clusters in `seurat_clusters`). Outputs:
- `markers.tsv.gz`: `cluster, gene, avg_log2fc, pct_in, pct_out, p_adj, rank` — presto if installed, else `FindAllMarkers(..., logfc.threshold=0.25, min.pct=0.1)` on `data`; top 50 per cluster.
- `cluster_summary.json`: per cluster `{n_cells, pct_cells, per_sample:{sample:n}, median_genes, median_umis, median_pct_mito, top_markers:[20 genes]}`.
- `signatures.tsv.gz`: per cluster mean `AddModuleScore` for built-in sets in `R/stages/data/signatures.yaml` (at least: interferon, inflammation, proliferation, hypoxia, platelet, erythrocyte, and ~15 broad lineage sets: T, CD4, CD8, NK, B, plasma, mono CD14, mono CD16, cDC, pDC, neutrophil, mast, epithelial, endothelial, fibroblast, hepatocyte, neuron, glia).
- `celltypist.tsv.gz` (optional; skipped with a note if celltypist is not installed or has no models offline): `cell_id, predicted_label, majority_voting, conf_score`, model named in `config.annotation.celltypist_model` (default `Immune_All_Low.pkl`).
- `evidence_bundle.json`: everything above aggregated per cluster + `sample_metadata_hint` (the sample sheet's extra columns, values only), **no per-cell data**.

## 5. G2 context, G3 annotation (`python/cellwarden/agents/context.py`, `annotation.py`)

- Consensus annotation in code first (`python/cellwarden/annotation/consensus.py`): per cluster, weighted vote of signature scores (z-scored across clusters), CellTypist majority (if present) and top markers against `python/cellwarden/annotation/markers_human.yaml` (broad and fine labels with CL IDs, ≥ 40 cell types). Output `S3/annotation_consensus.tsv`: `cluster, label_broad, label_fine, cl_id, confidence (0-1), votes_json, status (consensus|low_confidence|conflict)`.
- G2 proposal: `{"tissue":{"label","uberon_id","confidence"},"disease_candidates":[{"label","efo_id","confidence","evidence"}],"celltypist_model":str}`. Open Targets `search` resolves EFO/UBERON IDs (`cellwarden.opentargets.search_entity`); offline → IDs `null`.
- G3 proposal: `{"clusters":[{"cluster","label_broad","label_fine","cl_id","confidence","status":"consensus|agent|review"}]}`. The agent may only change clusters with status `low_confidence|conflict`; code rejects any other change. Final `S3/annotation.tsv` is written from `gates/G3/decision.json` and added to the object as `cell_type_broad`, `cell_type_fine` by a small R step at the start of S4.

## 6. S4 pseudobulk DGE + Open Targets (`R/stages/s4_dge.R`, `python/cellwarden/opentargets/`)

Config:
```yaml
dge:
  sample_col: sample_id          # pseudobulk unit (replicate)
  condition_col: condition       # from samples.csv metadata
  group_by: cell_type_broad      # or cluster
  contrasts:
    - {name: disease_vs_control, numerator: disease, denominator: control, covariates: [batch]}
  min_cells_per_sample: 20
  min_replicates: 2
  method: deseq2                  # deseq2 | limma_voom (fallback when DESeq2 missing)
  padj: 0.05
  lfc: 0.5
opentargets:
  enabled: true
  max_genes_per_direction: 50
  cache_dir: refs/opentargets    # shared across runs; files never expire unless --refresh
```
- Pseudobulk via `AggregateExpression(group.by = c(group_by, sample_col))`; skip groups with < `min_replicates` per side and say so in `dge_summary.json`.
- `dge/<contrast>__<group>.tsv.gz`: `gene, baseMean, log2fc, lfcSE, pvalue, padj, direction (up|down|ns)`.
- `dge_summary.json`: per contrast × group: n_up, n_down, n_tested, skipped_reason, top 10 up/down.
- Open Targets client (`cellwarden.opentargets.client`): GraphQL at `https://api.platform.opentargets.org/api/v4/graphql`, `requests`, retry with backoff, **file cache** keyed by query hash, `offline` mode that only reads the cache. Functions: `target_by_symbol(symbol)` → `{ensembl_id, approved_symbol, tractability, known_drugs:[{drug, phase, disease, mechanism}], associated_diseases:[{efo_id, name, score}] (top 10)}`, `search_entity(text, entity)` → best hits with IDs. `opentargets_summary.json`: per contrast × group × direction: per-gene evidence (direction always from our data), plus counts. Tests use **recorded responses** in `tests/fixtures/opentargets/`, never the network.
- G4 proposal: `{"contrasts":[{"contrast","group","summary":str,"hypotheses":[{"gene","direction","evidence_ids":[...],"text"}]}]}`, every statement labelled hypothesis; reviewer checks that every cited ID exists in `opentargets_summary.json`.

## 7. S5 report (`python/cellwarden/report/`)

`report.html`: single self-contained file (inline CSS/JS, no CDN, SVG/PNG plots base64) built with jinja2: run summary, resources, QC (per-sample tables + decisions), clusters/annotation (UMAP coloured by cluster and by label, drawn from `S2` embeddings exported by a short R step to `S5/umap.tsv.gz`), markers, DGE tables, Open Targets evidence, gate decisions and LLM cost. `explorer.html`: the UMAP with a client-side gene/metadata colour picker using data embedded as JSON (cap 100k points, subsample above). Each gate's `report.html` reuses the same templates.

## 8. Portability (`python/cellwarden/site.py`, `conf/sites/`, `containers/`)

- `cellwarden init-site [--write conf/sites/<name>.yaml]`: detects scheduler (`sbatch`/`qsub`/`bsub`), GPU (`nvidia-smi`), container runtime (`apptainer`/`singularity`/`docker`), internet (HTTPS to the LLM endpoint and Open Targets), and drafts the site YAML of DESIGN 7.2. `config.yaml` key `site: conf/sites/iridisx.yaml` supplies defaults for `execution`, `resources`, `container`.
- `cellwarden doctor` extended: LLM key present + a 1-token tool-use test (skipped with a note when no key), Open Targets reachable, budget remaining, R/Python package table, `execution` mode supported, site file valid.
- `execution: slurm|pbs|lsf|sge`: `cellwarden run` writes a batch script from `conf/templates/<scheduler>.sh.j2` and submits one job that calls `cellwarden run --execution direct`; `split` submits S0–S3/S4 as offline jobs and runs gates/Open Targets on the submitting node. Both only need to be exercised with a `mock` scheduler in tests.
- `containers/cellwarden.def` (Apptainer) and `Dockerfile`, same software list as DESIGN 7.5; not built in tests.
- `conf/sites/{iridisx,workstation,docker}.yaml`.

## 9. Tests

Unit tests next to existing ones (`tests/python`, `tests/R`), all offline: `mock` LLM provider with scripted tool calls, recorded Open Targets responses, the `stage_inputs` fixture. `tests/integration/run_full.sh` extends `run_stages.sh` through S5 with `llm.provider: mock` and `opentargets.offline: true`, and checks each gate's four files, the DGE tables, and that `report.html` opens (parses) and contains the expected sections.
