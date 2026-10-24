# Security and data

## What leaves your machine
- **LLM review agents** (only when an API key or a model server is configured): they receive
  aggregated evidence only, such as QC summaries per sample, cluster sizes and marker genes, DE
  gene lists, and free-text sample metadata, marked as data. The count matrix and individual
  cells are never sent. Every call is logged in `<run>/llm/agent_log.jsonl`. With
  `--llm none`, or a model server you run yourself, nothing is sent to an outside service.
- **Open Targets** (S4): gene symbols and Ensembl IDs of the top DE genes are looked up in the
  public Open Targets API. `opentargets.offline: true` uses the local cache only.
- **Nothing else:** no telemetry or usage reporting.

## Keys
API keys are read from environment variables (for example a private `.env` passed with
`--env-file`). They are never written into the run directory, the logs or the reports. The run
records only the variable name.

## Reporting a vulnerability
Please don't open a public issue. Use GitHub's private vulnerability reporting on this
repository (Security → Report a vulnerability), or email the maintainers in AUTHORS.
