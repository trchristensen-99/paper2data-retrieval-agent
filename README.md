# Paper2Data Retrieval Agent

Multi-agent scientific paper retrieval pipeline for the Paper2Data project.

## What it produces
- Structured JSON record (`structured_record.json`)
- Human-readable markdown report (`retrieval_report.md`)
- Retrieval log (`retrieval_log.md`)

## Quickstart
1. Ensure `OPENAI_API_KEY` is set.
2. Install dependencies with `uv`:
   - `uv sync`
3. Run pipeline:
   - `uv run python -m src.main ../data_for_agents_example/data_for_retrieval_agent/s41597-021-00905-y.md`
   - optional faster run: `uv run python -m src.main ../data_for_agents_example/data_for_retrieval_agent/s41597-021-00905-y.md --fast`
   - strict network-gated run: `uv run python -m src.main ../data_for_agents_example/data_for_retrieval_agent/s41597-021-00905-y.md --strict-network`

4. Run external service preflight (recommended before long runs):
   - `uv run python -m src.network_preflight`
   - strict exit code for CI/debugging: `uv run python -m src.network_preflight --strict --json`

If runs fail with `APIConnectionError` / `nodename nor servname provided`:
- Check DNS: `nslookup api.openai.com`
- Ensure VPN/proxy/firewall is not blocking API access.
- On macOS, set DNS servers (Network Settings) to `1.1.1.1` and `8.8.8.8`, then retry.
- Configure proxy if required by your network:
  - `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` (standard)
  - or `P2D_HTTP_PROXY` in `.env`

## Model Configuration
- Default model for all agents is `gpt-5-mini`.
- Override globally:
  - `P2D_DEFAULT_MODEL=<model_name>`
- Override individual agents if needed:
  - `P2D_MODEL_METADATA`
  - `P2D_MODEL_METHODS`
  - `P2D_MODEL_RESULTS`
  - `P2D_MODEL_DATA_AVAILABILITY`
  - `P2D_MODEL_QUALITY`
  - `P2D_MODEL_METADATA_ENRICHMENT`
  - `P2D_MODEL_SYNTHESIS`
  - `P2D_MODEL_MANAGER`
  - `P2D_MODEL_HARMONIZER`

## Architecture
- Manager agent orchestrates:
  1. Metadata Agent (`gpt-5-mini` by default)
  2. Methods Agent (`gpt-5-mini` by default)
  3. Results Agent (`gpt-5-mini` by default)
  4. Data Availability Agent (`gpt-5-mini` by default + tools)
  5. Synthesis Agent (`gpt-5-mini` by default)

All agent interactions are logged for debugging in output retrieval logs.

## Queryable Database
You can ingest extraction outputs into a persistent SQLite database that supports search and record harmonization.

- Initialize DB (creates it if missing):
  - `uv run python -m src.database_cli --db outputs/paper_terminal.db init`
- Ingest one file or a directory of extraction runs:
  - `uv run python -m src.database_cli --db outputs/paper_terminal.db ingest --input outputs`
- Search:
  - `uv run python -m src.database_cli --db outputs/paper_terminal.db query --q "wildfire figshare"`
- Inspect one record:
  - `uv run python -m src.database_cli --db outputs/paper_terminal.db show --paper-id <paper_id>`
- DB stats:
  - `uv run python -m src.database_cli --db outputs/paper_terminal.db stats`
- Review and selectively update one existing DB entry against source markdown:
  - `uv run python -m src.database_cli --db outputs/paper_terminal.db review-update --paper-id <paper_id> --paper-markdown <path/to/paper.md> --sections metadata,methods,results`
- List suspect metadata rows and optionally print review-update commands:
  - `uv run python -m src.database_cli --db outputs/paper_terminal.db suspect-metadata --limit 100`
  - `uv run python -m src.database_cli --db outputs/paper_terminal.db suspect-metadata --print-review-commands --paper-root ../data_for_agents_example/data30_final`
- Local web UI (summary + search/filter + full record JSON viewer):
  - `uv run python -m src.web_app --db outputs/paper_terminal.db --host 127.0.0.1 --port 8080`
  - Open `http://127.0.0.1:8080`
  - Includes sortable columns and optional advanced filters for category, subcategory, journal/source venue, repository, data availability status, assay type, organism, and confidence threshold.
- Compare baseline vs updated batch summaries:
  - `uv run python -m src.compare_batches --baseline <baseline_summary.json> --updated <updated_summary.json>`
  - optional faster batch run: `uv run python -m src.batch_run --input-root ../data_for_agents_example/data30_final --output-root outputs/reextract_fast --fast`
  - strict network-gated batch run:
    - `uv run python -m src.batch_run --input-root ../data_for_agents_example/data30_final --output-root outputs/reextract --strict-network`

When a new entry matches an existing paper (by DOI, PMID, or normalized title), records are harmonized using an AI merge agent with deterministic fallback rules.
- Each canonical paper row represents one paper source (`source_count=1`); consolidated ingestion history is tracked in `paper_versions` (not shown in the main table).
- If journal is missing, the DB stores a source venue fallback (e.g., arXiv/bioRxiv/domain/DOI).

## Taxonomy and Organism Formatting
- Metadata now uses a fixed category/subcategory taxonomy:
  - `biology`, `computational`, `environmental`, `clinical`, `general_science`
- Methods organism formatting is normalized to:
  - `Latin name [common: common name]`
- Very high-species-count studies are compacted with:
  - `MULTI_SPECIES(total=<n>)`

## Confidence Metric
- `extraction_confidence` is computed deterministically from:
  - metadata coverage (title/authors/DOI-PMID/venue),
  - methods coverage (design/assay/sample size/stat tests),
  - results coverage (paper-type-aware completeness + overinterpretation penalty),
  - data-access verification status,
  - QC penalties for missing/suspicious empty fields.
- It is no longer purely model self-reported.
