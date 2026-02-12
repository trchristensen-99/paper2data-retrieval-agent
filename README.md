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

## Architecture
- Manager agent orchestrates:
  1. Metadata Agent (`gpt-4.1-mini`)
  2. Methods Agent (`gpt-4.1`)
  3. Results Agent (`gpt-4.1`)
  4. Data Availability Agent (`gpt-4.1` + tools)
  5. Synthesis Agent (`gpt-4.1`)

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
- Compare baseline vs updated batch summaries:
  - `uv run python -m src.compare_batches --baseline <baseline_summary.json> --updated <updated_summary.json>`

When a new entry matches an existing paper (by DOI, PMID, or normalized title), records are harmonized using an AI merge agent with deterministic fallback rules.
