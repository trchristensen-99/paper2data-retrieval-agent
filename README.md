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
   - `uv run python -m main ../data_for_agents_example/data_for_retrieval_agent/s41597-021-00905-y.md`

## Architecture
- Manager agent orchestrates:
  1. Metadata Agent (`gpt-4.1-mini`)
  2. Methods Agent (`gpt-4.1`)
  3. Results Agent (`gpt-4.1`)
  4. Data Availability Agent (`gpt-4.1` + tools)
  5. Synthesis Agent (`gpt-4.1`)

All agent interactions are logged for debugging in output retrieval logs.
