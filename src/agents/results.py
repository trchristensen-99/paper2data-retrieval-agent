from __future__ import annotations

import json
from typing import Any

from agents import Agent, AgentOutputSchema, Runner

from src.schemas.models import ResultsSummary
from src.utils.config import MODELS
from src.utils.logging import log_event
from src.utils.retry import run_with_rate_limit_retry

RESULTS_PROMPT = """You are the No-Spin Zone results extractor.
You must adapt extraction to paper type:
- experimental: populate `experimental_findings` (effect sizes/comparisons first).
- dataset_descriptor: populate `dataset_properties` (size, dimensions, coverage, temporal range, format, license if present).
  Also populate `dataset_profile` with structured fields and `tables_extracted` for key tables.
  For `tables_extracted`, include row-level `data` objects for SQL ingestion when possible.
- review/meta_analysis: populate `synthesized_claims` with evidence-oriented concise claims.
  If scoping review / PRISMA-like flow is present, capture those counts in `dataset_profile.prisma_flow`.
- methods: populate `method_benchmarks` with task/metric/value/baseline/context when available.
- commentary: keep concise `synthesized_claims` only.

Rules:
- No narrative interpretation or causal spin.
- Do not fabricate p-values/CIs/effect sizes when absent.
- Add `provenance` for quantitative extractions whenever possible (text snippet + location hint).
- Set `paper_type` in ResultsSummary to match the routed paper type.
"""

results_agent = Agent(
    name="results_agent",
    model=MODELS.results,
    instructions=RESULTS_PROMPT,
    output_type=AgentOutputSchema(ResultsSummary, strict_json_schema=False),
)

results_repair_agent = Agent(
    name="results_repair_agent",
    model=MODELS.results,
    instructions=(
        RESULTS_PROMPT
        + "\n\nReturn ONLY raw JSON values for ResultsSummary fields."
        + "\nDo NOT output JSON Schema descriptors like title/type/items/$defs/properties."
    ),
)


def _coerce_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _sanitize_results_payload(payload: dict[str, Any]) -> dict[str, Any]:
    experimental = payload.get("experimental_findings")
    if not isinstance(experimental, list):
        experimental = payload.get("quantitative_findings")
    if not isinstance(experimental, list):
        experimental = []
    dataset_props = payload.get("dataset_properties")
    if not isinstance(dataset_props, list):
        dataset_props = []
    dataset_profile = payload.get("dataset_profile")
    if not isinstance(dataset_profile, dict):
        dataset_profile = None
    synthesized_claims = payload.get("synthesized_claims")
    if not isinstance(synthesized_claims, list):
        synthesized_claims = payload.get("qualitative_findings")
    if not isinstance(synthesized_claims, list):
        synthesized_claims = []
    method_benchmarks = payload.get("method_benchmarks")
    if not isinstance(method_benchmarks, list):
        method_benchmarks = []
    tables_raw = payload.get("tables_extracted")
    tables_extracted: list[dict[str, Any]] = []
    if isinstance(tables_raw, list):
        for idx, item in enumerate(tables_raw, start=1):
            if isinstance(item, dict):
                table_id = item.get("table_id") or item.get("id") or item.get("table") or item.get("table_title") or f"Table {idx}"
                title = item.get("title") or item.get("table_title") or str(table_id)
                columns = item.get("columns") if isinstance(item.get("columns"), list) else []
                data = item.get("data") if isinstance(item.get("data"), list) else []
                clean_data: list[dict[str, str]] = []
                for row in data:
                    if not isinstance(row, dict):
                        continue
                    clean_row: dict[str, str] = {}
                    for k, v in row.items():
                        if isinstance(v, dict):
                            clean_row[str(k)] = json.dumps(v, ensure_ascii=True)
                        elif v is None:
                            clean_row[str(k)] = ""
                        else:
                            clean_row[str(k)] = str(v)
                    clean_data.append(clean_row)
                summary = item.get("summary") or item.get("description")
                key_content = item.get("key_content") if isinstance(item.get("key_content"), list) else []
                tables_extracted.append(
                    {
                        "table_id": str(table_id),
                        "title": str(title) if title is not None else None,
                        "columns": [str(x) for x in columns],
                        "data": clean_data,
                        "summary": str(summary) if isinstance(summary, str) else None,
                        "key_content": [str(x) for x in key_content if str(x).strip()],
                        "provenance": item.get("provenance"),
                    }
                )
            elif isinstance(item, str) and item.strip():
                tables_extracted.append(
                    {
                        "table_id": f"Table {idx}",
                        "title": item.strip(),
                        "columns": [],
                        "data": [],
                        "summary": None,
                        "key_content": [],
                    }
                )
    key_figures_raw = payload.get("key_figures")
    key_figures: list[dict[str, Any]] = []
    if isinstance(key_figures_raw, list):
        for idx, item in enumerate(key_figures_raw, start=1):
            if isinstance(item, dict):
                figure_id = item.get("figure_id") or item.get("id") or item.get("figure") or f"Figure {idx}"
                description = item.get("description") or item.get("summary") or item.get("context") or ""
                findings = item.get("key_findings")
                if not isinstance(findings, list):
                    findings = []
                key_figures.append(
                    {
                        "figure_id": str(figure_id),
                        "description": str(description),
                        "key_findings": [str(x) for x in findings if str(x).strip()],
                    }
                )
            elif isinstance(item, str) and item.strip():
                key_figures.append(
                    {
                        "figure_id": f"Figure {idx}",
                        "description": item.strip(),
                        "key_findings": [],
                    }
                )
    paper_type = payload.get("paper_type")
    if not isinstance(paper_type, str) or not paper_type.strip():
        paper_type = None
    findings_block = payload.get("findings")
    if not isinstance(findings_block, dict):
        findings_block = None

    return {
        "paper_type": paper_type,
        "experimental_findings": experimental,
        "dataset_properties": dataset_props,
        "dataset_profile": dataset_profile,
        "synthesized_claims": _coerce_list_of_str(synthesized_claims),
        "method_benchmarks": method_benchmarks,
        "tables_extracted": tables_extracted,
        "key_figures": key_figures,
        "findings": findings_block,
    }


async def run_results_agent(
    paper_markdown: str,
    paper_type: str | None = None,
    guidance: str | None = None,
) -> ResultsSummary:
    log_event("agent.results.start", {"chars": len(paper_markdown)})
    agent_input = paper_markdown
    if guidance:
        agent_input = (
            f"{paper_markdown}\n\n"
            f"[QUALITY_REPAIR_INSTRUCTION]\n{guidance}\n"
        )
    if paper_type:
        agent_input = (
            f"[PAPER_TYPE]\n{paper_type}\n\n"
            f"{agent_input}"
        )
    try:
        result = await run_with_rate_limit_retry(lambda: Runner.run(results_agent, input=agent_input))
        output = result.final_output
        if not isinstance(output, ResultsSummary):
            output = ResultsSummary.model_validate(output)
    except Exception as exc:  # noqa: BLE001
        log_event("agent.results.retry_on_parse_error", {"error": str(exc)})
        repair_input = (
            f"{agent_input}\n\n"
            "[FORMAT_FIX]\n"
            "Return JSON object with keys exactly:\n"
            "paper_type, experimental_findings, dataset_properties, synthesized_claims, "
            "method_benchmarks, dataset_profile, tables_extracted, key_figures, findings.\n"
            "Values must be raw data, not schema metadata.\n"
        )
        repair_result = await run_with_rate_limit_retry(
            lambda: Runner.run(results_repair_agent, input=repair_input)
        )
        raw = repair_result.final_output
        if isinstance(raw, ResultsSummary):
            output = raw
        elif isinstance(raw, dict):
            output = ResultsSummary.model_validate(_sanitize_results_payload(raw))
        else:
            parsed = json.loads(str(raw))
            if not isinstance(parsed, dict):
                raise ValueError("results_repair_agent returned non-object JSON")
            output = ResultsSummary.model_validate(_sanitize_results_payload(parsed))
    log_event(
        "agent.results.end",
        {
            "paper_type": output.paper_type,
            "experimental_findings": len(output.experimental_findings),
            "dataset_properties": len(output.dataset_properties),
            "tables_extracted": len(output.tables_extracted),
            "synthesized_claims": len(output.synthesized_claims),
            "method_benchmarks": len(output.method_benchmarks),
        },
    )
    return output
