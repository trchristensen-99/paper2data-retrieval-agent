from __future__ import annotations

import json
from typing import Any

from agents import Agent, AgentOutputSchema, Runner

from src.schemas.models import (
    DataAvailabilityReport,
    MetadataRecord,
    MethodsSummary,
    QualityCheckOutput,
    ResultsSummary,
)
from src.utils.config import MODELS
from src.utils.logging import log_event
from src.utils.retry import run_with_rate_limit_retry

QUALITY_PROMPT = """You are a quality-control agent for scientific extraction outputs.
Your job is to detect missing key fields and suspiciously empty values.

Rules:
- Do not request retries for legitimately unavailable values that are clearly absent in the paper.
- Request retries when important fields are empty but evidence likely exists in the paper.
- Prefer targeted retries for one of: metadata, methods, results, data_availability.
- Keep retry reasons concise and actionable.
"""

quality_control_agent = Agent(
    name="quality_control_agent",
    model=MODELS.quality,
    instructions=QUALITY_PROMPT,
    output_type=AgentOutputSchema(QualityCheckOutput, strict_json_schema=False),
)

quality_repair_agent = Agent(
    name="quality_control_repair_agent",
    model=MODELS.quality,
    instructions=(
        QUALITY_PROMPT
        + "\nReturn JSON values only, never schema definitions."
    ),
)


def _sanitize_quality_payload(payload: dict[str, Any]) -> dict[str, Any]:
    missing = payload.get("missing_fields")
    if not isinstance(missing, list):
        missing = []
    suspicious = payload.get("suspicious_empty_fields")
    if not isinstance(suspicious, list):
        suspicious = []
    should_retry = bool(payload.get("should_retry", False))
    retry_instructions = payload.get("retry_instructions")
    if not isinstance(retry_instructions, list):
        retry_instructions = []
    notes = payload.get("notes")
    if not isinstance(notes, str):
        notes = "QC repair fallback used."
    return {
        "missing_fields": missing,
        "suspicious_empty_fields": suspicious,
        "should_retry": should_retry,
        "retry_instructions": retry_instructions,
        "notes": notes,
    }


async def run_quality_control_agent(
    paper_markdown: str,
    metadata: MetadataRecord,
    methods: MethodsSummary,
    results: ResultsSummary,
    data_availability: DataAvailabilityReport,
) -> QualityCheckOutput:
    log_event("agent.quality_control.start", {})
    payload = {
        "paper_markdown": paper_markdown,
        "current_output": {
            "metadata": metadata.model_dump(),
            "methods": methods.model_dump(),
            "results": results.model_dump(),
            "data_availability": data_availability.model_dump(),
        },
        "must_review_fields": [
            "metadata.title",
            "metadata.authors",
            "metadata.journal",
            "metadata.publication_date",
            "metadata.keywords",
            "metadata.paper_type",
            "metadata.license",
            "methods.experimental_design",
            "results.paper_type",
            "results.experimental_findings",
            "results.dataset_properties",
            "results.synthesized_claims",
            "results.method_benchmarks",
            "data_availability.overall_status",
            "data_availability.check_status",
        ],
    }

    try:
        result = await run_with_rate_limit_retry(
            lambda: Runner.run(quality_control_agent, input=json.dumps(payload))
        )
        output = result.final_output
        if not isinstance(output, QualityCheckOutput):
            output = QualityCheckOutput.model_validate(output)
    except Exception as exc:  # noqa: BLE001
        log_event("agent.quality_control.retry_on_parse_error", {"error": str(exc)})
        repair_result = await run_with_rate_limit_retry(
            lambda: Runner.run(quality_repair_agent, input=json.dumps(payload))
        )
        raw = repair_result.final_output
        if isinstance(raw, QualityCheckOutput):
            output = raw
        elif isinstance(raw, dict):
            output = QualityCheckOutput.model_validate(_sanitize_quality_payload(raw))
        else:
            parsed = json.loads(str(raw))
            if not isinstance(parsed, dict):
                parsed = {}
            output = QualityCheckOutput.model_validate(_sanitize_quality_payload(parsed))
    log_event("agent.quality_control.end", output.model_dump())
    return output
