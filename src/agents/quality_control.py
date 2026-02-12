from __future__ import annotations

import json

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
            "methods.experimental_design",
            "results.quantitative_findings",
            "results.spin_assessment",
            "data_availability.overall_status",
        ],
    }

    result = await run_with_rate_limit_retry(
        lambda: Runner.run(quality_control_agent, input=json.dumps(payload))
    )
    output = result.final_output
    if not isinstance(output, QualityCheckOutput):
        output = QualityCheckOutput.model_validate(output)
    log_event("agent.quality_control.end", output.model_dump())
    return output
