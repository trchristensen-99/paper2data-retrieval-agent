from __future__ import annotations

import json
from datetime import datetime

from agents import Agent, Runner

from src.schemas.models import PaperRecord, SynthesisInput, SynthesisOutput
from src.utils.config import AGENT_VERSION, MODELS
from src.utils.logging import events_as_markdown, log_event

SYNTHESIS_PROMPT = """Merge extracted outputs into one PaperRecord.
Generate:
1) A complete PaperRecord with extraction_confidence [0,1]
2) A concise, human-readable retrieval report markdown
3) A retrieval log markdown summary of data-access checks and extraction path
"""

synthesis_agent = Agent(
    name="synthesis_agent",
    model=MODELS.synthesis,
    instructions=SYNTHESIS_PROMPT,
    output_type=SynthesisOutput,
)


async def run_synthesis_agent(payload: SynthesisInput) -> SynthesisOutput:
    log_event("agent.synthesis.start", {"payload": "received"})
    result = await Runner.run(synthesis_agent, input=payload.model_dump_json(indent=2))
    output = result.final_output
    if not isinstance(output, SynthesisOutput):
        output = SynthesisOutput.model_validate(output)

    output.record.extraction_timestamp = datetime.utcnow().isoformat()
    output.record.agent_version = AGENT_VERSION

    if not output.retrieval_log_markdown.strip():
        output.retrieval_log_markdown = events_as_markdown()

    log_event(
        "agent.synthesis.end",
        {"extraction_confidence": output.record.extraction_confidence},
    )
    return output


def fallback_synthesis(payload: SynthesisInput) -> SynthesisOutput:
    """Fallback when synthesis model output fails validation."""
    record = PaperRecord(
        metadata=payload.metadata,
        methods=payload.methods,
        results=payload.results,
        data_accessions=payload.data_accessions,
        data_availability=payload.data_availability,
        extraction_timestamp=datetime.utcnow().isoformat(),
        extraction_confidence=0.6,
    )

    report = "\n".join(
        [
            "# Retrieval Report",
            "",
            f"## Title\n{payload.metadata.title}",
            "",
            "## Methods Snapshot",
            payload.methods.experimental_design,
            "",
            "## Key Quantitative Findings",
            *[
                f"- {f.claim} | {f.metric}: {f.value}"
                for f in payload.results.quantitative_findings
            ],
            "",
            "## Data Availability",
            f"Overall status: {payload.data_availability.overall_status}",
            "",
            "## Raw Record",
            "```json",
            json.dumps(record.model_dump(), indent=2),
            "```",
        ]
    )

    return SynthesisOutput(
        record=record,
        retrieval_report_markdown=report,
        retrieval_log_markdown=events_as_markdown(),
    )
