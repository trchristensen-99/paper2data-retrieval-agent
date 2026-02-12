from __future__ import annotations

from agents import Agent, Runner

from src.schemas.models import MetadataRecord
from src.utils.config import MODELS
from src.utils.logging import log_event

METADATA_PROMPT = """Extract publication metadata from the scientific paper markdown.
Return only fields that are explicitly supported by the text.
Leave optional values null when unavailable.
"""

metadata_agent = Agent(
    name="metadata_agent",
    model=MODELS.metadata,
    instructions=METADATA_PROMPT,
    output_type=MetadataRecord,
)


async def run_metadata_agent(paper_markdown: str) -> MetadataRecord:
    log_event("agent.metadata.start", {"chars": len(paper_markdown)})
    result = await Runner.run(metadata_agent, input=paper_markdown)
    output = result.final_output
    if not isinstance(output, MetadataRecord):
        output = MetadataRecord.model_validate(output)
    log_event("agent.metadata.end", output.model_dump())
    return output
