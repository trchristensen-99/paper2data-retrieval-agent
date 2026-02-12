from __future__ import annotations

from agents import Agent, AgentOutputSchema, Runner

from src.schemas.models import MetadataRecord
from src.utils.config import MODELS
from src.utils.logging import log_event
from src.utils.retry import run_with_rate_limit_retry

METADATA_PROMPT = """Extract publication metadata from the scientific paper markdown.
Return only fields that are explicitly supported by the text.
Leave optional values null when unavailable.
"""

metadata_agent = Agent(
    name="metadata_agent",
    model=MODELS.metadata,
    instructions=METADATA_PROMPT,
    output_type=AgentOutputSchema(MetadataRecord, strict_json_schema=False),
)


async def run_metadata_agent(paper_markdown: str) -> MetadataRecord:
    log_event("agent.metadata.start", {"chars": len(paper_markdown)})
    result = await run_with_rate_limit_retry(
        lambda: Runner.run(metadata_agent, input=paper_markdown)
    )
    output = result.final_output
    if not isinstance(output, MetadataRecord):
        output = MetadataRecord.model_validate(output)
    log_event("agent.metadata.end", output.model_dump())
    return output
