from __future__ import annotations

import json

from agents import Agent, AgentOutputSchema, Runner

from src.schemas.models import MetadataRecord
from src.tools.biblio_tools import search_crossref_by_doi, search_crossref_by_title, search_pubmed
from src.utils.config import MODELS
from src.utils.logging import log_event
from src.utils.retry import run_with_rate_limit_retry
from src.utils.taxonomy import all_categories_text

METADATA_PROMPT = """Extract publication metadata from the scientific paper markdown.
Return only fields that are explicitly supported by the text.
Leave optional values null when unavailable.

Critical:
- For `journal`, extract the true publication venue (journal, preprint server, or source website).
- Do NOT default to "Scientific Data" unless explicitly supported by the paper or tool evidence.
- If venue is unclear from markdown, call tools to validate DOI/PMID and venue.
- Return `keywords` as a list (possibly empty), never null.

Category taxonomy (must use only these values):
""" + all_categories_text() + """
"""

metadata_agent = Agent(
    name="metadata_agent",
    model=MODELS.metadata,
    instructions=METADATA_PROMPT,
    tools=[search_crossref_by_doi, search_crossref_by_title, search_pubmed],
    output_type=AgentOutputSchema(MetadataRecord, strict_json_schema=False),
)


async def run_metadata_agent(paper_markdown: str, guidance: str | None = None) -> MetadataRecord:
    log_event("agent.metadata.start", {"chars": len(paper_markdown)})
    agent_input = paper_markdown
    if guidance:
        agent_input = (
            f"{paper_markdown}\n\n"
            f"[QUALITY_REPAIR_INSTRUCTION]\n{guidance}\n"
        )
    result = await run_with_rate_limit_retry(lambda: Runner.run(metadata_agent, input=agent_input))
    output = result.final_output
    if not isinstance(output, MetadataRecord):
        output = MetadataRecord.model_validate(output)
    log_event("agent.metadata.output", json.loads(output.model_dump_json()))
    log_event("agent.metadata.end", output.model_dump())
    return output
