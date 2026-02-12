from __future__ import annotations

import json

from agents import Agent, AgentOutputSchema, Runner

from src.schemas.models import MetadataEnrichmentOutput, MetadataRecord
from src.tools.biblio_tools import search_crossref_by_title, search_pubmed
from src.utils.config import MODELS
from src.utils.logging import log_event
from src.utils.retry import run_with_rate_limit_retry

ENRICHMENT_PROMPT = """Use tool calls to enrich missing bibliographic metadata.
Primary targets: DOI, PMID, and publication venue (journal/preprint/source website).

Rules:
- Prefer exact title matches.
- Trust Crossref venue only when `is_high_confidence_match=true` (or exact DOI match).
- If Crossref/PubMed evidence conflicts with extracted journal, prefer evidence-backed venue.
- If uncertain, leave fields null and explain uncertainty in notes.
- Do not overwrite high-confidence existing values; only return inferred additions.
"""

metadata_enrichment_agent = Agent(
    name="metadata_enrichment_agent",
    model=MODELS.metadata_enrichment,
    instructions=ENRICHMENT_PROMPT,
    tools=[search_crossref_by_title, search_pubmed],
    output_type=AgentOutputSchema(MetadataEnrichmentOutput, strict_json_schema=False),
)


async def run_metadata_enrichment_agent(
    metadata: MetadataRecord,
    paper_markdown: str,
) -> MetadataEnrichmentOutput:
    log_event("agent.metadata_enrichment.start", {"title": metadata.title})
    payload = {
        "metadata": metadata.model_dump(),
        "paper_title": metadata.title,
        "paper_excerpt": paper_markdown[:5000],
        "task": "Find missing DOI/PMID/journal/publication_date using tools. Resolve venue if extracted value is suspicious.",
    }
    result = await run_with_rate_limit_retry(
        lambda: Runner.run(metadata_enrichment_agent, input=json.dumps(payload))
    )
    output = result.final_output
    if not isinstance(output, MetadataEnrichmentOutput):
        output = MetadataEnrichmentOutput.model_validate(output)
    log_event("agent.metadata_enrichment.end", output.model_dump())
    return output
