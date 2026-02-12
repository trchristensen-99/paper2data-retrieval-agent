from __future__ import annotations

import json

from agents import Agent, AgentOutputSchema, Runner

from src.schemas.models import PaperAnatomyOutput
from src.utils.config import MODELS
from src.utils.logging import log_event
from src.utils.retry import run_with_rate_limit_retry


ANATOMY_PROMPT = """Extract a structural map of the paper for downstream agents.
Return only concrete extracted values:
- sections found
- table labels/titles
- figure labels/titles
- all URLs
- accession-like identifiers (GSE/SRP/SRR/DOI-like data ids)
- PRISMA/scoping-flow counts when present

Do not interpret findings; this is structure-only context.
"""


anatomy_agent = Agent(
    name="anatomy_agent",
    model=MODELS.anatomy,
    instructions=ANATOMY_PROMPT,
    output_type=AgentOutputSchema(PaperAnatomyOutput, strict_json_schema=False),
)


async def run_anatomy_agent(paper_markdown: str) -> PaperAnatomyOutput:
    log_event("agent.anatomy.start", {"chars": len(paper_markdown)})
    result = await run_with_rate_limit_retry(lambda: Runner.run(anatomy_agent, input=paper_markdown))
    output = result.final_output
    if not isinstance(output, PaperAnatomyOutput):
        output = PaperAnatomyOutput.model_validate(output)
    log_event("agent.anatomy.end", json.loads(output.model_dump_json()))
    return output

