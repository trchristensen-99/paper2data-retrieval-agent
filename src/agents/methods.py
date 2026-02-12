from __future__ import annotations

from agents import Agent, AgentOutputSchema, Runner

from src.schemas.models import MethodsSummary
from src.utils.config import MODELS
from src.utils.logging import log_event

METHODS_PROMPT = """Extract methods details needed for statistical assessment:
- organisms, cell types, assay types
- sample sizes
- statistical tests used
- concise but faithful experimental design summary
- methods completeness assessment for reproducibility

Do not speculate. Mark missing details explicitly in summary text.
"""

methods_agent = Agent(
    name="methods_agent",
    model=MODELS.methods,
    instructions=METHODS_PROMPT,
    output_type=AgentOutputSchema(MethodsSummary, strict_json_schema=False),
)


async def run_methods_agent(paper_markdown: str) -> MethodsSummary:
    log_event("agent.methods.start", {"chars": len(paper_markdown)})
    result = await Runner.run(methods_agent, input=paper_markdown)
    output = result.final_output
    if not isinstance(output, MethodsSummary):
        output = MethodsSummary.model_validate(output)
    log_event("agent.methods.end", output.model_dump())
    return output
