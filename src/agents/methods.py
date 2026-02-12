from __future__ import annotations

from agents import Agent, AgentOutputSchema, Runner

from src.schemas.models import MethodsSummary
from src.utils.config import MODELS
from src.utils.logging import log_event
from src.utils.organisms import normalize_organism_entries
from src.utils.retry import run_with_rate_limit_retry

METHODS_PROMPT = """Extract methods details needed for statistical assessment:
- organisms, cell types, assay types
- sample sizes
- statistical tests used
- concise but faithful experimental design summary
- methods completeness assessment for reproducibility

Do not speculate. Mark missing details explicitly in summary text.

Organism formatting requirement:
- Use "Latin name [common: common name]" whenever possible.
- For very broad surveys with many species, return representative named species plus one marker:
  "MULTI_SPECIES(total=<n>)".
"""

methods_agent = Agent(
    name="methods_agent",
    model=MODELS.methods,
    instructions=METHODS_PROMPT,
    output_type=AgentOutputSchema(MethodsSummary, strict_json_schema=False),
)


async def run_methods_agent(paper_markdown: str, guidance: str | None = None) -> MethodsSummary:
    log_event("agent.methods.start", {"chars": len(paper_markdown)})
    agent_input = paper_markdown
    if guidance:
        agent_input = (
            f"{paper_markdown}\n\n"
            f"[QUALITY_REPAIR_INSTRUCTION]\n{guidance}\n"
        )
    result = await run_with_rate_limit_retry(
        lambda: Runner.run(methods_agent, input=agent_input)
    )
    output = result.final_output
    if not isinstance(output, MethodsSummary):
        output = MethodsSummary.model_validate(output)
    output.organisms = normalize_organism_entries(output.organisms)
    log_event("agent.methods.end", output.model_dump())
    return output
