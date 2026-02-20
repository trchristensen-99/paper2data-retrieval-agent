from __future__ import annotations

from pydantic import BaseModel, Field
from agents import Agent, AgentOutputSchema, Runner

from src.utils.config import MODELS
from src.utils.logging import log_event
from src.utils.retry import run_with_rate_limit_retry


class ArchetypeOutput(BaseModel):
    paper_archetype: str = Field(
        description="dataset|methodology|clinical_trial|review|experimental_study|software_benchmark|commentary"
    )
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


ARCHETYPE_PROMPT = """Classify the paper into one archetype:
- dataset
- methodology
- clinical_trial
- review
- experimental_study
- software_benchmark
- commentary

Return only archetype classification with concise rationale.
"""


archetype_agent = Agent(
    name="archetype_agent",
    model=MODELS.archetype,
    instructions=ARCHETYPE_PROMPT,
    output_type=AgentOutputSchema(ArchetypeOutput, strict_json_schema=False),
)


async def run_archetype_agent(paper_markdown: str) -> ArchetypeOutput:
    log_event("agent.archetype.start", {"chars": len(paper_markdown)})
    result = await run_with_rate_limit_retry(lambda: Runner.run(archetype_agent, input=paper_markdown))
    output = result.final_output
    if not isinstance(output, ArchetypeOutput):
        output = ArchetypeOutput.model_validate(output)
    log_event("agent.archetype.end", output.model_dump())
    return output

