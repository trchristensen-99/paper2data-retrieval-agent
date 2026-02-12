from __future__ import annotations

from agents import Agent, AgentOutputSchema, Runner

from src.schemas.models import ResultsSummary
from src.utils.config import MODELS
from src.utils.logging import log_event
from src.utils.retry import run_with_rate_limit_retry

RESULTS_PROMPT = """You are the No-Spin Zone results extractor.
Extract quantitative findings as raw facts only.
Rules:
- Prioritize effect sizes and magnitude over p-values.
- Include confidence intervals when present.
- No narrative interpretation or causal spin.
- Keep each finding grounded in explicit numbers from the paper.
- Provide a confidence score [0,1] for each extracted finding.
"""

results_agent = Agent(
    name="results_agent",
    model=MODELS.results,
    instructions=RESULTS_PROMPT,
    output_type=AgentOutputSchema(ResultsSummary, strict_json_schema=False),
)


async def run_results_agent(paper_markdown: str) -> ResultsSummary:
    log_event("agent.results.start", {"chars": len(paper_markdown)})
    result = await run_with_rate_limit_retry(
        lambda: Runner.run(results_agent, input=paper_markdown)
    )
    output = result.final_output
    if not isinstance(output, ResultsSummary):
        output = ResultsSummary.model_validate(output)
    log_event("agent.results.end", {"quantitative_findings": len(output.quantitative_findings)})
    return output
