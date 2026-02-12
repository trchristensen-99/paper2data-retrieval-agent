from __future__ import annotations

import json
from typing import Any

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

results_repair_agent = Agent(
    name="results_repair_agent",
    model=MODELS.results,
    instructions=(
        RESULTS_PROMPT
        + "\n\nReturn ONLY raw JSON values for ResultsSummary fields."
        + "\nDo NOT output JSON Schema descriptors like title/type/items/$defs/properties."
    ),
)


def _coerce_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _sanitize_results_payload(payload: dict[str, Any]) -> dict[str, Any]:
    quantitative = payload.get("quantitative_findings")
    if not isinstance(quantitative, list):
        quantitative = []
    key_figures = payload.get("key_figures")
    if not isinstance(key_figures, list):
        key_figures = []
    spin_assessment = payload.get("spin_assessment")
    if not isinstance(spin_assessment, str) or not spin_assessment.strip():
        spin_assessment = "not_assessed"

    return {
        "quantitative_findings": quantitative,
        "qualitative_findings": _coerce_list_of_str(payload.get("qualitative_findings")),
        "key_figures": key_figures,
        "spin_assessment": spin_assessment,
    }


async def run_results_agent(paper_markdown: str, guidance: str | None = None) -> ResultsSummary:
    log_event("agent.results.start", {"chars": len(paper_markdown)})
    agent_input = paper_markdown
    if guidance:
        agent_input = (
            f"{paper_markdown}\n\n"
            f"[QUALITY_REPAIR_INSTRUCTION]\n{guidance}\n"
        )
    try:
        result = await run_with_rate_limit_retry(lambda: Runner.run(results_agent, input=agent_input))
        output = result.final_output
        if not isinstance(output, ResultsSummary):
            output = ResultsSummary.model_validate(output)
    except Exception as exc:  # noqa: BLE001
        log_event("agent.results.retry_on_parse_error", {"error": str(exc)})
        repair_input = (
            f"{agent_input}\n\n"
            "[FORMAT_FIX]\n"
            "Return JSON object with keys exactly:\n"
            "quantitative_findings, qualitative_findings, key_figures, spin_assessment.\n"
            "Values must be raw data, not schema metadata.\n"
        )
        repair_result = await run_with_rate_limit_retry(
            lambda: Runner.run(results_repair_agent, input=repair_input)
        )
        raw = repair_result.final_output
        if isinstance(raw, ResultsSummary):
            output = raw
        elif isinstance(raw, dict):
            output = ResultsSummary.model_validate(_sanitize_results_payload(raw))
        else:
            parsed = json.loads(str(raw))
            if not isinstance(parsed, dict):
                raise ValueError("results_repair_agent returned non-object JSON")
            output = ResultsSummary.model_validate(_sanitize_results_payload(parsed))
    log_event("agent.results.end", {"quantitative_findings": len(output.quantitative_findings)})
    return output
