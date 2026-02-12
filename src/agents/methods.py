from __future__ import annotations

import json
from typing import Any

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

methods_repair_agent = Agent(
    name="methods_repair_agent",
    model=MODELS.methods,
    instructions=(
        METHODS_PROMPT
        + "\n\nReturn ONLY raw JSON values for MethodsSummary fields."
        + "\nDo NOT output JSON Schema descriptors like title/type/items/description."
    ),
)


def _coerce_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("value"), str):
            return value["value"]
    return str(value)


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, dict):
        items = value.get("items")
        if isinstance(items, list):
            return [str(x).strip() for x in items if str(x).strip()]
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            return [text.strip()]
    text = str(value).strip()
    return [text] if text else []


def _coerce_sample_sizes(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        cleaned = {
            str(k): v
            for k, v in value.items()
            if str(k).lower() not in {"title", "type", "description", "properties"}
        }
        return cleaned if cleaned else {}
    return {"notes": str(value)}


def _sanitize_methods_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "organisms": _coerce_list(payload.get("organisms")),
        "cell_types": _coerce_list(payload.get("cell_types")),
        "assay_types": _coerce_list(payload.get("assay_types")),
        "sample_sizes": _coerce_sample_sizes(payload.get("sample_sizes")),
        "statistical_tests": _coerce_list(payload.get("statistical_tests")),
        "experimental_design": _coerce_scalar(payload.get("experimental_design")),
        "methods_completeness": _coerce_scalar(payload.get("methods_completeness")),
    }


async def run_methods_agent(paper_markdown: str, guidance: str | None = None) -> MethodsSummary:
    log_event("agent.methods.start", {"chars": len(paper_markdown)})
    agent_input = paper_markdown
    if guidance:
        agent_input = (
            f"{paper_markdown}\n\n"
            f"[QUALITY_REPAIR_INSTRUCTION]\n{guidance}\n"
        )
    try:
        result = await run_with_rate_limit_retry(lambda: Runner.run(methods_agent, input=agent_input))
        output = result.final_output
        if not isinstance(output, MethodsSummary):
            output = MethodsSummary.model_validate(output)
    except Exception as exc:  # noqa: BLE001
        log_event("agent.methods.retry_on_parse_error", {"error": str(exc)})
        repair_input = (
            f"{agent_input}\n\n"
            "[FORMAT_FIX]\n"
            "Return JSON object with keys exactly:\n"
            "organisms, cell_types, assay_types, sample_sizes, statistical_tests, "
            "experimental_design, methods_completeness.\n"
            "Values must be raw data, not schema metadata.\n"
        )
        repair_result = await run_with_rate_limit_retry(
            lambda: Runner.run(methods_repair_agent, input=repair_input)
        )
        raw = repair_result.final_output
        if isinstance(raw, MethodsSummary):
            output = raw
        elif isinstance(raw, dict):
            output = MethodsSummary.model_validate(_sanitize_methods_payload(raw))
        else:
            parsed = json.loads(str(raw))
            if not isinstance(parsed, dict):
                raise ValueError("methods_repair_agent returned non-object JSON")
            output = MethodsSummary.model_validate(_sanitize_methods_payload(parsed))
    output.organisms = normalize_organism_entries(output.organisms)
    log_event("agent.methods.end", output.model_dump())
    return output
