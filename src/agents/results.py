from __future__ import annotations

import json
from typing import Any

from agents import Agent, AgentOutputSchema, Runner

from src.schemas.models import ResultsSummary
from src.utils.config import MODELS
from src.utils.logging import log_event
from src.utils.retry import run_with_rate_limit_retry

RESULTS_PROMPT = """You are the No-Spin Zone results extractor.
You must adapt extraction to paper type:
- experimental: populate `experimental_findings` (effect sizes/comparisons first), optional substantive `spin_assessment`.
- dataset_descriptor: populate `dataset_properties` (size, dimensions, coverage, temporal range, format, license if present).
- review/meta_analysis: populate `synthesized_claims` with evidence-oriented concise claims.
- methods: populate `method_benchmarks` with task/metric/value/baseline/context when available.
- commentary: keep concise `synthesized_claims` only.

Rules:
- No narrative interpretation or causal spin.
- Do not fabricate p-values/CIs/effect sizes when absent.
- Set `paper_type` in ResultsSummary to match the routed paper type.
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
    experimental = payload.get("experimental_findings")
    if not isinstance(experimental, list):
        experimental = payload.get("quantitative_findings")
    if not isinstance(experimental, list):
        experimental = []
    dataset_props = payload.get("dataset_properties")
    if not isinstance(dataset_props, list):
        dataset_props = []
    synthesized_claims = payload.get("synthesized_claims")
    if not isinstance(synthesized_claims, list):
        synthesized_claims = payload.get("qualitative_findings")
    if not isinstance(synthesized_claims, list):
        synthesized_claims = []
    method_benchmarks = payload.get("method_benchmarks")
    if not isinstance(method_benchmarks, list):
        method_benchmarks = []
    key_figures = payload.get("key_figures")
    if not isinstance(key_figures, list):
        key_figures = []
    spin_assessment = payload.get("spin_assessment")
    if not isinstance(spin_assessment, str) or not spin_assessment.strip():
        spin_assessment = None
    paper_type = payload.get("paper_type")
    if not isinstance(paper_type, str) or not paper_type.strip():
        paper_type = None

    return {
        "paper_type": paper_type,
        "experimental_findings": experimental,
        "dataset_properties": dataset_props,
        "synthesized_claims": _coerce_list_of_str(synthesized_claims),
        "method_benchmarks": method_benchmarks,
        "key_figures": key_figures,
        "spin_assessment": spin_assessment,
    }


async def run_results_agent(
    paper_markdown: str,
    paper_type: str | None = None,
    guidance: str | None = None,
) -> ResultsSummary:
    log_event("agent.results.start", {"chars": len(paper_markdown)})
    agent_input = paper_markdown
    if guidance:
        agent_input = (
            f"{paper_markdown}\n\n"
            f"[QUALITY_REPAIR_INSTRUCTION]\n{guidance}\n"
        )
    if paper_type:
        agent_input = (
            f"[PAPER_TYPE]\n{paper_type}\n\n"
            f"{agent_input}"
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
            "paper_type, experimental_findings, dataset_properties, synthesized_claims, "
            "method_benchmarks, key_figures, spin_assessment.\n"
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
    log_event(
        "agent.results.end",
        {
            "paper_type": output.paper_type,
            "experimental_findings": len(output.experimental_findings),
            "dataset_properties": len(output.dataset_properties),
            "synthesized_claims": len(output.synthesized_claims),
            "method_benchmarks": len(output.method_benchmarks),
        },
    )
    return output
